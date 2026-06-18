"""Load and filter saved training runs from the runs/ directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / 'runs'

ALL_FILTER = 'All'


@dataclass
class RunSummary:
    run_id: str
    path: Path
    label: str
    product: str
    location: str
    timesteps: int
    total_cost: float
    service_level: float
    started_at: str
    has_records: bool


@dataclass
class LoadedRun:
    summary: RunSummary
    config: dict
    records: list
    future_records: list
    lead_time: int


def _parse_started_at(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.min


def _format_started_at(value: str) -> str:
    dt = _parse_started_at(value)
    if dt == datetime.min:
        return value or 'unknown'
    return dt.strftime('%Y-%m-%d %H:%M')


def _shorten(text: str, max_len: int = 28) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + '…'


def _make_label(run_id: str, config: dict, has_records: bool) -> str:
    started = _format_started_at(config.get('started_at', ''))
    timesteps = int(config.get('timesteps', 0))
    total_cost = float(config.get('total_cost', 0))
    product = _shorten(str(config.get('product', '')))
    location = _shorten(str(config.get('location', '')))
    suffix = '' if has_records else ' [no data]'
    return (
        f'{started} · {timesteps:,} steps · €{total_cost:,.0f} · '
        f'{product} / {location}{suffix}'
    )


def list_runs(runs_dir: Path | None = None) -> list[RunSummary]:
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    if not base.exists():
        return []

    summaries: list[RunSummary] = []
    for run_path in sorted(base.iterdir()):
        if not run_path.is_dir():
            continue
        config_path = run_path / 'config.json'
        if not config_path.exists():
            continue
        try:
            with open(config_path, encoding='utf-8') as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        has_records = (run_path / 'records.json').exists()
        run_id = run_path.name
        summaries.append(RunSummary(
            run_id=run_id,
            path=run_path,
            label=_make_label(run_id, config, has_records),
            product=str(config.get('product', '')),
            location=str(config.get('location', '')),
            timesteps=int(config.get('timesteps', 0)),
            total_cost=float(config.get('total_cost', 0)),
            service_level=float(config.get('service_level', 0)),
            started_at=str(config.get('started_at', '')),
            has_records=has_records,
        ))

    summaries.sort(key=lambda s: _parse_started_at(s.started_at), reverse=True)
    return summaries


def load_run(run_path: Path | str) -> LoadedRun:
    run_path = Path(run_path)
    config_path = run_path / 'config.json'
    records_path = run_path / 'records.json'

    if not config_path.exists():
        raise FileNotFoundError(f'Missing config.json in {run_path}')
    if not records_path.exists():
        raise FileNotFoundError(f'Missing records.json in {run_path}')

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    with open(records_path, encoding='utf-8') as f:
        payload = json.load(f)

    records = payload.get('records', [])
    future_records = payload.get('future_records', [])
    if not records:
        raise ValueError(f'No historical records in {records_path}')

    summaries = [s for s in list_runs(run_path.parent) if s.run_id == run_path.name]
    summary = summaries[0] if summaries else RunSummary(
        run_id=run_path.name,
        path=run_path,
        label=_make_label(run_path.name, config, True),
        product=str(config.get('product', '')),
        location=str(config.get('location', '')),
        timesteps=int(config.get('timesteps', 0)),
        total_cost=float(config.get('total_cost', 0)),
        service_level=float(config.get('service_level', 0)),
        started_at=str(config.get('started_at', '')),
        has_records=True,
    )

    return LoadedRun(
        summary=summary,
        config=config,
        records=records,
        future_records=future_records,
        lead_time=int(payload.get('lead_time', config.get('lead_time', 0))),
    )


def filter_runs(
    summaries: list[RunSummary],
    product: str = ALL_FILTER,
    location: str = ALL_FILTER,
    timesteps_min: int | None = None,
    timesteps_max: int | None = None,
    require_records: bool = True,
) -> list[RunSummary]:
    out = summaries
    if require_records:
        out = [s for s in out if s.has_records]
    if product and product != ALL_FILTER:
        out = [s for s in out if s.product == product]
    if location and location != ALL_FILTER:
        out = [s for s in out if s.location == location]
    if timesteps_min is not None:
        out = [s for s in out if s.timesteps >= timesteps_min]
    if timesteps_max is not None:
        out = [s for s in out if s.timesteps <= timesteps_max]
    return out


def distinct_products(summaries: list[RunSummary]) -> list[str]:
    return sorted({s.product for s in summaries if s.product})


def distinct_locations(summaries: list[RunSummary], product: str = ALL_FILTER) -> list[str]:
    items = summaries
    if product != ALL_FILTER:
        items = [s for s in items if s.product == product]
    return sorted({s.location for s in items if s.location})
