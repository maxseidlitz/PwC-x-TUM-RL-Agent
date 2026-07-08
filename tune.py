"""
Hyperparameter search for the PPO inventory agent.

Usage
-----
  python tune.py
  python tune.py --out results.xlsx
  python tune.py --grid --out grid_results.xlsx

Two modes:

  1. Explicit combinations (PARAM_COMBOS)
     List each exact param dict you want to test. No Cartesian explosion.
     PARAM_COMBOS takes priority when non-empty.

  2. Full grid (PARAM_GRID)
     Use --grid, or leave PARAM_COMBOS empty and fill PARAM_GRID.

Results are written to an Excel workbook with two sheets:
  • Summary  – one row per run, all params + KPIs
  • Per-Week – flattened per-week records for every run
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory_ppo import (
    DEFAULT_FILE_PATH,
    TrainingConfig,
    compute_kpis,
    run_training_pipeline,
)


# ---------------------------------------------------------------------------
# Product-location pairs to evaluate
# Demand group is NOT written manually.
# It is detected automatically from the input data.
# ---------------------------------------------------------------------------

PRODUCT_LOCATIONS: list[tuple[str, str]] = [
    ("Ice Cream Strawberry Flavor", "Logistics Hub Lissabon"),
    ("Ice Cream Chocolate Flavor", "Logistics Hub Munich"),
    ("Ice Cream Mango Flavor", "Production Factory Berlin"),
    ("Ice Cream Mint Flavor", "Logistics Hub Madrid"),
    ("Ice Cream Pistaccio Flavor", "Logistics Hub Helsinki"),
]


# ---------------------------------------------------------------------------
# MODE 1 — Explicit combinations
#
# These are the manually selected configurations.
# The demand_group field is used only for filtering:
#   high product-location   -> H1-H5
#   medium product-location -> M1-M5
#   low product-location    -> L1-L5
#
# Metadata fields such as run_id, run_name, demand_group, key_change are
# written to Excel, but they are NOT passed into TrainingConfig.
# ---------------------------------------------------------------------------

PARAM_COMBOS: list[dict] = [
    # -----------------------------------------------------------------------
    # High Demand: mean > 100 units/week
    # -----------------------------------------------------------------------
    {
        "run_id": "H1",
        "run_name": "H-Baseline",
        "demand_group": "high",
        "key_change": "Reference — no changes",
        "timesteps": 10_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "H2",
        "run_name": "H-MoreTraining",
        "demand_group": "high",
        "key_change": "Timesteps",
        "timesteps": 100_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "H3",
        "run_name": "H-StableLR",
        "demand_group": "high",
        "key_change": "Learning rate",
        "timesteps": 100_000,
        "learning_rate": 3e-4,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "H4",
        "run_name": "H-LongHorizon",
        "demand_group": "high",
        "key_change": "Gamma, n_steps, batch size",
        "timesteps": 100_000,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "n_steps": 4096,
        "batch_size": 128,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "H5",
        "run_name": "H-FullConfig",
        "demand_group": "high",
        "key_change": "Timesteps, learning rate, gamma, n_steps, batch size",
        "timesteps": 150_000,
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "n_steps": 4096,
        "batch_size": 128,
        "n_forecast_weeks": 4,
    },

    # -----------------------------------------------------------------------
    # Medium Demand: 10 <= mean <= 100 units/week
    # -----------------------------------------------------------------------
    {
        "run_id": "M1",
        "run_name": "M-Baseline",
        "demand_group": "medium",
        "key_change": "Reference — no changes",
        "timesteps": 10_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "M2",
        "run_name": "M-MoreTraining",
        "demand_group": "medium",
        "key_change": "Timesteps",
        "timesteps": 50_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "M3",
        "run_name": "M-StableLR",
        "demand_group": "medium",
        "key_change": "Learning rate",
        "timesteps": 50_000,
        "learning_rate": 3e-4,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "M4",
        "run_name": "M-LargeRollout",
        "demand_group": "medium",
        "key_change": "n_steps, batch size",
        "timesteps": 50_000,
        "learning_rate": 3e-4,
        "gamma": 0.990,
        "n_steps": 4096,
        "batch_size": 128,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "M5",
        "run_name": "M-FullConfig",
        "demand_group": "medium",
        "key_change": "Timesteps, learning rate, n_steps, batch size",
        "timesteps": 100_000,
        "learning_rate": 3e-4,
        "gamma": 0.990,
        "n_steps": 4096,
        "batch_size": 128,
        "n_forecast_weeks": 4,
    },

    # -----------------------------------------------------------------------
    # Low Demand: mean < 10 units/week
    # -----------------------------------------------------------------------
    {
        "run_id": "L1",
        "run_name": "L-Baseline",
        "demand_group": "low",
        "key_change": "Reference — no changes",
        "timesteps": 10_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "L2",
        "run_name": "L-MoreTraining",
        "demand_group": "low",
        "key_change": "Timesteps",
        "timesteps": 50_000,
        "learning_rate": 1e-3,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "L3",
        "run_name": "L-ConservativeLR",
        "demand_group": "low",
        "key_change": "Learning rate",
        "timesteps": 50_000,
        "learning_rate": 1e-4,
        "gamma": 0.990,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "L4",
        "run_name": "L-ShortHorizon",
        "demand_group": "low",
        "key_change": "Learning rate, gamma",
        "timesteps": 50_000,
        "learning_rate": 3e-4,
        "gamma": 0.970,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
    {
        "run_id": "L5",
        "run_name": "L-FullConfig",
        "demand_group": "low",
        "key_change": "Timesteps, learning rate, gamma",
        "timesteps": 50_000,
        "learning_rate": 1e-4,
        "gamma": 0.970,
        "n_steps": 2048,
        "batch_size": 64,
        "n_forecast_weeks": 4,
    },
]


# ---------------------------------------------------------------------------
# MODE 2 — Full Cartesian grid
# Used only when PARAM_COMBOS is empty or when --grid is passed.
# ---------------------------------------------------------------------------

PARAM_GRID: dict[str, list] = {
    "timesteps": [20_000, 100_000],
    "learning_rate": [1e-4, 1e-3],
    "n_steps": [128, 512, 2048],
    "gamma": [0.99],
    "batch_size": [64],
    "n_forecast_weeks": [4],
}


# ---------------------------------------------------------------------------
# Fixed config fields
# These are not varied across the sweep.
# ---------------------------------------------------------------------------

FIXED: dict = {
    "file_path": DEFAULT_FILE_PATH,
    "scenarios": [],          # [] = all available scenarios
    "holding_cost": 13.0,
    "ordering_cost": 60.0,
    "lost_sales_cost": 2500.0,
    "max_order_qty": 0,       # 0 = auto-detect
    "verbose": 0,
}


# ---------------------------------------------------------------------------
# Demand group detection
# ---------------------------------------------------------------------------

LOW_DEMAND_MAX = 10.0
HIGH_DEMAND_MIN = 100.0

METADATA_KEYS = {"run_id", "run_name", "demand_group", "key_change"}


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _classify_demand_group(mean_weekly_demand: float | None) -> str:
    """
    Classification rule:
      mean > 100        -> high
      10 <= mean <= 100 -> medium
      mean < 10         -> low

    If demand cannot be detected, use medium as a safe fallback so the script
    does not crash.
    """
    if mean_weekly_demand is None:
        return "medium"

    if mean_weekly_demand > HIGH_DEMAND_MIN:
        return "high"

    if mean_weekly_demand >= LOW_DEMAND_MAX:
        return "medium"

    return "low"


def _normalize_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Try to find a column by exact normalized match first, then by partial match.
    This makes the script more robust to names such as:
      Product Name, product_name, PRODUCT, etc.
    """
    normalized_cols = {_normalize_name(col): col for col in df.columns}
    normalized_candidates = [_normalize_name(c) for c in candidates]

    for candidate in normalized_candidates:
        if candidate in normalized_cols:
            return normalized_cols[candidate]

    for norm_col, original_col in normalized_cols.items():
        for candidate in normalized_candidates:
            if candidate and candidate in norm_col:
                return original_col

    return None


def _match_text_column(series: pd.Series, target: str) -> pd.Series:
    normalized_target = _normalize_name(target)
    return series.astype(str).map(_normalize_name) == normalized_target


def _read_excel_sheets(file_path: str | Path):
    file_path = Path(file_path)

    if not file_path.exists():
        return

    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if not df.empty:
                yield sheet_name, df
        except Exception:
            continue


def _mean_weekly_demand(
    product: str,
    location: str,
    file_path: str | Path = DEFAULT_FILE_PATH,
) -> tuple[float | None, str]:
    """
    Detect mean weekly demand for one product-location pair.

    The function first tries to find explicit product/location/demand columns.
    If no demand column is found, it falls back to numeric columns that look
    demand-related. If that also fails, it falls back to other numeric columns
    while excluding obvious non-demand fields.

    It returns:
      mean_weekly_demand, detection_note
    """
    product_candidates = [
        "product",
        "product_name",
        "product name",
        "item",
        "item_name",
        "sku",
        "material",
    ]

    location_candidates = [
        "location",
        "location_name",
        "location name",
        "warehouse",
        "hub",
        "logistics hub",
        "site",
        "plant",
        "factory",
    ]

    demand_candidates = [
        "demand",
        "weekly_demand",
        "weekly demand",
        "mean_demand",
        "mean demand",
        "avg_demand",
        "avg demand",
        "forecast",
        "forecast_demand",
        "forecast demand",
        "consumption",
        "weekly_consumption",
        "weekly consumption",
        "sales",
        "units",
        "quantity",
        "qty",
    ]

    demand_like_tokens = [
        "demand",
        "forecast",
        "consumption",
        "sales",
        "unit",
        "quantity",
        "qty",
        "week",
    ]

    exclude_numeric_tokens = [
        "cost",
        "price",
        "lead",
        "stock",
        "inventory",
        "capacity",
        "id",
        "scenario",
        "service",
        "holding",
        "ordering",
        "lost",
        "safety",
        "min",
        "max",
    ]

    for sheet_name, df in _read_excel_sheets(file_path):
        product_col = _find_column(df, product_candidates)
        location_col = _find_column(df, location_candidates)

        if product_col is None or location_col is None:
            continue

        subset = df[
            _match_text_column(df[product_col], product)
            & _match_text_column(df[location_col], location)
        ]

        if subset.empty:
            continue

        demand_col = _find_column(subset, demand_candidates)

        if demand_col is not None:
            values = pd.to_numeric(subset[demand_col], errors="coerce").dropna()
            if not values.empty:
                note = f"sheet={sheet_name}; demand_col={demand_col}"
                return float(values.mean()), note

        # First fallback: numeric columns with demand-like names
        demand_like_cols = []
        for col in subset.columns:
            if col in {product_col, location_col}:
                continue

            norm_col = _normalize_name(col)
            if any(token in norm_col for token in demand_like_tokens):
                values = pd.to_numeric(subset[col], errors="coerce")
                if values.notna().any():
                    demand_like_cols.append(col)

        if demand_like_cols:
            values = (
                subset[demand_like_cols]
                .apply(pd.to_numeric, errors="coerce")
                .stack()
                .dropna()
            )
            if not values.empty:
                note = (
                    f"sheet={sheet_name}; fallback=demand_like_numeric_cols; "
                    f"cols={len(demand_like_cols)}"
                )
                return float(values.mean()), note

        # Second fallback: other numeric columns, excluding obvious non-demand fields
        numeric_cols = []
        for col in subset.columns:
            if col in {product_col, location_col}:
                continue

            norm_col = _normalize_name(col)
            if any(token in norm_col for token in exclude_numeric_tokens):
                continue

            values = pd.to_numeric(subset[col], errors="coerce")
            if values.notna().any():
                numeric_cols.append(col)

        if numeric_cols:
            values = (
                subset[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .stack()
                .dropna()
            )
            if not values.empty:
                note = (
                    f"sheet={sheet_name}; fallback=generic_numeric_cols; "
                    f"cols={len(numeric_cols)}"
                )
                return float(values.mean()), note

    return None, "Could not detect demand from Excel; fallback group is medium"


def _detect_product_location_group(
    product: str,
    location: str,
    file_path: str | Path = DEFAULT_FILE_PATH,
) -> tuple[str, float | None, str]:
    mean_demand, note = _mean_weekly_demand(product, location, file_path)
    group = _classify_demand_group(mean_demand)

    if mean_demand is None:
        note = f"{note}; detected_group={group}"

    return group, mean_demand, note


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def _cartesian(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _resolve_combos(combos: list[dict], grid: dict[str, list]) -> list[dict]:
    """
    Return explicit combos if provided, otherwise expand the grid.
    """
    if combos:
        return combos
    return _cartesian(grid)


def _extract_model_params(params: dict) -> dict:
    """
    Remove metadata fields before creating TrainingConfig.
    TrainingConfig should only receive actual model/training parameters.
    """
    return {k: v for k, v in params.items() if k not in METADATA_KEYS}


def _combo_matches_group(params: dict, detected_group: str) -> bool:
    """
    A combo is eligible if:
      - it has no demand_group,
      - demand_group is empty/all,
      - or demand_group matches the detected product-location group.
    """
    param_group = params.get("demand_group")

    if param_group in [None, "", "all"]:
        return True

    return str(param_group).lower() == str(detected_group).lower()


# ---------------------------------------------------------------------------
# Running one experiment
# ---------------------------------------------------------------------------

def _run_one(
    product: str,
    location: str,
    detected_demand_group: str,
    mean_weekly_demand: float | None,
    demand_detection_note: str,
    params: dict,
    run_index: int,
    total: int,
) -> tuple[dict, list]:
    model_params = _extract_model_params(params)

    run_id = params.get("run_id", f"run_{run_index}")
    run_name = params.get("run_name", "")
    param_demand_group = params.get("demand_group", "all")
    key_change = params.get("key_change", "")

    mean_print = (
        "unknown"
        if mean_weekly_demand is None
        else f"{mean_weekly_demand:.1f}"
    )

    cfg = TrainingConfig(
        **{
            **FIXED,
            "product": product,
            "location": location,
            **model_params,
        }
    )

    print(
        f"\n[{run_index}/{total}] {run_id} - {run_name}\n"
        f"    {product} @ {location}\n"
        f"    detected_demand_group={detected_demand_group}  "
        f"mean_weekly_demand={mean_print}\n"
        + "    "
        + "  ".join(f"{k}={v}" for k, v in model_params.items())
    )

    t0 = time.time()

    try:
        result = run_training_pipeline(cfg, verbose=False)
        elapsed = time.time() - t0

        row = {
            "run_index": run_index,
            "status": "ok",
            "wall_time_s": round(elapsed, 1),
            "product": product,
            "location": location,
            "run_id": run_id,
            "run_name": run_name,
            "detected_demand_group": detected_demand_group,
            "param_demand_group": param_demand_group,
            "mean_weekly_demand": _round_or_none(mean_weekly_demand, 2),
            "demand_detection_note": demand_detection_note,
            "key_change": key_change,
            "run_dir": str(result.run_dir),
            **model_params,
            "total_cost": round(result.total_cost, 2),
            "service_level": round(result.service_level, 2),
            "total_ordered": result.total_ordered,
            "avg_inventory": round(result.avg_inventory, 2),
            "forecast_weeks": len(result.records),
        }

        for sc_name, sc_recs in (result.per_scenario_records or {}).items():
            sc_kpi = compute_kpis(sc_recs)
            safe = str(sc_name).replace(" ", "_")
            row[f"{safe}_total_cost"] = round(sc_kpi["total_cost"], 2)
            row[f"{safe}_service_level"] = round(sc_kpi["service_level"], 2)
            row[f"{safe}_total_ordered"] = sc_kpi["total_ordered"]
            row[f"{safe}_avg_inventory"] = round(sc_kpi["avg_inventory"], 2)

        print(
            f"   -> cost €{result.total_cost:,.0f}  "
            f"svc {result.service_level:.1f}%  "
            f"({elapsed:.0f}s)"
        )

        return row, result.records

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"   -> FAILED: {exc}")
        traceback.print_exc()

        row = {
            "run_index": run_index,
            "status": f"error: {exc}",
            "wall_time_s": round(elapsed, 1),
            "product": product,
            "location": location,
            "run_id": run_id,
            "run_name": run_name,
            "detected_demand_group": detected_demand_group,
            "param_demand_group": param_demand_group,
            "mean_weekly_demand": _round_or_none(mean_weekly_demand, 2),
            "demand_detection_note": demand_detection_note,
            "key_change": key_change,
            "run_dir": "",
            **model_params,
            "total_cost": None,
            "service_level": None,
            "total_ordered": None,
            "avg_inventory": None,
            "forecast_weeks": None,
        }

        return row, []


# ---------------------------------------------------------------------------
# Building output tables
# ---------------------------------------------------------------------------

def _build_per_week_df(all_records: list[tuple[dict, list]]) -> pd.DataFrame:
    rows = []

    for metadata, records in all_records:
        for rec in records:
            row = dict(metadata)
            row.update(rec)
            rows.append(row)

    return pd.DataFrame(rows)


def _metadata_for_per_week(row: dict) -> dict:
    cols = [
        "run_index",
        "product",
        "location",
        "run_id",
        "run_name",
        "detected_demand_group",
        "param_demand_group",
        "mean_weekly_demand",
        "timesteps",
        "learning_rate",
        "gamma",
        "n_steps",
        "batch_size",
        "n_forecast_weeks",
    ]

    return {col: row.get(col) for col in cols if col in row}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    product_locations: list[tuple[str, str]],
    combos: list[dict],
    out_path: str,
) -> None:
    mode = (
        "full grid"
        if all("run_id" not in combo for combo in combos)
        else "explicit combinations"
    )

    print("\n=== Demand group detection ===")

    classified_product_locations = []

    for product, location in product_locations:
        detected_group, mean_demand, detection_note = _detect_product_location_group(
            product=product,
            location=location,
            file_path=FIXED["file_path"],
        )

        mean_print = "unknown" if mean_demand is None else f"{mean_demand:.1f}"

        print(
            f"    {product} @ {location} "
            f"-> mean_weekly_demand={mean_print} "
            f"-> group={detected_group}"
        )

        classified_product_locations.append(
            (
                product,
                location,
                detected_group,
                mean_demand,
                detection_note,
            )
        )

    scheduled_runs = []

    for (
        product,
        location,
        detected_group,
        mean_demand,
        detection_note,
    ) in classified_product_locations:
        for params in combos:
            if _combo_matches_group(params, detected_group):
                scheduled_runs.append(
                    (
                        product,
                        location,
                        detected_group,
                        mean_demand,
                        detection_note,
                        params,
                    )
                )

    total = len(scheduled_runs)

    if total == 0:
        print("\nNo runs were scheduled. Check PARAM_COMBOS and demand groups.")
        return

    print(f"\n=== Hyperparameter sweep ({mode}) ===")
    print(
        f"    {len(product_locations)} product-location pairs "
        f"× group-specific param combos = {total} runs"
    )
    print(f"    Output: {out_path}")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary_rows: list[dict] = []
    per_week_records: list[tuple[dict, list]] = []

    for run_index, (
        product,
        location,
        detected_group,
        mean_demand,
        detection_note,
        params,
    ) in enumerate(scheduled_runs, start=1):
        row, records = _run_one(
            product=product,
            location=location,
            detected_demand_group=detected_group,
            mean_weekly_demand=mean_demand,
            demand_detection_note=detection_note,
            params=params,
            run_index=run_index,
            total=total,
        )

        summary_rows.append(row)

        if records:
            per_week_records.append(
                (
                    _metadata_for_per_week(row),
                    records,
                )
            )

    df_summary = pd.DataFrame(summary_rows)

    ok = df_summary[df_summary["status"] == "ok"].copy()
    failed = df_summary[df_summary["status"] != "ok"].copy()

    if not ok.empty:
        sort_cols = [
            "detected_demand_group",
            "product",
            "location",
            "total_cost",
        ]
        ok = ok.sort_values(sort_cols)

    df_sorted = pd.concat([ok, failed], ignore_index=True)

    df_per_week = _build_per_week_df(per_week_records)

    out = Path(out_path)

    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        df_sorted.to_excel(writer, sheet_name="Summary", index=False)
        df_per_week.to_excel(writer, sheet_name="Per-Week", index=False)

        for sheet_name in ["Summary", "Per-Week"]:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(
                    (len(str(cell.value)) for cell in col if cell.value is not None),
                    default=8,
                )
                ws.column_dimensions[col[0].column_letter].width = min(
                    max_len + 2,
                    45,
                )

    print(f"\n=== Done. Results saved to: {out.resolve()} ===")

    if not ok.empty:
        best = ok.sort_values("total_cost").iloc[0]

        print(
            f"    Best run #{int(best['run_index'])}: "
            f"{best['run_id']} - {best['run_name']}\n"
            f"    {best['product']} @ {best['location']}\n"
            f"    cost €{best['total_cost']:,.0f}  "
            f"svc {best['service_level']:.1f}%"
        )

        param_keys = [
            "timesteps",
            "learning_rate",
            "gamma",
            "n_steps",
            "batch_size",
            "n_forecast_weeks",
        ]

        print(
            "    Params: "
            + "  ".join(
                f"{k}={best[k]}"
                for k in param_keys
                if k in best and pd.notna(best[k])
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO hyperparameter sweep")

    parser.add_argument(
        "--out",
        default="tuning_results.xlsx",
        help="Output Excel file path. Default: tuning_results.xlsx",
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        help="Force full Cartesian grid even if PARAM_COMBOS is defined.",
    )

    args = parser.parse_args()

    combos = (
        _cartesian(PARAM_GRID)
        if args.grid
        else _resolve_combos(PARAM_COMBOS, PARAM_GRID)
    )

    run_sweep(PRODUCT_LOCATIONS, combos, args.out)


if __name__ == "__main__":
    main()