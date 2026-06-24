"""
Hyperparameter grid search for the PPO inventory agent.

Usage
-----
  python tune.py                  # run with built-in grid below
  python tune.py --out results.xlsx

Each combination of parameters is trained once and the KPIs are written to an
Excel workbook with two sheets:
  • Summary  – one row per run, all params + KPIs, sorted by total cost
  • Per-Week – flattened per-week records for every run (for deeper analysis)
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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
# Grid definition — edit these lists to control which values are swept.
# Every combination of all lists is tested (full Cartesian product).
# ---------------------------------------------------------------------------
PARAM_GRID: dict[str, list] = {
    "timesteps":        [10_000, 50_000],
    "learning_rate":    [1e-3, 5e-4],
    "gamma":            [0.95, 0.99],
    "n_steps":          [2048],
    "batch_size":       [64],
    "n_forecast_weeks": [4],
}

# Fixed config fields (not varied)
FIXED: dict = {
    "file_path":       DEFAULT_FILE_PATH,
    "product":         "Ice Cream Strawberry Flavor",
    "location":        "Logistics Hub Lissabon",
    "scenarios":       [],      # [] = all available scenarios
    "holding_cost":    13.0,
    "ordering_cost":   60.0,
    "lost_sales_cost": 2500.0,
    "max_order_qty":   0,       # 0 = auto-detect (3× peak demand)
    "verbose":         0,
}

# ---------------------------------------------------------------------------


def _cartesian(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _run_one(params: dict, run_index: int, total: int) -> tuple[dict, list]:
    cfg = TrainingConfig(**{**FIXED, **params})
    print(
        f"\n[{run_index}/{total}] "
        + "  ".join(f"{k}={v}" for k, v in params.items())
    )
    t0 = time.time()
    try:
        result = run_training_pipeline(cfg, verbose=False)
        elapsed = time.time() - t0

        row = {
            "run_index":     run_index,
            "status":        "ok",
            "wall_time_s":   round(elapsed, 1),
            "run_dir":       str(result.run_dir),
            **params,
            "total_cost":    round(result.total_cost, 2),
            "service_level": round(result.service_level, 2),
            "total_ordered": result.total_ordered,
            "avg_inventory": round(result.avg_inventory, 2),
            "forecast_weeks": len(result.records),
        }

        # Per-scenario KPI columns (if multiple scenarios were used)
        for sc_name, sc_recs in (result.per_scenario_records or {}).items():
            sc_kpi = compute_kpis(sc_recs)
            safe = sc_name.replace(" ", "_")
            row[f"{safe}_total_cost"]    = round(sc_kpi["total_cost"], 2)
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
        return {
            "run_index":     run_index,
            "status":        f"error: {exc}",
            "wall_time_s":   round(elapsed, 1),
            "run_dir":       "",
            **params,
            "total_cost":    None,
            "service_level": None,
            "total_ordered": None,
            "avg_inventory": None,
            "forecast_weeks": None,
        }, []


def _build_per_week_df(all_records: list[tuple[int, list]]) -> pd.DataFrame:
    rows = []
    for run_idx, records in all_records:
        for rec in records:
            row = {"run_index": run_idx}
            row.update(rec)
            rows.append(row)
    return pd.DataFrame(rows)


def run_grid(grid: dict, fixed: dict, out_path: str) -> None:
    combos = _cartesian(grid)
    total = len(combos)
    print(f"\n=== Hyperparameter sweep: {total} combinations ===")
    print(f"    Output: {out_path}")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary_rows: list[dict] = []
    per_week_records: list[tuple[int, list]] = []

    for i, params in enumerate(combos, start=1):
        row, records = _run_one(params, i, total)
        summary_rows.append(row)
        if records:
            per_week_records.append((i, records))

    df_summary = pd.DataFrame(summary_rows)

    # Sort successful runs by total cost ascending; failures go to the bottom
    ok = df_summary[df_summary["status"] == "ok"].sort_values("total_cost")
    failed = df_summary[df_summary["status"] != "ok"]
    df_sorted = pd.concat([ok, failed], ignore_index=True)

    df_per_week = _build_per_week_df(per_week_records)

    out = Path(out_path)
    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        df_sorted.to_excel(writer, sheet_name="Summary", index=False)
        df_per_week.to_excel(writer, sheet_name="Per-Week", index=False)

        # Auto-size columns in Summary sheet
        ws = writer.sheets["Summary"]
        for col in ws.columns:
            max_len = max((len(str(cell.value)) for cell in col if cell.value), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    print(f"\n=== Done. Results saved to: {out.resolve()} ===")
    if not ok.empty:
        best = ok.iloc[0]
        print(
            f"    Best run #{int(best['run_index'])}: "
            f"cost €{best['total_cost']:,.0f}  "
            f"svc {best['service_level']:.1f}%"
        )
        print("    Params: " + "  ".join(f"{k}={best[k]}" for k in grid if k in best))


def main():
    parser = argparse.ArgumentParser(description="PPO hyperparameter grid search")
    parser.add_argument(
        "--out", default="tuning_results.xlsx",
        help="Output Excel file path (default: tuning_results.xlsx)",
    )
    args = parser.parse_args()
    run_grid(PARAM_GRID, FIXED, args.out)


if __name__ == "__main__":
    main()
