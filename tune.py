"""
Hyperparameter grid search for the PPO inventory agent.

Usage
-----
  python tune.py                  # run with built-in grid below
  python tune.py --out results.xlsx

Each combination of (product, location) × hyperparameters is trained once.
Results are written to an Excel workbook with two sheets:
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
# Product-location combinations to evaluate
# ---------------------------------------------------------------------------
PRODUCT_LOCATIONS: list[tuple[str, str]] = [
    ("Ice Cream Strawberry Flavor", "Logistics Hub Lissabon"),
    ("Ice Cream Chocolate Flavor",  "Logistics Hub Munich"),
    ("Ice Cream Mango Flavor",      "Production Factory Berlin"),
    ("Ice Cream Mint Flavor",       "Logistics Hub Madrid"),
    ("Ice Cream Pistaccio Flavor",  "Logistics Hub Helsinki"),
]

# ---------------------------------------------------------------------------
# Hyperparameter grid — every combination is tested (full Cartesian product)
# ---------------------------------------------------------------------------
PARAM_GRID: dict[str, list] = {
    "timesteps":        [20_000, 100_000],
    "learning_rate":    [1e-4, 1e-3],
    "n_steps":          [128, 512, 2048],
    "gamma":            [0.99],
    "batch_size":       [64],
    "n_forecast_weeks": [4],
}

# Fixed config fields (not varied across the sweep)
FIXED: dict = {
    "file_path":       DEFAULT_FILE_PATH,
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


def _run_one(
    product: str,
    location: str,
    params: dict,
    run_index: int,
    total: int,
) -> tuple[dict, list]:
    cfg = TrainingConfig(**{**FIXED, "product": product, "location": location, **params})
    print(
        f"\n[{run_index}/{total}] {product} @ {location}\n"
        + "    " + "  ".join(f"{k}={v}" for k, v in params.items())
    )
    t0 = time.time()
    try:
        result = run_training_pipeline(cfg, verbose=False)
        elapsed = time.time() - t0

        row = {
            "run_index":     run_index,
            "status":        "ok",
            "wall_time_s":   round(elapsed, 1),
            "product":       product,
            "location":      location,
            "run_dir":       str(result.run_dir),
            **params,
            "total_cost":    round(result.total_cost, 2),
            "service_level": round(result.service_level, 2),
            "total_ordered": result.total_ordered,
            "avg_inventory": round(result.avg_inventory, 2),
            "forecast_weeks": len(result.records),
        }

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
            "product":       product,
            "location":      location,
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


def run_grid(
    product_locations: list[tuple[str, str]],
    grid: dict,
    fixed: dict,
    out_path: str,
) -> None:
    combos = _cartesian(grid)
    total = len(product_locations) * len(combos)
    print(f"\n=== Hyperparameter sweep ===")
    print(f"    {len(product_locations)} product-location pairs × {len(combos)} param combos = {total} runs")
    print(f"    Output: {out_path}")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary_rows: list[dict] = []
    per_week_records: list[tuple[int, list]] = []
    run_index = 0

    for product, location in product_locations:
        for params in combos:
            run_index += 1
            row, records = _run_one(product, location, params, run_index, total)
            summary_rows.append(row)
            if records:
                per_week_records.append((run_index, records))

    df_summary = pd.DataFrame(summary_rows)

    ok = df_summary[df_summary["status"] == "ok"].sort_values("total_cost")
    failed = df_summary[df_summary["status"] != "ok"]
    df_sorted = pd.concat([ok, failed], ignore_index=True)

    df_per_week = _build_per_week_df(per_week_records)

    out = Path(out_path)
    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        df_sorted.to_excel(writer, sheet_name="Summary", index=False)
        df_per_week.to_excel(writer, sheet_name="Per-Week", index=False)

        ws = writer.sheets["Summary"]
        for col in ws.columns:
            max_len = max((len(str(cell.value)) for cell in col if cell.value), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 40)

    print(f"\n=== Done. Results saved to: {out.resolve()} ===")
    if not ok.empty:
        best = ok.iloc[0]
        print(
            f"    Best run #{int(best['run_index'])}: "
            f"{best['product']} @ {best['location']}\n"
            f"    cost €{best['total_cost']:,.0f}  svc {best['service_level']:.1f}%"
        )
        print("    Params: " + "  ".join(f"{k}={best[k]}" for k in grid if k in best))


def main():
    parser = argparse.ArgumentParser(description="PPO hyperparameter grid search")
    parser.add_argument(
        "--out", default="tuning_results.xlsx",
        help="Output Excel file path (default: tuning_results.xlsx)",
    )
    args = parser.parse_args()
    run_grid(PRODUCT_LOCATIONS, PARAM_GRID, FIXED, args.out)


if __name__ == "__main__":
    main()
