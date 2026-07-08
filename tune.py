"""
Hyperparameter search for the PPO inventory agent.

Usage
-----
  python tune.py
  python tune.py --out results.xlsx
  python tune.py --grid --out grid_results.xlsx

Main logic
----------
The script automatically discovers product-location pairs from the demand data.

For each demand category, it selects product-location pairs based on the limits:

  high   : 4 product-location pairs
  medium : 11 product-location pairs
  low    : 21 product-location pairs

Demand category rules:

  high   : mean weekly demand > 100
  medium : 10 <= mean weekly demand <= 100
  low    : mean weekly demand < 10

Then it applies only the matching PPO parameter configurations:

  high   pairs -> H1-H5
  medium pairs -> M1-M5
  low    pairs -> L1-L5

Excel output
------------
The output workbook contains exactly 3 demand-category sheets:

  High-Demand
  Medium-Demand
  Low-Demand

Each sheet contains the runs for the selected product-location pairs in that
demand category.
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
    run_training_pipeline,
)


# ---------------------------------------------------------------------------
# Product-location selection
# ---------------------------------------------------------------------------
# Empty list means:
#   automatically scan the Excel file and select product-location pairs
#   by demand category.
#
# If you ever want to manually test only specific pairs, you can fill this list:
# PRODUCT_LOCATIONS = [
#     ("Ice Cream Strawberry Flavor", "Logistics Hub Lissabon"),
# ]
# ---------------------------------------------------------------------------

PRODUCT_LOCATIONS: list[tuple[str, str]] = []

PRODUCT_LOCATION_LIMITS_BY_GROUP = {
    "high": 4,
    "medium": 11,
    "low": 21,
}


# ---------------------------------------------------------------------------
# MODE 1 - Explicit combinations
#
# PARAM_COMBOS takes priority when non-empty.
# Metadata fields such as run_id, run_name, demand_group, key_change are used
# for filtering and Excel output only. They are NOT passed into TrainingConfig.
# ---------------------------------------------------------------------------

PARAM_COMBOS: list[dict] = [
    # -----------------------------------------------------------------------
    # High Demand: mean > 100 units/week
    # -----------------------------------------------------------------------
    {
        "run_id": "H1",
        "run_name": "H-Baseline",
        "demand_group": "high",
        "key_change": "Reference - no changes",
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
        "key_change": "Reference - no changes",
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
        "key_change": "Reference - no changes",
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
# MODE 2 - Full Cartesian grid
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
# Demand classification settings
# ---------------------------------------------------------------------------

LOW_DEMAND_MAX = 10.0
HIGH_DEMAND_MIN = 100.0

GROUP_ORDER = ["high", "medium", "low"]

GROUP_SHEET_NAMES = {
    "high": "High-Demand",
    "medium": "Medium-Demand",
    "low": "Low-Demand",
}

METADATA_KEYS = {"run_id", "run_name", "demand_group", "key_change"}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _normalize_name(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _classify_demand_group(mean_weekly_demand: float | None) -> str:
    if mean_weekly_demand is None:
        return "medium"

    if mean_weekly_demand > HIGH_DEMAND_MIN:
        return "high"

    if mean_weekly_demand >= LOW_DEMAND_MAX:
        return "medium"

    return "low"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
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


def _read_excel_sheets(file_path: str | Path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {file_path}")

    xls = pd.ExcelFile(file_path)

    sheet_names = sorted(
        xls.sheet_names,
        key=lambda name: 0 if "demand" in _normalize_name(name) else 1,
    )

    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if not df.empty:
                yield sheet_name, df
        except Exception:
            continue


def _match_manual_pair(
    all_pairs_df: pd.DataFrame,
    product: str,
    location: str,
) -> pd.DataFrame:
    return all_pairs_df[
        (all_pairs_df["product"].astype(str).map(_normalize_name) == _normalize_name(product))
        & (all_pairs_df["location"].astype(str).map(_normalize_name) == _normalize_name(location))
    ]


# ---------------------------------------------------------------------------
# Demand data discovery
# ---------------------------------------------------------------------------

def _find_demand_value_columns(
    df: pd.DataFrame,
    product_col: str,
    location_col: str,
) -> tuple[list[str], str]:
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
        "date",
        "year",
        "month",
    ]

    protected_cols = {product_col, location_col}

    demand_like_cols = []

    for col in df.columns:
        if col in protected_cols:
            continue

        norm_col = _normalize_name(col)

        if any(token in norm_col for token in demand_like_tokens):
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                demand_like_cols.append(col)

    if demand_like_cols:
        return demand_like_cols, "demand_like_numeric_cols"

    generic_numeric_cols = []

    for col in df.columns:
        if col in protected_cols:
            continue

        norm_col = _normalize_name(col)

        if any(token in norm_col for token in exclude_numeric_tokens):
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            generic_numeric_cols.append(col)

    return generic_numeric_cols, "generic_numeric_cols"


def _discover_all_product_location_pairs(
    file_path: str | Path = DEFAULT_FILE_PATH,
) -> pd.DataFrame:
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

    for sheet_name, df in _read_excel_sheets(file_path):
        product_col = _find_column(df, product_candidates)
        location_col = _find_column(df, location_candidates)

        if product_col is None or location_col is None:
            continue

        demand_cols, detection_method = _find_demand_value_columns(
            df=df,
            product_col=product_col,
            location_col=location_col,
        )

        if not demand_cols:
            continue

        work = df[[product_col, location_col, *demand_cols]].copy()

        for col in demand_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        work["_row_mean_weekly_demand"] = work[demand_cols].mean(
            axis=1,
            skipna=True,
        )

        work = work.dropna(subset=["_row_mean_weekly_demand"])

        if work.empty:
            continue

        grouped = (
            work.groupby([product_col, location_col], as_index=False)
            .agg(mean_weekly_demand=("_row_mean_weekly_demand", "mean"))
            .rename(
                columns={
                    product_col: "product",
                    location_col: "location",
                }
            )
        )

        grouped["detected_demand_group"] = grouped["mean_weekly_demand"].apply(
            _classify_demand_group
        )

        grouped["demand_detection_note"] = (
            f"sheet={sheet_name}; "
            f"method={detection_method}; "
            f"cols={len(demand_cols)}"
        )

        grouped = grouped[
            [
                "product",
                "location",
                "mean_weekly_demand",
                "detected_demand_group",
                "demand_detection_note",
            ]
        ].copy()

        grouped = grouped.sort_values(
            ["detected_demand_group", "mean_weekly_demand"],
            ascending=[True, False],
        ).reset_index(drop=True)

        return grouped

    raise ValueError(
        "Could not discover product-location pairs from the Excel file. "
        "Please check product, location, and demand columns."
    )


def _select_top_pairs_by_group(
    all_pairs_df: pd.DataFrame,
    limits_by_group: dict[str, int] = PRODUCT_LOCATION_LIMITS_BY_GROUP,
) -> pd.DataFrame:
    selected_groups = []

    for group in GROUP_ORDER:
        group_limit = limits_by_group.get(group)

        group_df = all_pairs_df[
            all_pairs_df["detected_demand_group"] == group
        ].copy()

        group_df = group_df.sort_values(
            "mean_weekly_demand",
            ascending=False,
        )

        if group_limit is not None:
            group_df = group_df.head(group_limit)

        group_df = group_df.reset_index(drop=True)
        group_df["selected_pair_rank"] = range(1, len(group_df) + 1)

        selected_groups.append(group_df)

    selected_df = pd.concat(selected_groups, ignore_index=True)

    if selected_df.empty:
        raise ValueError("No product-location pairs were selected.")

    return selected_df


def _select_manual_pairs(
    all_pairs_df: pd.DataFrame,
    product_locations: list[tuple[str, str]],
) -> pd.DataFrame:
    manual_rows = []

    for product, location in product_locations:
        match = _match_manual_pair(all_pairs_df, product, location)

        if match.empty:
            raise ValueError(f"No data found for {product} @ {location}")

        manual_rows.append(match.iloc[0].to_dict())

    selected_df = pd.DataFrame(manual_rows)

    selected_df = selected_df.sort_values(
        ["detected_demand_group", "mean_weekly_demand"],
        ascending=[True, False],
    ).reset_index(drop=True)

    selected_df["selected_pair_rank"] = (
        selected_df.groupby("detected_demand_group").cumcount() + 1
    )

    return selected_df


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def _cartesian(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _resolve_combos(combos: list[dict], grid: dict[str, list]) -> list[dict]:
    if combos:
        return combos
    return _cartesian(grid)


def _extract_model_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in METADATA_KEYS}


def _combo_matches_group(params: dict, detected_group: str) -> bool:
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
    selected_pair_rank: int,
    params: dict,
    run_index: int,
    total: int,
) -> dict:
    model_params = _extract_model_params(params)

    run_id = params.get("run_id", f"run_{run_index}")
    run_name = params.get("run_name", "grid_run")
    param_demand_group = params.get("demand_group", "all")
    key_change = params.get("key_change", "")

    mean_print = (
        "unknown"
        if mean_weekly_demand is None
        else f"{mean_weekly_demand:.2f}"
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
        f"    group={detected_demand_group}  "
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
            "selected_pair_rank": selected_pair_rank,
            "product": product,
            "location": location,
            "mean_weekly_demand": _round_or_none(mean_weekly_demand, 2),
            "detected_demand_group": detected_demand_group,
            "param_demand_group": param_demand_group,
            "run_id": run_id,
            "run_name": run_name,
            "key_change": key_change,
            "run_dir": str(result.run_dir),
            **model_params,
            "total_cost": round(result.total_cost, 2),
            "service_level": round(result.service_level, 2),
            "total_ordered": result.total_ordered,
            "avg_inventory": round(result.avg_inventory, 2),
            "forecast_weeks": len(result.records or []),
            "demand_detection_note": demand_detection_note,
        }

        print(
            f"   -> cost EUR {result.total_cost:,.0f}  "
            f"svc {result.service_level:.1f}%  "
            f"({elapsed:.0f}s)"
        )

        return row

    except Exception as exc:
        elapsed = time.time() - t0

        print(f"   -> FAILED: {exc}")
        traceback.print_exc()

        row = {
            "run_index": run_index,
            "status": f"error: {exc}",
            "wall_time_s": round(elapsed, 1),
            "selected_pair_rank": selected_pair_rank,
            "product": product,
            "location": location,
            "mean_weekly_demand": _round_or_none(mean_weekly_demand, 2),
            "detected_demand_group": detected_demand_group,
            "param_demand_group": param_demand_group,
            "run_id": run_id,
            "run_name": run_name,
            "key_change": key_change,
            "run_dir": "",
            **model_params,
            "total_cost": None,
            "service_level": None,
            "total_ordered": None,
            "avg_inventory": None,
            "forecast_weeks": None,
            "demand_detection_note": demand_detection_note,
        }

        return row


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def _preferred_column_order(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "run_index",
        "status",
        "wall_time_s",
        "selected_pair_rank",
        "product",
        "location",
        "mean_weekly_demand",
        "detected_demand_group",
        "param_demand_group",
        "run_id",
        "run_name",
        "key_change",
        "timesteps",
        "learning_rate",
        "gamma",
        "n_steps",
        "batch_size",
        "n_forecast_weeks",
        "total_cost",
        "service_level",
        "total_ordered",
        "avg_inventory",
        "forecast_weeks",
        "run_dir",
        "demand_detection_note",
    ]

    ordered_cols = [col for col in preferred_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    return df[ordered_cols + remaining_cols]


def _make_empty_group_sheet(group: str, all_pairs_df: pd.DataFrame) -> pd.DataFrame:
    available_count = int(
        (all_pairs_df["detected_demand_group"] == group).sum()
    )

    return pd.DataFrame(
        [
            {
                "note": (
                    f"No product-location pairs were selected for {group} demand. "
                    f"Available pairs in this group: {available_count}."
                )
            }
        ]
    )


def _write_group_sheets(
    out_path: str,
    summary_df: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
    all_pairs_df: pd.DataFrame,
) -> None:
    out = Path(out_path)

    with pd.ExcelWriter(str(out), engine="openpyxl") as writer:
        for group in GROUP_ORDER:
            sheet_name = GROUP_SHEET_NAMES[group]

            group_summary = summary_df[
                summary_df["detected_demand_group"] == group
            ].copy()

            if group_summary.empty:
                group_df = _make_empty_group_sheet(group, all_pairs_df)
            else:
                group_df = group_summary.sort_values(
                    [
                        "selected_pair_rank",
                        "product",
                        "location",
                        "total_cost",
                    ],
                    ascending=[True, True, True, True],
                )

                group_df = _preferred_column_order(group_df)

            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

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


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    combos: list[dict],
    out_path: str,
    product_locations: list[tuple[str, str]] | None = None,
    limits_by_group: dict[str, int] = PRODUCT_LOCATION_LIMITS_BY_GROUP,
) -> None:
    mode = (
        "full grid"
        if all("run_id" not in combo for combo in combos)
        else "explicit combinations"
    )

    print("\n=== Product-location discovery ===")

    all_pairs_df = _discover_all_product_location_pairs(FIXED["file_path"])

    print(f"    Total discovered product-location pairs: {len(all_pairs_df)}")

    for group in GROUP_ORDER:
        count = int((all_pairs_df["detected_demand_group"] == group).sum())
        limit = limits_by_group.get(group)
        print(f"    {group}: {count} available pairs, selected limit: {limit}")

    print("\n=== Product-location selection ===")

    if product_locations:
        selected_pairs_df = _select_manual_pairs(
            all_pairs_df=all_pairs_df,
            product_locations=product_locations,
        )
    else:
        selected_pairs_df = _select_top_pairs_by_group(
            all_pairs_df=all_pairs_df,
            limits_by_group=limits_by_group,
        )

    for group in GROUP_ORDER:
        group_pairs = selected_pairs_df[
            selected_pairs_df["detected_demand_group"] == group
        ].copy()

        print(
            f"\n    {group.upper()} demand: "
            f"{len(group_pairs)} selected pair(s)"
        )

        if group_pairs.empty:
            print("        No available product-location pairs in this group.")
            continue

        for _, row in group_pairs.iterrows():
            print(
                f"        #{int(row['selected_pair_rank'])}: "
                f"{row['product']} @ {row['location']} | "
                f"mean_weekly_demand={row['mean_weekly_demand']:.2f}"
            )

    scheduled_runs = []

    for _, pair_row in selected_pairs_df.iterrows():
        product = pair_row["product"]
        location = pair_row["location"]
        detected_group = pair_row["detected_demand_group"]
        mean_demand = float(pair_row["mean_weekly_demand"])
        detection_note = pair_row["demand_detection_note"]
        selected_pair_rank = int(pair_row["selected_pair_rank"])

        for params in combos:
            if _combo_matches_group(params, detected_group):
                scheduled_runs.append(
                    (
                        product,
                        location,
                        detected_group,
                        mean_demand,
                        detection_note,
                        selected_pair_rank,
                        params,
                    )
                )

    total = len(scheduled_runs)

    if total == 0:
        print("\nNo runs were scheduled. Check PARAM_COMBOS and demand groups.")
        return

    print(f"\n=== Hyperparameter sweep ({mode}) ===")
    print(
        f"    {len(selected_pairs_df)} selected product-location pairs "
        f"with group-specific parameter combos = {total} runs"
    )
    print(f"    Output: {out_path}")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary_rows: list[dict] = []

    for run_index, (
        product,
        location,
        detected_group,
        mean_demand,
        detection_note,
        selected_pair_rank,
        params,
    ) in enumerate(scheduled_runs, start=1):
        row = _run_one(
            product=product,
            location=location,
            detected_demand_group=detected_group,
            mean_weekly_demand=mean_demand,
            demand_detection_note=detection_note,
            selected_pair_rank=selected_pair_rank,
            params=params,
            run_index=run_index,
            total=total,
        )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    _write_group_sheets(
        out_path=out_path,
        summary_df=summary_df,
        selected_pairs_df=selected_pairs_df,
        all_pairs_df=all_pairs_df,
    )

    ok = summary_df[summary_df["status"] == "ok"].copy()

    if not ok.empty:
        best = ok.sort_values("total_cost").iloc[0]

        print(
            f"    Best run #{int(best['run_index'])}: "
            f"{best['run_id']} - {best['run_name']}\n"
            f"    {best['product']} @ {best['location']}\n"
            f"    group {best['detected_demand_group']} | "
            f"cost EUR {best['total_cost']:,.0f} | "
            f"svc {best['service_level']:.1f}%"
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

    run_sweep(
        combos=combos,
        out_path=args.out,
        product_locations=PRODUCT_LOCATIONS,
        limits_by_group=PRODUCT_LOCATION_LIMITS_BY_GROUP,
    )


if __name__ == "__main__":
    main()