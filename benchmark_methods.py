"""Reusable traditional inventory benchmark policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inventory_ppo import list_scenarios, load_data


METHOD_PPO = 'PPO'
METHOD_STATIC_SS = 'Static (s, S)'
METHOD_FORECAST_OUT = 'Forecast-based Order-up-to'

COMPARISON_FILENAME = 'benchmark_comparison.xlsx'


def _arrival_label(week_labels, order_step_idx, lead_time):
    idx = order_step_idx + lead_time
    if idx < len(week_labels):
        return str(week_labels[idx])
    return f"+{idx - len(week_labels) + 1}wk"


def _simulate_policy(
    method,
    demand,
    forecast,
    week_labels,
    lead_time,
    initial_inventory,
    holding_cost,
    ordering_cost,
    lost_sales_cost,
    order_rule,
):
    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    week_labels = [str(w) for w in week_labels]
    lead_time = int(lead_time)
    pipeline = [0] * lead_time
    inventory = float(initial_inventory)
    records = []

    for step, demand_value in enumerate(demand):
        arriving_qty = pipeline.pop(0) if pipeline else 0
        inventory += arriving_qty
        inventory_after_arrival = inventory

        unmet_demand = max(float(demand_value) - inventory, 0.0)
        inventory = max(inventory - float(demand_value), 0.0)

        inventory_position = inventory + sum(pipeline)
        order_qty = max(float(order_rule(step, inventory_position, forecast)), 0.0)

        holding_cost_total = float(holding_cost) * inventory
        ordering_cost_total = float(ordering_cost) * order_qty
        lost_sales_cost_total = float(lost_sales_cost) * unmet_demand
        reward = -(holding_cost_total + ordering_cost_total + lost_sales_cost_total)

        if lead_time > 0:
            pipeline.append(order_qty)

        records.append({
            'week': week_labels[step] if step < len(week_labels) else f'W{step}',
            'due': _arrival_label(week_labels, step, int(lead_time)),
            'method': method,
            'actual_demand': float(demand_value),
            'forecast': float(forecast[step]) if step < len(forecast) else np.nan,
            'arriving_qty': arriving_qty,
            'inventory_after_arrival': inventory_after_arrival,
            'order_qty': order_qty,
            'unmet_demand': unmet_demand,
            'inventory': inventory,
            'reward': reward,
            'holding_cost_total': holding_cost_total,
            'ordering_cost_total': ordering_cost_total,
            'lost_sales_cost_total': lost_sales_cost_total,
        })

    return records


def static_ss_benchmark(
    demand,
    forecast,
    week_labels,
    lead_time,
    initial_inventory,
    holding_cost,
    ordering_cost,
    lost_sales_cost,
):
    """Fixed reorder point/order-up-to policy derived from observed demand scale."""
    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    signal = forecast[: len(demand)] if len(forecast) >= len(demand) else demand
    avg = float(np.mean(signal)) if len(signal) else 0.0
    std = float(np.std(signal)) if len(signal) else 0.0
    cover_weeks = max(int(lead_time), 1)
    s_level = avg * cover_weeks
    S_level = s_level + avg + std

    def order_rule(_step, inventory_position, _forecast):
        if inventory_position <= s_level:
            return S_level - inventory_position
        return 0.0

    return _simulate_policy(
        METHOD_STATIC_SS, demand, forecast, week_labels, lead_time, initial_inventory,
        holding_cost, ordering_cost, lost_sales_cost, order_rule,
    )


def forecast_order_up_to_benchmark(
    demand,
    forecast,
    week_labels,
    lead_time,
    initial_inventory,
    holding_cost,
    ordering_cost,
    lost_sales_cost,
):
    """Dynamic order-up-to policy using forecast demand over lead time plus one week."""
    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) == 0:
        forecast = demand
    horizon = max(int(lead_time), 0) + 1

    def order_rule(step, inventory_position, forecast_values):
        window = forecast_values[step: step + horizon]
        if len(window) < horizon:
            pad_value = window[-1] if len(window) else 0.0
            window = np.pad(window, (0, horizon - len(window)), constant_values=pad_value)
        target = float(np.sum(window))
        return target - inventory_position

    return _simulate_policy(
        METHOD_FORECAST_OUT, demand, forecast, week_labels, lead_time, initial_inventory,
        holding_cost, ordering_cost, lost_sales_cost, order_rule,
    )


def _average_records(records_by_scenario):
    if not records_by_scenario:
        return []
    n_weeks = min(len(records) for records in records_by_scenario)
    numeric_keys = [
        'actual_demand', 'forecast', 'arriving_qty', 'inventory_after_arrival',
        'order_qty', 'unmet_demand', 'inventory', 'reward',
        'holding_cost_total', 'ordering_cost_total', 'lost_sales_cost_total',
    ]
    averaged = []
    for i in range(n_weeks):
        row = {k: v for k, v in records_by_scenario[0][i].items() if k not in numeric_keys}
        for key in numeric_keys:
            row[key] = float(np.mean([records[i][key] for records in records_by_scenario]))
        averaged.append(row)
    return averaged


def generate_benchmarks_for_selection(config):
    """Build benchmark records for the selected product/location and active scenarios."""
    all_scenarios = list_scenarios(config.file_path)
    active_scenarios = config.scenarios if config.scenarios else (all_scenarios[:1] or [None])

    per_method = {METHOD_STATIC_SS: [], METHOD_FORECAST_OUT: []}
    for scenario in active_scenarios:
        scenario_key = scenario if scenario in all_scenarios else None
        (
            _hist_demand,
            _forecast_data,
            lead_time,
            initial_inventory,
            _hist_week_labels,
            future_forecast,
            future_week_labels,
        ) = load_data(config.file_path, config.product, config.location, scenario=scenario_key)

        planning_demand = future_forecast if len(future_forecast) else _forecast_data
        planning_weeks = future_week_labels if len(future_week_labels) else _hist_week_labels
        planning_forecast = planning_demand

        per_method[METHOD_STATIC_SS].append(static_ss_benchmark(
            planning_demand, planning_forecast, planning_weeks, lead_time, initial_inventory,
            config.holding_cost, config.ordering_cost, config.lost_sales_cost,
        ))
        per_method[METHOD_FORECAST_OUT].append(forecast_order_up_to_benchmark(
            planning_demand, planning_forecast, planning_weeks, lead_time, initial_inventory,
            config.holding_cost, config.ordering_cost, config.lost_sales_cost,
        ))

    return {
        method: _average_records(records_list) if len(records_list) > 1 else records_list[0]
        for method, records_list in per_method.items()
        if records_list
    }


def comparison_rows(product, location, method_records):
    rows = []
    for method, records in method_records.items():
        cumulative_cost = 0.0
        for record in records:
            total_cost = -float(record['reward'])
            cumulative_cost += total_cost
            rows.append({
                'product': product,
                'location': location,
                'week': record['week'],
                'method': method,
                'demand': record.get('actual_demand'),
                'forecast': record.get('forecast', record.get('actual_demand')),
                'beginning_inventory': record.get('inventory_after_arrival'),
                'order_quantity': record.get('order_qty'),
                'ending_inventory': record.get('inventory'),
                'lost_sales_or_shortage': record.get('unmet_demand'),
                'holding_cost': record.get('holding_cost_total'),
                'ordering_cost': record.get('ordering_cost_total'),
                'lost_sales_cost': record.get('lost_sales_cost_total'),
                'total_cost': total_cost,
                'cumulative_cost': cumulative_cost,
            })
    return rows


def write_comparison_excel(product, location, method_records, out_path):
    """Write one row per week per method to an Excel comparison file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(product, location, method_records)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(['product', 'location', 'method'], as_index=False)
        .agg(
            total_cost=('total_cost', 'sum'),
            total_ordered=('order_quantity', 'sum'),
            total_lost_sales_or_shortage=('lost_sales_or_shortage', 'sum'),
            average_ending_inventory=('ending_inventory', 'mean'),
        )
        if not df.empty else pd.DataFrame()
    )
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Weekly Comparison', index=False)
        summary.to_excel(writer, sheet_name='Method Summary', index=False)
    return out_path
