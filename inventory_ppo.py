import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import warnings

warnings.simplefilter("ignore")

DEFAULT_FILE_PATH = "Sample_Data_RL4IM_SCENARIOS.xlsx"
RUNS_DIR = Path("runs")

base_stock_results = []  # list of (S, records) tuples — synced from structured baseline data for matplotlib/excel
BS_COLORS = ['#2CA02C', '#9467BD', '#8C564B', '#E377C2']
BS_VARIANTS = ('conservative', 'middle', 'aggressive')
BS_VARIANT_LABELS = {
    'conservative': 'Conservative',
    'middle': 'Middle',
    'aggressive': 'Aggressive',
}


def _load_chart_theme():
    """Import shared TUM chart palette (falls back to defaults if ui package unavailable)."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from ui.tum_theme import (  # noqa: WPS433
            BG, C_DEM, C_HLD, C_HOR, C_INV, C_LST, C_ORD, C_ORC, C_UNM,
            FONT_FAMILY, FUT_SHADE, GRID, MUTED, PANEL, TEXT, TUM_BLUE_DARK,
        )
        return {
            'BG': BG, 'PANEL': PANEL, 'GRID': GRID, 'TEXT': TEXT, 'MUTED': MUTED,
            'C_INV': C_INV, 'C_DEM': C_DEM, 'C_UNM': C_UNM, 'C_ORD': C_ORD,
            'C_HLD': C_HLD, 'C_ORC': C_ORC, 'C_LST': C_LST, 'C_HOR': C_HOR,
            'FUT_SHADE': FUT_SHADE, 'FONT_FAMILY': FONT_FAMILY, 'TUM_BLUE_DARK': TUM_BLUE_DARK,
        }
    except ImportError:
        return {
            'BG': '#F5F8FA', 'PANEL': '#FFFFFF', 'GRID': '#CCCCCC', 'TEXT': '#333333',
            'MUTED': '#666666', 'C_INV': '#0065BD', 'C_DEM': '#E57373', 'C_UNM': '#C62828',
            'C_ORD': '#F57C00', 'C_HLD': '#4A90D9', 'C_ORC': '#FBC02D', 'C_LST': '#D32F2F',
            'C_HOR': '#005293', 'FUT_SHADE': '#FFF9E6', 'FONT_FAMILY': 'Arial', 'TUM_BLUE_DARK': '#005293',
        }


class SingleEchelonEnv(gym.Env):
    """
    Custom Environment for a single-echelon inventory system.
    """
    def __init__(self, demand_data, forecast_data, lead_time, initial_inventory,
                 holding_cost=13, ordering_cost=60, lost_sales_cost=2500,
                 max_order_qty=500, n_forecast_weeks=4):
        super(SingleEchelonEnv, self).__init__()

        self.demand_data = demand_data
        self.forecast_data = forecast_data
        self.lead_time = lead_time
        self.initial_inventory = initial_inventory

        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.lost_sales_cost = lost_sales_cost
        self.max_order_qty = max_order_qty
        self.n_forecast_weeks = n_forecast_weeks

        # Action space: Continuous [0, 1], will be scaled to [0, max_order_qty] and rounded
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observation space:
        # 1. Current Inventory
        # 2. Pipeline Inventory (length = lead_time)
        # 3. Forecast for next N weeks
        obs_dim = 1 + self.lead_time + self.n_forecast_weeks
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = len(demand_data) - n_forecast_weeks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.pipeline = [0] * self.lead_time
        return self._get_obs(), {}

    def _get_obs(self):
        forecasts = self.forecast_data[self.current_step : self.current_step + self.n_forecast_weeks]
        if len(forecasts) < self.n_forecast_weeks:
            forecasts = np.pad(forecasts, (0, self.n_forecast_weeks - len(forecasts)), 'edge')
        obs = np.concatenate([
            [float(self.inventory)],
            [float(x) for x in self.pipeline],
            [float(x) for x in forecasts]
        ]).astype(np.float32)
        
        # Normalize by max_order_qty so all components stay in a [0, ~1] range
        # regardless of product scale (avoids near-zero gradients for low-demand SKUs).
        scale = max(float(self.max_order_qty), 1.0)
        return obs / scale

    def step(self, action):
        # 1. Receive incoming order from pipeline (Lead Time delay)
        arriving_qty = self.pipeline.pop(0)
        self.inventory += arriving_qty
        inventory_after_arrival = self.inventory

        # 2. Observe actual demand for the current week
        actual_demand = self.demand_data[self.current_step]

        # 3. Satisfy demand (Lost Sales Model: no backlogging)
        if self.inventory >= actual_demand:
            unmet_demand = 0
            self.inventory -= actual_demand
        else:
            unmet_demand = actual_demand - self.inventory
            self.inventory = 0

        # 4. Agent's Action: scale [0,1] to [0, max_order_qty]
        order_qty = int(round(float(action[0]) * self.max_order_qty))

        # 5. Calculate Reward (Negative of Total Costs)
        reward = -(
            (self.holding_cost    * self.inventory) +
            (self.ordering_cost   * order_qty) +
            (self.lost_sales_cost * unmet_demand)
        )

        # 6. Add new order to the end of the pipeline
        self.pipeline.append(order_qty)

        # 7. Update step and check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "actual_demand":           actual_demand,
            "arriving_qty":            arriving_qty,
            "inventory_after_arrival": inventory_after_arrival,
            "order_qty":               order_qty,
            "unmet_demand":            unmet_demand,
            "inventory":               self.inventory,
            "reward":                  reward,
            "holding_cost_total":      self.holding_cost    * self.inventory,
            "ordering_cost_total":     self.ordering_cost   * order_qty,
            "lost_sales_cost_total":   self.lost_sales_cost * unmet_demand,
        }

        # Return scaled reward for better PPO convergence
        return self._get_obs(), reward / 1000.0, terminated, truncated, info


def load_data(file_path, product, location, scenario=None):
    """
    Load data from the Excel file and handle potential whitespace in sheet/column names.
    Supports an optional Scenario column in the Forecast sheet (v3+).
    Returns: demand, forecast, lead_time, initial_inventory, week_labels,
             future_forecast, future_week_labels
    """
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    def get_sheet_name(base_name):
        for name in sheet_names:
            if name.strip() == base_name:
                return name
        return base_name

    df_demand    = pd.read_excel(file_path, sheet_name=get_sheet_name('Demand'))
    df_inventory = pd.read_excel(file_path, sheet_name=get_sheet_name('Current Inventory'))
    df_lead_time = pd.read_excel(file_path, sheet_name=get_sheet_name('Lead Time'))
    df_forecast  = pd.read_excel(file_path, sheet_name=get_sheet_name('Forecast'))

    for df in [df_demand, df_inventory, df_lead_time, df_forecast]:
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    week_labels = df_demand.columns[2:].tolist()

    demand_row = df_demand[(df_demand['Product'] == product) & (df_demand['Location'] == location)]
    if demand_row.empty:
        raise ValueError(f"No demand data found for {product} at {location}")
    demand = demand_row.iloc[0, 2:].values.astype(int)

    inventory_row = df_inventory[
        (df_inventory['Product'] == product) & (df_inventory['Location'] == location)]
    if inventory_row.empty:
        raise ValueError(f"No inventory data found for {product} at {location}")
    inventory = inventory_row.iloc[0, 2]

    lt_row = df_lead_time[
        (df_lead_time['Product'] == product) & (df_lead_time['Location'] == location)]
    if lt_row.empty:
        raise ValueError(f"No lead time data found for {product} at {location}")
    lead_time = int(lt_row['Lead Time in weeks'].values[0])

    # Detect scenario column — shifts week data from index 2 to 3
    has_scenario_col = 'Scenario' in df_forecast.columns
    forecast_col_start = 3 if has_scenario_col else 2

    filt = (df_forecast['Product'] == product) & (df_forecast['Location'] == location)
    if has_scenario_col and scenario is not None:
        filt = filt & (df_forecast['Scenario'] == scenario)
    forecast_rows = df_forecast[filt]
    if forecast_rows.empty:
        raise ValueError(f"No forecast data found for {product} at {location}"
                         + (f" (scenario={scenario})" if scenario else ""))
    forecast = forecast_rows.iloc[0, forecast_col_start:].values.astype(int)

    demand_week_set = set(str(w) for w in week_labels)
    forecast_all_week_labels = df_forecast.columns[forecast_col_start:].tolist()
    future_week_labels = [w for w in forecast_all_week_labels if str(w) not in demand_week_set]
    if future_week_labels:
        future_forecast = forecast_rows.iloc[0][future_week_labels].values.astype(int)
    else:
        future_forecast = np.array([], dtype=int)
        future_week_labels = []

    return demand, forecast, lead_time, inventory, week_labels, future_forecast, future_week_labels


def list_scenarios(file_path):
    """Return sorted unique scenario names from the Forecast sheet, or [] if no Scenario column."""
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    for name in sheet_names:
        if name.strip() == 'Forecast':
            sheet_name = name
            break
    else:
        sheet_name = 'Forecast'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    if 'Scenario' not in df.columns:
        return []
    return sorted(df['Scenario'].dropna().unique().tolist())


def suggest_max_order_qty(demand_data: np.ndarray) -> int:
    """Return a sensible order cap: 3× peak weekly demand, minimum 10."""
    return max(1, int(np.max(demand_data)) * 3)


def list_product_location_pairs(file_path):
    """Return valid (product, location) pairs from the Demand sheet."""
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names

    def get_sheet_name(base_name):
        for name in sheet_names:
            if name.strip() == base_name:
                return name
        return base_name

    df_demand = pd.read_excel(file_path, sheet_name=get_sheet_name('Demand'))
    df_demand.columns = [col.strip() if isinstance(col, str) else col for col in df_demand.columns]
    pairs = df_demand[['Product', 'Location']].drop_duplicates()
    return [(row['Product'], row['Location']) for _, row in pairs.iterrows()]


def list_products(file_path):
    pairs = list_product_location_pairs(file_path)
    return sorted({p for p, _ in pairs})


def list_locations_for_product(file_path, product):
    pairs = list_product_location_pairs(file_path)
    return sorted({loc for p, loc in pairs if p == product})


@dataclass
class TrainingConfig:
    file_path: str = DEFAULT_FILE_PATH
    product: str = "Ice Cream Strawberry Flavor"
    location: str = "Logistics Hub Lissabon"
    scenarios: list = field(default_factory=list)  # empty = use first/only scenario
    timesteps: int = 10000
    learning_rate: float = 1e-3
    holding_cost: float = 13
    ordering_cost: float = 60
    lost_sales_cost: float = 2500
    max_order_qty: int = 0  # 0 = auto-detect from data (3× peak demand)
    n_forecast_weeks: int = 4
    gamma: float = 0.99
    n_steps: int = 2048
    batch_size: int = 64
    verbose: int = 1


@dataclass
class RunResult:
    config: TrainingConfig
    records: list
    future_records: list
    product: str
    location: str
    lead_time: int
    total_cost: float
    service_level: float
    total_ordered: int
    avg_inventory: float
    run_dir: Path
    started_at: str
    finished_at: str
    duration_seconds: float
    hist_demand: list = field(default_factory=list)
    hist_week_labels: list = field(default_factory=list)
    base_stock_results: list = field(default_factory=list)
    model: object = field(repr=False, default=None)
    per_scenario_records: dict = field(default_factory=dict)


def product_slug(product):
    slug = re.sub(r'[^a-z0-9]+', '-', product.lower()).strip('-')
    return slug or 'product'


def make_run_dir(product, base_dir=None):
    base = Path(base_dir) if base_dir else RUNS_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    run_dir = base / f"{stamp}_{product_slug(product)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_kpis(records):
    if not records:
        return {'total_cost': 0.0, 'service_level': 0.0, 'total_ordered': 0, 'avg_inventory': 0.0}
    demand = np.array([r['actual_demand'] for r in records], dtype=float)
    unmet = np.array([r['unmet_demand'] for r in records], dtype=float)
    orders = np.array([r['order_qty'] for r in records], dtype=float)
    inventory = np.array([r['inventory'] for r in records], dtype=float)
    rewards = np.array([r['reward'] for r in records], dtype=float)
    return {
        'total_cost': float(-rewards.sum()),
        'service_level': 100.0 * (1 - unmet.sum() / max(demand.sum(), 1)),
        'total_ordered': int(orders.sum()),
        'avg_inventory': float(inventory.mean()),
    }


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, on_progress=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.on_progress = on_progress

    def _on_step(self):
        if self.on_progress:
            self.on_progress(self.num_timesteps, self.total_timesteps)
        return True


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _serialize_records(records):
    out = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            else:
                row[k] = v
        out.append(row)
    return out


def run_future_projection(model, final_inventory, final_pipeline, future_forecast,
                           future_week_labels, lead_time, n_forecast_weeks=4,
                           holding_cost=13, ordering_cost=60, lost_sales_cost=2500,
                           max_order_qty=200):
    if len(future_forecast) == 0:
        print("No future forecast weeks found in the data — skipping forward projection.")
        return []

    padded = np.append(future_forecast, np.full(n_forecast_weeks, future_forecast[-1]))

    env = SingleEchelonEnv(
        demand_data=padded, forecast_data=padded, lead_time=lead_time,
        initial_inventory=int(final_inventory), holding_cost=holding_cost,
        ordering_cost=ordering_cost, lost_sales_cost=lost_sales_cost,
        max_order_qty=max_order_qty, n_forecast_weeks=n_forecast_weeks,
    )
    obs, _ = env.reset()
    env.inventory = int(final_inventory)
    env.pipeline  = list(final_pipeline)
    env.max_steps = len(future_forecast)
    obs = env._get_obs()

    records = []
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        step_idx = env.current_step
        obs, _, terminated, _, info = env.step(action)
        week_val = future_week_labels[step_idx] if step_idx < len(future_week_labels) else f"F+{step_idx}"
        records.append({'week': week_val, **info})

    return records


def run_base_stock_policy(
    S, demand_data, week_labels, lead_time, initial_inventory,
    holding_cost=13, ordering_cost=60, lost_sales_cost=2500,
):
    """
    Simulate a fixed base-stock policy: order_qty = max(0, S - inventory - pipeline_sum).
    Replicates SingleEchelonEnv step logic and returns records in the same format as PPO.
    """
    demand_data = np.asarray(demand_data, dtype=int)
    week_labels = list(week_labels)
    inventory = int(initial_inventory)
    pipeline = [0] * lead_time
    records = []

    def arrival_label(order_step_idx):
        idx = order_step_idx + lead_time
        if idx < len(week_labels):
            return str(week_labels[idx])
        return f"+{idx - len(week_labels) + 1}wk"

    for step_idx in range(len(demand_data)):
        arriving_qty = pipeline.pop(0)
        inventory += arriving_qty
        inventory_after_arrival = inventory

        actual_demand = int(demand_data[step_idx])

        if inventory >= actual_demand:
            unmet_demand = 0
            inventory -= actual_demand
        else:
            unmet_demand = actual_demand - inventory
            inventory = 0

        pipeline_sum = sum(pipeline)
        order_qty = max(0, int(S) - (inventory + pipeline_sum))

        reward = -(
            (holding_cost * inventory) +
            (ordering_cost * order_qty) +
            (lost_sales_cost * unmet_demand)
        )

        pipeline.append(order_qty)

        week_val = week_labels[step_idx] if step_idx < len(week_labels) else f"F+{step_idx}"
        records.append({
            'week': week_val,
            'due': arrival_label(step_idx),
            'actual_demand': actual_demand,
            'arriving_qty': arriving_qty,
            'inventory_after_arrival': inventory_after_arrival,
            'order_qty': order_qty,
            'unmet_demand': unmet_demand,
            'inventory': inventory,
            'reward': reward,
            'holding_cost_total': holding_cost * inventory,
            'ordering_cost_total': ordering_cost * order_qty,
            'lost_sales_cost_total': lost_sales_cost * unmet_demand,
        })

    return records


def _base_stock_legacy_pairs(bs_results):
    """Convert structured baseline list to (S, records) tuples for matplotlib/excel."""
    return [(b['S'], b['records']) for b in bs_results]


def _sync_base_stock_global(bs_results):
    global base_stock_results
    base_stock_results = _base_stock_legacy_pairs(bs_results)


def _serialize_base_stock_results(bs_results):
    return [
        {
            'S': b['S'],
            'variant': b['variant'],
            'kpis': b['kpis'],
            'records': _serialize_records(b['records']),
        }
        for b in bs_results
    ]


def deserialize_base_stock_results(payload):
    """Restore structured baseline list from records.json payload."""
    if not payload:
        return []
    out = []
    for item in payload:
        records = item.get('records', [])
        kpis = item.get('kpis')
        if not kpis and records:
            kpis = _policy_kpis(records)
        out.append({
            'S': int(item['S']),
            'variant': item.get('variant', 'middle'),
            'records': records,
            'kpis': kpis or _policy_kpis([]),
        })
    return out


def config_from_dict(config_dict, file_path=None):
    """Build TrainingConfig from a saved config.json dict."""
    fp = file_path or config_dict.get('file_path', DEFAULT_FILE_PATH)
    return TrainingConfig(
        file_path=fp,
        product=str(config_dict.get('product', '')),
        location=str(config_dict.get('location', '')),
        scenarios=list(config_dict.get('scenarios') or []),
        timesteps=int(config_dict.get('timesteps', 10000)),
        learning_rate=float(config_dict.get('learning_rate', 1e-3)),
        holding_cost=float(config_dict.get('holding_cost', 13)),
        ordering_cost=float(config_dict.get('ordering_cost', 60)),
        lost_sales_cost=float(config_dict.get('lost_sales_cost', 2500)),
        max_order_qty=int(config_dict.get('max_order_qty', 0)),
        n_forecast_weeks=int(config_dict.get('n_forecast_weeks', 4)),
        gamma=float(config_dict.get('gamma', 0.99)),
        n_steps=int(config_dict.get('n_steps', 2048)),
        batch_size=int(config_dict.get('batch_size', 64)),
        verbose=int(config_dict.get('verbose', 0)),
    )


def compute_base_stock_baselines(config, demand_data, lead_time, initial_inventory,
                                 active_scenarios=None, all_scenarios=None):
    """Compute three Base Stock variants on the planning period (future forecast)."""
    if all_scenarios is None:
        all_scenarios = list_scenarios(config.file_path)
    if active_scenarios is None:
        active_scenarios = config.scenarios if config.scenarios else (all_scenarios[:1] or [None])

    avg_demand = float(np.mean(demand_data))
    variant_specs = [
        ('conservative', int(round(avg_demand * lead_time))),
        ('middle', int(round(avg_demand * (lead_time + 1)))),
        ('aggressive', int(round(avg_demand * (lead_time + 2)))),
    ]

    results = []
    for variant, S in variant_specs:
        scenario_bs_list = []
        for sc in active_scenarios:
            sc_key = sc if sc in all_scenarios else None
            _, _, _, _, _, sc_future_forecast, sc_future_week_labels = load_data(
                config.file_path, config.product, config.location, scenario=sc_key,
            )
            bs_records = run_base_stock_policy(
                S, sc_future_forecast, sc_future_week_labels, lead_time,
                initial_inventory,
                holding_cost=config.holding_cost,
                ordering_cost=config.ordering_cost,
                lost_sales_cost=config.lost_sales_cost,
            )
            if bs_records:
                scenario_bs_list.append(bs_records)
        if len(scenario_bs_list) > 1:
            bs_records = _average_records(scenario_bs_list)
        elif scenario_bs_list:
            bs_records = scenario_bs_list[0]
        else:
            bs_records = []
        results.append({
            'S': S,
            'variant': variant,
            'records': bs_records,
            'kpis': _policy_kpis(bs_records),
        })
    return results


def ensure_base_stock_baselines(config, file_path=None):
    """Compute baselines from config (for runs missing persisted baseline data)."""
    if isinstance(config, dict):
        cfg = config_from_dict(config, file_path=file_path)
    else:
        cfg = config
    all_scenarios = list_scenarios(cfg.file_path)
    active_scenarios = cfg.scenarios if cfg.scenarios else (all_scenarios[:1] or [None])
    first_scenario = active_scenarios[0] if active_scenarios[0] in all_scenarios else None
    demand_data, _, lead_time, initial_inventory, _, _, _ = load_data(
        cfg.file_path, cfg.product, cfg.location, scenario=first_scenario,
    )
    return compute_base_stock_baselines(
        cfg, demand_data, lead_time, initial_inventory,
        active_scenarios=active_scenarios, all_scenarios=all_scenarios,
    )


def _policy_kpis(records):
    """Return display-ready KPI dict for a list of simulation records."""
    if not records:
        return {'Total Cost (€)': 0.0, 'Service Level (%)': 0.0,
                'Total Ordered': 0, 'Avg Inventory': 0.0}
    demand  = np.array([r['actual_demand'] for r in records], dtype=float)
    unmet   = np.array([r['unmet_demand']  for r in records], dtype=float)
    orders  = np.array([r['order_qty']     for r in records], dtype=float)
    inv     = np.array([r['inventory']     for r in records], dtype=float)
    rewards = np.array([r['reward']        for r in records], dtype=float)
    return {
        'Total Cost (€)':    round(float(-rewards.sum()), 2),
        'Service Level (%)': round(100.0 * (1 - unmet.sum() / max(demand.sum(), 1)), 2),
        'Total Ordered':     int(orders.sum()),
        'Avg Inventory':     round(float(inv.mean()), 2),
    }


def export_to_excel(records, future_records, product, location, total_cost, out_path='results.xlsx',
                    hist_demand=None, hist_week_labels=None):

    hist_rows = [{
        'Week':                r['week'],
        'Due Week':            r.get('due', ''),
        'Forecast Demand':     r['actual_demand'],
        'Arrived Qty':         r['arriving_qty'],
        'Order Qty':           r['order_qty'],
        'Unmet Demand':        r['unmet_demand'],
        'Inventory (End)':     r['inventory'],
        'Holding Cost (€)':    r['holding_cost_total'],
        'Ordering Cost (€)':   r['ordering_cost_total'],
        'Lost Sales Cost (€)': r['lost_sales_cost_total'],
        'Total Cost (€)':      -r['reward'],
    } for r in records]

    fut_rows = [{
        'Week':                r['week'],
        'Due Week':            r.get('due', ''),
        'Forecast Demand':     r['actual_demand'],
        'Arrived Qty':         r['arriving_qty'],
        'Order Qty':           r['order_qty'],
        'Inventory (End)':     r['inventory'],
        'Holding Cost (€)':    r['holding_cost_total'],
        'Ordering Cost (€)':   r['ordering_cost_total'],
        'Lost Sales Cost (€)': r['lost_sales_cost_total'],
        'Total Cost (€)':      -r['reward'],
    } for r in future_records]

    ppo_kpis = _policy_kpis(records)
    summary_rows = [
        {'Metric': 'Product',               'Value': product},
        {'Metric': 'Location',              'Value': location},
        {'Metric': 'Total Cost (€)',         'Value': round(total_cost, 2)},
        {'Metric': 'Service Level (%)',      'Value': round(ppo_kpis['Service Level (%)'], 2)},
        {'Metric': 'Total Ordered (units)',  'Value': ppo_kpis['Total Ordered']},
        {'Metric': 'Avg Inventory (units)',  'Value': round(ppo_kpis['Avg Inventory'], 2)},
        {'Metric': 'Forecast Weeks',          'Value': len(records)},
    ]

    comp_rows = [{'Policy': 'PPO Agent', **ppo_kpis}]
    for S, bs_recs in base_stock_results:
        comp_rows.append({'Policy': f'Base Stock S={S}', **_policy_kpis(bs_recs)})

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
        if len(comp_rows) > 1:
            pd.DataFrame(comp_rows).to_excel(writer, sheet_name='Policy Comparison', index=False)
        if hist_demand is not None and len(hist_demand) > 0:
            hist_labels = hist_week_labels if hist_week_labels else list(range(len(hist_demand)))
            pd.DataFrame({'Week': hist_labels, 'Actual Demand': hist_demand}).to_excel(
                writer, sheet_name='Historical Demand', index=False)
        pd.DataFrame(hist_rows).to_excel(writer, sheet_name='Inventory Plan (PPO)', index=False)
        if fut_rows:
            pd.DataFrame(fut_rows).to_excel(writer, sheet_name='Future Projection', index=False)
        for S, bs_recs in base_stock_results:
            bs_rows = [{
                'Week':                r['week'],
                'Demand':              r['actual_demand'],
                'Arrived Qty':         r['arriving_qty'],
                'Order Qty':           r['order_qty'],
                'Unmet Demand':        r['unmet_demand'],
                'Inventory (End)':     r['inventory'],
                'Holding Cost (€)':    r['holding_cost_total'],
                'Ordering Cost (€)':   r['ordering_cost_total'],
                'Lost Sales Cost (€)': r['lost_sales_cost_total'],
                'Total Cost (€)':      -r['reward'],
            } for r in bs_recs]
            pd.DataFrame(bs_rows).to_excel(writer, sheet_name=f'BS S={S}'[:31], index=False)

    print(f"Results exported to {out_path}")


def evaluate_model(model, env, week_labels, future_week_labels, lead_time, verbose=True):
    all_week_labels = week_labels + future_week_labels

    def arrival_label(order_step_idx):
        idx = order_step_idx + lead_time
        if idx < len(all_week_labels):
            return str(all_week_labels[idx])
        return f"+{idx - len(all_week_labels) + 1}wk"

    obs, _ = env.reset()
    total_cost = 0
    records = []

    if verbose:
        header = (f"{'Week':<7} | {'Demand':<6} | {'Arrived':<7} | {'Order':<5} | "
                  f"{'Due':<7} | {'Unmet':<5} | {'Inv':<5} | "
                  f"{'Hold':<7} | {'OrdC':<7} | {'LostC':<8} | {'Cost':<9}")
        print(f"\n--- Evaluation on Historical Data --- Lead Time: {lead_time} weeks ---")
        print(header)
        print("-" * len(header))

    terminated = False
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        step_idx = env.current_step
        obs, _, terminated, _, info = env.step(action)
        total_cost -= info['reward']
        week_val = week_labels[step_idx] if step_idx < len(week_labels) else f"W{step_idx}"
        due_val = arrival_label(step_idx)
        records.append({'week': week_val, 'due': due_val, **info})
        if verbose:
            print(f"{week_val:<7} | {info['actual_demand']:<6} | {info['arriving_qty']:<7} | "
                  f"{info['order_qty']:<5} | {due_val:<7} | {info['unmet_demand']:<5} | "
                  f"{info['inventory']:<5} | {info['holding_cost_total']:<7.0f} | "
                  f"{info['ordering_cost_total']:<7.0f} | {info['lost_sales_cost_total']:<8.0f} | "
                  f"{-info['reward']:<9.0f}")

    if verbose:
        print("-" * len(header))
        print(f"Total Episode Cost: €{total_cost:,.2f}")
        print("--------------------------------------")

    return records, total_cost, env.inventory, list(env.pipeline), arrival_label


def save_run_artifacts(run_dir, config, model, records, future_records, product, location,
                       total_cost, lead_time, started_at, finished_at, duration_seconds,
                       hist_demand=None, hist_week_labels=None, base_stock_results_data=None,
                       per_scenario_records=None):

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    hist_demand = list(hist_demand) if hist_demand is not None else []
    hist_week_labels = [str(w) for w in hist_week_labels] if hist_week_labels is not None else []
    base_stock_results_data = base_stock_results_data or []
    per_scenario_records = per_scenario_records or {}
    _sync_base_stock_global(base_stock_results_data)

    kpis = compute_kpis(records)
    config_payload = {
        **asdict(config),
        'lead_time': lead_time,
        'started_at': started_at,
        'finished_at': finished_at,
        'duration_seconds': duration_seconds,
        'total_cost': round(total_cost, 2),
        'service_level': round(kpis['service_level'], 2),
        'total_ordered': kpis['total_ordered'],
        'avg_inventory': round(kpis['avg_inventory'], 2),
        'forecast_weeks': len(records),
    }
    with open(run_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=2, default=_json_default)

    model.save(str(run_dir / 'model'))

    export_to_excel(
        records, future_records, product, location, total_cost,
        out_path=str(run_dir / 'results.xlsx'),
        hist_demand=hist_demand, hist_week_labels=hist_week_labels,
    )
    visualize_results(
        records, product, location, future_records=future_records,
        out_path=str(run_dir / 'results.png'), show=False,
        hist_demand=hist_demand, hist_week_labels=hist_week_labels,
    )

    records_payload = {
        'product': product,
        'location': location,
        'lead_time': lead_time,
        'records': _serialize_records(records),
        'future_records': _serialize_records(future_records),
        'hist_demand': [int(v) for v in hist_demand],
        'hist_week_labels': hist_week_labels,
        'kpis': {
            'total_cost': total_cost,
            'service_level': kpis['service_level'],
            'total_ordered': kpis['total_ordered'],
            'avg_inventory': kpis['avg_inventory'],
            'forecast_weeks': len(records),
        },
        'base_stock_results': _serialize_base_stock_results(base_stock_results_data),
    }
    if per_scenario_records:
        records_payload['per_scenario_records'] = {
            sc: _serialize_records(recs) for sc, recs in per_scenario_records.items()
        }
    with open(run_dir / 'records.json', 'w', encoding='utf-8') as f:
        json.dump(records_payload, f, indent=2, default=_json_default)

    print(f"Run artifacts saved to {run_dir}")


def _average_records(all_records_list):
    """Average per-week records across multiple scenario simulations (week-by-week mean)."""
    n_weeks = min(len(r) for r in all_records_list)
    numeric_keys = [
        'actual_demand', 'arriving_qty', 'inventory_after_arrival', 'order_qty',
        'unmet_demand', 'inventory', 'reward',
        'holding_cost_total', 'ordering_cost_total', 'lost_sales_cost_total',
    ]
    averaged = []
    for i in range(n_weeks):
        row = {k: v for k, v in all_records_list[0][i].items() if k not in numeric_keys}
        for k in numeric_keys:
            vals = [r[i][k] for r in all_records_list if k in r[i]]
            row[k] = float(np.mean(vals))
        averaged.append(row)
    return averaged


def run_training_pipeline(config, progress_callback=None, run_dir=None, verbose=True):
    started_at = datetime.now()
    started_at_iso = started_at.isoformat()

    if verbose:
        print(f"--- Inventory Optimization for {config.product} at {config.location} ---")

    # Resolve scenarios: use config.scenarios if set, otherwise fall back to first available
    all_scenarios = list_scenarios(config.file_path)
    active_scenarios = config.scenarios if config.scenarios else (all_scenarios[:1] or [None])

    # Load demand/lead-time/inventory using the first active scenario (same across scenarios)
    first_scenario = active_scenarios[0] if active_scenarios[0] in all_scenarios else None
    (demand_data, forecast_data, lead_time, initial_inventory, week_labels,
     future_forecast, future_week_labels) = load_data(
        config.file_path, config.product, config.location, scenario=first_scenario,
    )

    effective_max_order_qty = config.max_order_qty
    suggested_qty = suggest_max_order_qty(demand_data)
    peak_demand = int(np.max(demand_data))

    if effective_max_order_qty <= 0:
        # Auto mode: derive from data
        effective_max_order_qty = suggested_qty
        if verbose:
            print(
                f"[auto] max_order_qty set to {effective_max_order_qty} "
                f"(3× peak demand {peak_demand})"
            )
    elif effective_max_order_qty > suggested_qty * 5:
        # Too large: agent must hit a needle-in-a-haystack action range, preventing convergence.
        effective_max_order_qty = suggested_qty
        if verbose:
            print(
                f"[auto] max_order_qty {config.max_order_qty} >> "
                f"suggested {suggested_qty} (3× peak demand {peak_demand}); "
                f"using {effective_max_order_qty}"
            )
    elif effective_max_order_qty < peak_demand:
        # Too small: agent cannot cover peak demand in a single order.
        effective_max_order_qty = suggested_qty
        if verbose:
            print(
                f"[auto] max_order_qty {config.max_order_qty} < peak demand {peak_demand}; "
                f"using {effective_max_order_qty}"
            )

    # When forecast covers only future weeks (no overlap with demand period), use demand
    # as the training forecast so the agent's lookahead observation stays aligned with history.
    has_historical_forecast = len(forecast_data) > len(future_forecast)
    training_forecast = forecast_data if has_historical_forecast else demand_data

    env = SingleEchelonEnv(
        demand_data=demand_data,
        forecast_data=training_forecast,
        lead_time=lead_time,
        initial_inventory=initial_inventory,
        holding_cost=config.holding_cost,
        ordering_cost=config.ordering_cost,
        lost_sales_cost=config.lost_sales_cost,
        max_order_qty=effective_max_order_qty,
        n_forecast_weeks=config.n_forecast_weeks,
    )

    # Cap n_steps so the buffer never dwarfs the entire episode count available in
    # total_timesteps — at least 10 PPO updates should happen for any useful learning.
    episode_len = env.max_steps
    min_updates = 10
    max_n_steps = max(config.batch_size, config.timesteps // min_updates)
    effective_n_steps = min(config.n_steps, max_n_steps)
    # Round down to nearest multiple of episode_len for clean rollout boundaries
    if episode_len > 0 and effective_n_steps > episode_len:
        effective_n_steps = max(config.batch_size, (effective_n_steps // episode_len) * episode_len)
    if verbose and effective_n_steps != config.n_steps:
        print(
            f"[auto] n_steps reduced {config.n_steps} -> {effective_n_steps} "
            f"(episode_len={episode_len}, timesteps={config.timesteps}) "
            f"to ensure >={min_updates} PPO updates"
        )

    callbacks = []
    if progress_callback:
        callbacks.append(ProgressCallback(config.timesteps, on_progress=progress_callback))

    if verbose:
        print(f"Starting training for {config.timesteps} steps...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=config.verbose,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        n_steps=effective_n_steps,
        batch_size=config.batch_size,
    )
    model.learn(total_timesteps=config.timesteps, callback=callbacks or None)
    if verbose:
        print("Training complete.")

    # Run future projection for each selected scenario, then average week-by-week
    per_scenario_records = {}
    scenario_records_list = []
    for sc in active_scenarios:
        sc_key = sc if sc in all_scenarios else None
        _, _, _, _, _, sc_future_forecast, sc_future_week_labels = load_data(
            config.file_path, config.product, config.location, scenario=sc_key,
        )
        if verbose and len(sc_future_forecast) > 0:
            print(f"\n--- Inventory Planning — Scenario: {sc_key or 'default'} "
                  f"({len(sc_future_forecast)} weeks) --- Lead Time: {lead_time} weeks ---")
        sc_records = run_future_projection(
            model, initial_inventory, [0] * lead_time,
            sc_future_forecast, sc_future_week_labels,
            lead_time=lead_time,
            n_forecast_weeks=config.n_forecast_weeks,
            holding_cost=config.holding_cost,
            ordering_cost=config.ordering_cost,
            lost_sales_cost=config.lost_sales_cost,
            max_order_qty=effective_max_order_qty,
        )
        # Annotate due-week labels
        for i, r in enumerate(sc_records):
            arr_idx = i + lead_time
            if arr_idx < len(sc_future_week_labels):
                r['due'] = str(sc_future_week_labels[arr_idx])
            else:
                r['due'] = f"+{arr_idx - len(sc_future_week_labels) + 1}wk"
        if sc_records:
            label = sc_key or 'default'
            per_scenario_records[label] = sc_records
            scenario_records_list.append(sc_records)

    if len(scenario_records_list) > 1:
        records = _average_records(scenario_records_list)
        if verbose:
            print(f"\n[scenarios] Averaged results across {len(scenario_records_list)} scenarios: "
                  f"{active_scenarios}")
    elif scenario_records_list:
        records = scenario_records_list[0]
    else:
        records = []

    total_cost = float(sum(-r['reward'] for r in records))
    future_records = []

    bs_results = compute_base_stock_baselines(
        config, demand_data, lead_time, initial_inventory,
        active_scenarios=active_scenarios, all_scenarios=all_scenarios,
    )
    _sync_base_stock_global(bs_results)

    if verbose and bs_results:
        avg_demand = float(np.mean(demand_data))
        print(f"\n--- Base Stock Baselines (avg demand={avg_demand:.1f}) ---")
        for b in bs_results:
            kpi = b['kpis']
            print(f"  S={b['S']} ({b['variant']}): Cost €{kpi['Total Cost (€)']:,.2f}, "
                  f"Service Level {kpi['Service Level (%)']:.1f}%")

    if verbose and records:
        fut_header = (f"{'Week':<7} | {'Forecast':<8} | {'Arrived':<7} | {'Order':<5} | "
                      f"{'Due':<7} | {'Inv':<5} | {'Hold':<7} | {'OrdC':<7} | {'LostC':<8}")
        print(fut_header)
        print("-" * len(fut_header))
        for r in records:
            print(f"{r['week']:<7} | {r['actual_demand']:<8} | {r['arriving_qty']:<7} | "
                  f"{r['order_qty']:<5} | {r.get('due', ''):<7} | {r['inventory']:<5} | "
                  f"{r['holding_cost_total']:<7.0f} | {r['ordering_cost_total']:<7.0f} | "
                  f"{r['lost_sales_cost_total']:<8.0f}")
        print("-" * len(fut_header))

    finished_at = datetime.now()
    finished_at_iso = finished_at.isoformat()
    duration_seconds = (finished_at - started_at).total_seconds()

    if run_dir is None:
        run_dir = make_run_dir(config.product)

    hist_demand_list = demand_data.tolist()
    hist_week_labels_list = [str(w) for w in week_labels]

    save_run_artifacts(
        run_dir, config, model, records, future_records,
        config.product, config.location, total_cost, lead_time,
        started_at_iso, finished_at_iso, duration_seconds,
        hist_demand=hist_demand_list, hist_week_labels=hist_week_labels_list,
        base_stock_results_data=bs_results,
        per_scenario_records=per_scenario_records,
    )

    kpis = compute_kpis(records)
    return RunResult(
        config=config,
        records=records,
        future_records=future_records,
        product=config.product,
        location=config.location,
        lead_time=lead_time,
        total_cost=total_cost,
        service_level=kpis['service_level'],
        total_ordered=kpis['total_ordered'],
        avg_inventory=kpis['avg_inventory'],
        run_dir=Path(run_dir),
        started_at=started_at_iso,
        finished_at=finished_at_iso,
        duration_seconds=duration_seconds,
        hist_demand=hist_demand_list,
        hist_week_labels=hist_week_labels_list,
        base_stock_results=bs_results,
        model=model,
        per_scenario_records=per_scenario_records,
    )


def visualize_results(records, product, location, future_records=None, out_path='results.png', show=True,
                      hist_demand=None, hist_week_labels=None):
    future_records = future_records or []
    hist_demand = list(hist_demand) if hist_demand else []
    hist_week_labels = [str(w) for w in hist_week_labels] if hist_week_labels else []

    all_agent = records + future_records

    def col(key):
        return [r[key] for r in all_agent]

    n_planning = len(records)

    demand_agent            = np.array(col('actual_demand'),           dtype=float)
    orders                  = np.array(col('order_qty'),               dtype=float)
    inventory               = np.array(col('inventory'),               dtype=float)
    inventory_after_arrival = np.array(col('inventory_after_arrival'), dtype=float)
    unmet                   = np.array(col('unmet_demand'),            dtype=float)
    hold_c = np.array(col('holding_cost_total'),    dtype=float)
    ord_c  = np.array(col('ordering_cost_total'),   dtype=float)
    lost_c = np.array(col('lost_sales_cost_total'), dtype=float)
    rewards  = np.array(col('reward'), dtype=float)
    cum_cost = np.cumsum(-rewards)

    if hist_demand:
        # New mode: historical demand bars + future planning
        n_hist = len(hist_demand)
        weeks = hist_week_labels + [r['week'] for r in all_agent]
        x = list(range(len(weeks)))
        x_hist = list(range(n_hist))
        x_fut  = list(range(n_hist, n_hist + len(all_agent)))
        # KPIs from planning records only
        total_cost    = float(-rewards[:n_planning].sum())
        service_level = 100.0 * (1 - unmet[:n_planning].sum() / max(demand_agent[:n_planning].sum(), 1))
        total_ordered = int(orders[:n_planning].sum())
        avg_inventory = float(inventory[:n_planning].mean()) if n_planning else 0.0
    else:
        # Legacy mode: records shown as historical, future_records as projected
        n_hist = len(records)
        weeks  = [r['week'] for r in all_agent]
        x      = list(range(len(weeks)))
        x_hist = list(range(n_hist))
        x_fut  = list(range(n_hist, len(all_agent)))
        total_cost    = float(-rewards[:n_hist].sum())
        service_level = 100.0 * (1 - unmet[:n_hist].sum() / max(demand_agent[:n_hist].sum(), 1))
        total_ordered = int(orders[:n_hist].sum())
        avg_inventory = float(inventory[:n_hist].mean()) if n_hist else 0.0

    _dpi, fig_w, fig_h = 100, 20, 12

    theme = _load_chart_theme()
    BG = theme['BG']
    PANEL = theme['PANEL']
    GRID = theme['GRID']
    TEXT = theme['TEXT']
    MUTED = theme['MUTED']
    C_INV = theme['C_INV']
    C_DEM = theme['C_DEM']
    C_UNM = theme['C_UNM']
    C_ORD = theme['C_ORD']
    C_HLD = theme['C_HLD']
    C_ORC = theme['C_ORC']
    C_LST = theme['C_LST']
    C_HOR = theme['C_HOR']
    FUT_SHADE = theme['FUT_SHADE']
    font_family = theme['FONT_FAMILY']

    plt.rcParams.update({
        'font.family':       font_family,
        'font.sans-serif':   [font_family, 'DejaVu Sans', 'sans-serif'],
        'font.size':         10,
        'axes.facecolor':    PANEL,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   MUTED,
        'axes.titlecolor':   TEXT,
        'axes.titlesize':    10.5,
        'axes.titleweight':  'semibold',
        'axes.titlepad':     8,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.spines.left':  True,
        'axes.spines.bottom':True,
        'axes.grid':         True,
        'axes.grid.axis':    'y',
        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'grid.linestyle':    '-',
        'xtick.color':       MUTED,
        'ytick.color':       MUTED,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.frameon':    True,
        'legend.framealpha': 0.92,
        'legend.edgecolor':  GRID,
        'legend.fontsize':   8,
    })

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG, dpi=_dpi)
    gs  = fig.add_gridspec(
        5, 1,
        height_ratios=[0.20, 1.35, 0.85, 0.85, 0.85],
        hspace=0.55,
        left=0.07, right=0.97, top=0.93, bottom=0.08,
    )

    title_suffix = '  ·  vs Base Stock' if base_stock_results else ''
    fig.text(0.5, 0.968, f'PPO Inventory Policy{title_suffix}  ·  {product}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.948, location,
             ha='center', va='top', fontsize=9, color=MUTED)

    ax_kpi = fig.add_subplot(gs[0])
    ax_kpi.set_facecolor(BG)
    for sp in ax_kpi.spines.values(): sp.set_visible(False)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

    n_planning_weeks = n_planning if hist_demand else n_hist
    kpis = [
        ('Total Cost',       f'€{total_cost:,.0f}'),
        ('Service Level',    f'{service_level:.1f}%'),
        ('Total Ordered',    f'{total_ordered:,} units'),
        ('Avg Inventory',    f'{avg_inventory:,.0f} units'),
        ('Planning Weeks',   str(n_planning_weeks)),
    ]
    for i, (label, value) in enumerate(kpis):
        cx = (i + 0.5) / len(kpis)
        ax_kpi.text(cx, 0.78, value, transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', color=TEXT)
        ax_kpi.text(cx, 0.18, label, transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=7.5, color=MUTED, style='italic')

    ax0 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(gs[2], sharex=ax0)
    ax2 = fig.add_subplot(gs[3], sharex=ax0)
    ax3 = fig.add_subplot(gs[4], sharex=ax0)
    plt.setp([ax0.get_xticklabels(), ax1.get_xticklabels(),
              ax2.get_xticklabels()], visible=False)

    fmt_k = mticker.FuncFormatter(lambda v, _: f'{int(v):,}')
    fmt_e = mticker.FuncFormatter(lambda v, _: f'€{int(v):,}')

    def shade_future(ax):
        if x_fut:
            ax.axvspan(n_hist - 0.5, x_fut[-1] + 0.5,
                       alpha=0.07, color=FUT_SHADE, zorder=0)
            ax.axvline(n_hist - 0.5, color=C_HOR, linewidth=1.2,
                       linestyle='--', alpha=0.85, zorder=1, label='Forecast horizon')

    # ── Panel 1: Inventory & Demand ───────────────────────────────────────────
    ax = ax0
    shade_future(ax)
    if hist_demand and x_hist:
        # Historical section: actual demand bars only
        ax.bar(x_hist, hist_demand, color=C_DEM, alpha=0.28, zorder=2, label='Actual demand')
        # Agent planning section: forecast demand bars + inventory
        if x_fut:
            ax.bar(x_fut, demand_agent, color=C_DEM, alpha=0.18,
                   hatch='//', zorder=2, label='Forecast demand')
            ax.bar(x_fut, unmet, color=C_UNM, alpha=0.85, zorder=3, label='Unmet demand')
        if x_fut:
            ax.fill_between(x_fut, inventory_after_arrival, alpha=0.15, color=C_INV, zorder=4)
            ax.plot(x_fut, inventory_after_arrival, color=C_INV, linewidth=2.2, zorder=5,
                    label='PPO inventory')
            for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
                bs_inv = np.array([r['inventory_after_arrival'] for r in bs_recs], dtype=float)
                if len(bs_inv) == len(x_fut):
                    ax.plot(x_fut, bs_inv, color=color, linewidth=1.5, linestyle='--',
                            alpha=0.85, zorder=6, label=f'BS S={S}')
    else:
        # Legacy mode
        ax.bar(x_hist, demand_agent[:n_hist], color=C_DEM, alpha=0.28, zorder=2, label='Actual demand')
        ax.bar(x_hist, unmet[:n_hist],  color=C_UNM, alpha=0.85, zorder=3, label='Unmet demand')
        if x_fut:
            ax.bar(x_fut, demand_agent[n_hist:], color=C_DEM, alpha=0.18,
                   hatch='//', zorder=2, label='Forecast demand')
        ax.fill_between(x, inventory_after_arrival, alpha=0.15, color=C_INV, zorder=4)
        ax.plot(x, inventory_after_arrival, color=C_INV, linewidth=2.2, zorder=5,
                label='PPO inventory')
        for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
            bs_inv = np.array([r['inventory_after_arrival'] for r in bs_recs], dtype=float)
            ax.plot(x_hist, bs_inv, color=color, linewidth=1.5, linestyle='--',
                    alpha=0.85, zorder=6, label=f'BS S={S}')
    ax.set_ylabel('Units')
    ax.set_title('Inventory Level vs Demand')
    ax.legend(loc='upper right', ncol=max(1, (3 if not hist_demand else 2) + len(base_stock_results)))
    ax.yaxis.set_major_formatter(fmt_k)

    # ── Panel 2: Order Quantities ─────────────────────────────────────────────
    ax = ax1
    shade_future(ax)
    if hist_demand:
        # Only show planned orders in future period
        if x_fut:
            ax.bar(x_fut, orders, color=C_ORD, alpha=0.85, label='Planned order qty (PPO)')
            for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
                bs_ord = np.array([r['order_qty'] for r in bs_recs], dtype=float)
                if len(bs_ord) == len(x_fut):
                    ax.plot(x_fut, bs_ord, color=color, linewidth=1.5, linestyle='--',
                            alpha=0.85, zorder=6, label=f'BS S={S} orders')
    else:
        # Legacy mode
        ax.bar(x_hist, orders[:n_hist], color=C_ORD, alpha=0.85, label='Order qty (PPO)')
        if x_fut:
            ax.bar(x_fut, orders[n_hist:], color=C_ORD, alpha=0.35,
                   hatch='//', label='Projected order qty')
    ax.set_ylabel('Units ordered')
    ax.set_title('Weekly Order Quantities (PPO)')
    ax.legend(loc='upper right', ncol=1)
    ax.yaxis.set_major_formatter(fmt_k)

    # ── Panel 3: Cost Breakdown ───────────────────────────────────────────────
    ax = ax2
    shade_future(ax)
    if hist_demand:
        # Only show costs for planning period
        if x_fut:
            bh  = hold_c
            bo  = hold_c + ord_c
            ax.bar(x_fut, hold_c, color=C_HLD, alpha=0.90, label='Holding')
            ax.bar(x_fut, ord_c,  color=C_ORC, alpha=0.90, bottom=bh,  label='Ordering')
            ax.bar(x_fut, lost_c, color=C_LST, alpha=0.90, bottom=bo,  label='Lost sales')
            for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
                bs_hold = np.array([r['holding_cost_total'] for r in bs_recs], dtype=float)
                bs_ord = np.array([r['ordering_cost_total'] for r in bs_recs], dtype=float)
                bs_lost = np.array([r['lost_sales_cost_total'] for r in bs_recs], dtype=float)
                if len(bs_hold) == len(x_fut):
                    bs_total = bs_hold + bs_ord + bs_lost
                    ax.plot(x_fut, bs_total, color=color, linewidth=1.5, linestyle='--',
                            alpha=0.85, zorder=6, label=f'BS S={S} total cost')
    else:
        # Legacy mode
        bh = hold_c[:n_hist]
        bo = hold_c[:n_hist] + ord_c[:n_hist]
        ax.bar(x_hist, hold_c[:n_hist], color=C_HLD, alpha=0.90, label='Holding')
        ax.bar(x_hist, ord_c[:n_hist],  color=C_ORC, alpha=0.90, bottom=bh,  label='Ordering')
        ax.bar(x_hist, lost_c[:n_hist], color=C_LST, alpha=0.90, bottom=bo,  label='Lost sales')
        if x_fut:
            bf  = hold_c[n_hist:]
            bof = hold_c[n_hist:] + ord_c[n_hist:]
            ax.bar(x_fut, hold_c[n_hist:], color=C_HLD, alpha=0.35, hatch='//')
            ax.bar(x_fut, ord_c[n_hist:],  color=C_ORC, alpha=0.35, hatch='//', bottom=bf)
            ax.bar(x_fut, lost_c[n_hist:], color=C_LST, alpha=0.35, hatch='//', bottom=bof)
    ax.set_ylabel('Cost (€)')
    ax.set_title('Weekly Cost Breakdown (PPO)')
    ax.legend(loc='upper right', ncol=3)
    ax.yaxis.set_major_formatter(fmt_e)

    # ── Panel 4: Cumulative Cost ──────────────────────────────────────────────
    ax = ax3
    shade_future(ax)
    if hist_demand:
        # Only cumulative cost for planning period
        if x_fut:
            ax.fill_between(x_fut, cum_cost, alpha=0.12, color=C_LST)
            ax.plot(x_fut, cum_cost, color=C_LST, linewidth=2.2, label='PPO (planned)')
            for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
                bs_cum = np.cumsum([-r['reward'] for r in bs_recs])
                if len(bs_cum) == len(x_fut):
                    ax.plot(x_fut, bs_cum, color=color, linewidth=1.5, linestyle='--',
                            alpha=0.85, zorder=4, label=f'BS S={S}')
    else:
        # Legacy mode
        ax.fill_between(x[:n_hist], cum_cost[:n_hist], alpha=0.12, color=C_LST)
        ax.plot(x[:n_hist], cum_cost[:n_hist], color=C_LST, linewidth=2.2, label='PPO')
        if x_fut:
            jx = x[n_hist - 1:]; jy = cum_cost[n_hist - 1:]
            ax.fill_between(jx, jy, alpha=0.06, color=C_LST)
            ax.plot(jx, jy, color=C_LST, linewidth=2.2, linestyle='--', alpha=0.55,
                    label='PPO (projected)')
        for (S, bs_recs), color in zip(base_stock_results, BS_COLORS):
            bs_cum = np.cumsum([-r['reward'] for r in bs_recs])
            ax.plot(x_hist, bs_cum, color=color, linewidth=1.5, linestyle='--',
                    alpha=0.85, zorder=4, label=f'BS S={S}')
    ax.set_ylabel('Cumulative cost (€)')
    ax.set_title('Cumulative Cost (Planning Period)')
    ax.legend(loc='upper left', ncol=1 + len(base_stock_results))
    ax.yaxis.set_major_formatter(fmt_e)

    tick_step = max(1, len(weeks) // 24)
    ax3.set_xticks(x[::tick_step])
    ax3.set_xticklabels([str(w) for w in weeks[::tick_step]],
                        rotation=40, ha='right', fontsize=7.5)
    ax3.set_xlabel('Week', color=MUTED)
    ax3.set_xlim(-0.5, len(weeks) - 0.5)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\nVisualization saved to {out_path}")

    if show:
        try:
            plt.get_current_fig_manager().window.state('zoomed')   # TkAgg / Windows
        except Exception:
            try:
                plt.get_current_fig_manager().window.showMaximized()  # Qt backends
            except Exception:
                pass
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Inventory Optimization using PPO')
    parser.add_argument('--file-path', type=str, default=DEFAULT_FILE_PATH,
                        help='Path to the Excel data file')
    parser.add_argument('--product',    type=str, default='Ice Cream Strawberry Flavor',
                        help='Name of the product')
    parser.add_argument('--location',   type=str, default='Logistics Hub Lissabon',
                        help='Location of the warehouse')
    parser.add_argument('--timesteps',  type=int, default=10000,
                        help='Number of training timesteps')
    args = parser.parse_args()

    config = TrainingConfig(
        file_path=args.file_path,
        product=args.product,
        location=args.location,
        timesteps=args.timesteps,
        verbose=1,
    )
    try:
        result = run_training_pipeline(config)
        visualize_results(
            result.records, result.product, result.location,
            future_records=result.future_records, show=True,
        )
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
