import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import warnings

warnings.simplefilter("ignore")

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
        # Forecasts for the next N weeks
        forecasts = self.forecast_data[self.current_step : self.current_step + self.n_forecast_weeks]
        # Ensure we have enough forecasts, pad if necessary
        if len(forecasts) < self.n_forecast_weeks:
            forecasts = np.pad(forecasts, (0, self.n_forecast_weeks - len(forecasts)), 'edge')
        
        obs = np.concatenate([
            [float(self.inventory)],
            [float(x) for x in self.pipeline],
            [float(x) for x in forecasts]
        ]).astype(np.float32)
        
        # Simple normalization: divide by a constant (e.g., 100) to keep values in a reasonable range for NN
        return obs / 100.0

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # 1. Receive incoming order from pipeline (Lead Time delay)
        arriving_qty = self.pipeline.pop(0)
        self.inventory += arriving_qty
        inventory_after_arrival = self.inventory  # pre-demand level, includes delivery

        # 2. Observe actual demand for the current week
        actual_demand = self.demand_data[self.current_step]

        # 3. Satisfy demand (Lost Sales Model: no backlogging)
        if self.inventory >= actual_demand:
            unmet_demand = 0
            self.inventory -= actual_demand
        else:
            unmet_demand = actual_demand - self.inventory
            self.inventory = 0

        # 4. Agent's Action (Order Quantity)
        # Action is [0, 1], scale to [0, max_order_qty] and round to integer (1,000 KG units)
        order_qty = int(round(float(action[0]) * self.max_order_qty))

        # 5. Calculate Reward (Negative of Total Costs)
        # Holding Cost: 13 € per unit
        # Ordering Cost: 60 € per unit
        # Lost Sales Cost: 2500 € per unit
        reward = -(
            (self.holding_cost * self.inventory) +
            (self.ordering_cost * order_qty) +
            (self.lost_sales_cost * unmet_demand)
        )

        # 6. Add new order to the end of the pipeline
        self.pipeline.append(order_qty)

        # 7. Update step and check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "actual_demand": actual_demand,
            "arriving_qty": arriving_qty,
            "inventory_after_arrival": inventory_after_arrival,
            "order_qty": order_qty,
            "unmet_demand": unmet_demand,
            "inventory": self.inventory,
            "reward": reward,
            "holding_cost_total": self.holding_cost * self.inventory,
            "ordering_cost_total": self.ordering_cost * order_qty,
            "lost_sales_cost_total": self.lost_sales_cost * unmet_demand
        }

        # Return scaled reward for better PPO convergence
        return self._get_obs(), reward / 1000.0, terminated, truncated, info

def load_data(file_path, product, location):
    """
    Load data from the Excel file and handle potential whitespace in sheet/column names.
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

    df_demand = pd.read_excel(file_path, sheet_name=get_sheet_name('Demand'))
    df_inventory = pd.read_excel(file_path, sheet_name=get_sheet_name('Current Inventory'))
    df_lead_time = pd.read_excel(file_path, sheet_name=get_sheet_name('Lead Time'))
    df_forecast = pd.read_excel(file_path, sheet_name=get_sheet_name('Forecast'))

    for df in [df_demand, df_inventory, df_lead_time, df_forecast]:
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    week_labels = df_demand.columns[2:].tolist()

    demand_row = df_demand[(df_demand['Product'] == product) & (df_demand['Location'] == location)]
    if demand_row.empty:
        raise ValueError(f"No demand data found for {product} at {location}")
    demand = demand_row.iloc[0, 2:].values.astype(int)

    inventory_row = df_inventory[(df_inventory['Product'] == product) & (df_inventory['Location'] == location)]
    if inventory_row.empty:
        raise ValueError(f"No inventory data found for {product} at {location}")
    inventory = inventory_row.iloc[0, 2]

    lt_row = df_lead_time[(df_lead_time['Product'] == product) & (df_lead_time['Location'] == location)]
    if lt_row.empty:
        raise ValueError(f"No lead time data found for {product} at {location}")
    lead_time = int(lt_row['Lead Time in weeks'].values[0])

    forecast_row = df_forecast[(df_forecast['Product'] == product) & (df_forecast['Location'] == location)]
    if forecast_row.empty:
        raise ValueError(f"No forecast data found for {product} at {location}")
    forecast = forecast_row.iloc[0, 2:].values.astype(int)

    # Future weeks: forecast columns that are not present in the demand period
    demand_week_set = set(str(w) for w in week_labels)
    future_week_labels = [w for w in df_forecast.columns[2:].tolist()
                          if str(w) not in demand_week_set]
    if future_week_labels:
        future_forecast = forecast_row[future_week_labels].values[0].astype(int)
    else:
        future_forecast = np.array([], dtype=int)
        future_week_labels = []

    return demand, forecast, lead_time, inventory, week_labels, future_forecast, future_week_labels

import argparse


def run_future_projection(model, final_inventory, final_pipeline, future_forecast,
                           future_week_labels, lead_time, n_forecast_weeks=4,
                           holding_cost=13, ordering_cost=60, lost_sales_cost=2500,
                           max_order_qty=200):
    if len(future_forecast) == 0:
        print("No future forecast weeks found in the data — skipping forward projection.")
        return []

    # Pad so _get_obs() always has n_forecast_weeks to look ahead
    padded = np.append(future_forecast, np.full(n_forecast_weeks, future_forecast[-1]))

    env = SingleEchelonEnv(
        demand_data=padded,
        forecast_data=padded,
        lead_time=lead_time,
        initial_inventory=int(final_inventory),
        holding_cost=holding_cost,
        ordering_cost=ordering_cost,
        lost_sales_cost=lost_sales_cost,
        max_order_qty=max_order_qty,
        n_forecast_weeks=n_forecast_weeks,
    )
    obs, _ = env.reset()
    # Override with real end-of-evaluation state
    env.inventory = int(final_inventory)
    env.pipeline = list(final_pipeline)
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


def visualize_results(records, product, location, future_records=None):
    future_records = future_records or []
    n_hist      = len(records)
    all_records = records + future_records

    def col(key):
        return [r[key] for r in all_records]

    weeks                  = col('week')
    demand                 = np.array(col('actual_demand'),          dtype=float)
    orders                 = np.array(col('order_qty'),              dtype=float)
    inventory              = np.array(col('inventory'),              dtype=float)
    inventory_after_arrival= np.array(col('inventory_after_arrival'),dtype=float)
    unmet                  = np.array(col('unmet_demand'),           dtype=float)
    hold_c     = np.array(col('holding_cost_total'), dtype=float)
    ord_c      = np.array(col('ordering_cost_total'),dtype=float)
    lost_c     = np.array(col('lost_sales_cost_total'), dtype=float)
    rewards  = np.array(col('reward'), dtype=float)
    cum_cost = np.cumsum(-rewards)   # positive, growing cost

    x      = list(range(len(weeks)))
    x_hist = list(range(n_hist))
    x_fut  = list(range(n_hist, len(all_records)))

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_cost    = float(-rewards[:n_hist].sum())
    service_level = 100.0 * (1 - unmet[:n_hist].sum() / max(demand[:n_hist].sum(), 1))
    total_ordered = int(orders[:n_hist].sum())
    avg_inventory = float(inventory[:n_hist].mean()) if n_hist else 0.0

    # ── figure size (window is maximized on open, so this is just the initial canvas) ──
    _dpi, fig_w, fig_h = 100, 20, 12

    # ── palette ───────────────────────────────────────────────────────────────
    BG     = '#F0F4FA'
    PANEL  = '#FFFFFF'
    GRID   = '#DDE3EE'
    TEXT   = '#1E293B'
    MUTED  = '#64748B'
    C_INV  = '#3B82F6'
    C_DEM  = '#F87171'
    C_UNM  = '#DC2626'
    C_ORD  = '#FB923C'
    C_HLD  = '#60A5FA'
    C_ORC  = '#FBBF24'
    C_LST  = '#F43F5E'
    C_REW  = '#10B981'
    C_HOR  = '#D97706'

    plt.rcParams.update({
        'font.family':       'sans-serif',
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

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.968, f'PPO Inventory Policy  ·  {product}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.948, location,
             ha='center', va='top', fontsize=9, color=MUTED)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0])
    ax_kpi.set_facecolor(BG)
    for sp in ax_kpi.spines.values(): sp.set_visible(False)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])

    kpis = [
        ('Total Cost (hist.)', f'€{total_cost:,.0f}'),
        ('Service Level',      f'{service_level:.1f}%'),
        ('Total Ordered',      f'{total_ordered:,} units'),
        ('Avg Inventory',      f'{avg_inventory:,.0f} units'),
        ('Historical Weeks',   str(n_hist)),
        ('Projected Weeks',    str(len(future_records))),
    ]
    for i, (label, value) in enumerate(kpis):
        cx = (i + 0.5) / len(kpis)
        ax_kpi.text(cx, 0.78, value, transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', color=TEXT)
        ax_kpi.text(cx, 0.18, label, transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=7.5, color=MUTED, style='italic')

    # ── chart axes (share x) ──────────────────────────────────────────────────
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
                       alpha=0.07, color='#FEF9C3', zorder=0)
            ax.axvline(n_hist - 0.5, color=C_HOR, linewidth=1.2,
                       linestyle='--', alpha=0.85, zorder=1, label='Forecast horizon')

    # ── Panel 1: Inventory & Demand ───────────────────────────────────────────
    ax = ax0
    shade_future(ax)
    ax.bar(x_hist, demand[:n_hist], color=C_DEM, alpha=0.28, zorder=2, label='Actual demand')
    ax.bar(x_hist, unmet[:n_hist],  color=C_UNM, alpha=0.85, zorder=3, label='Unmet demand')
    if x_fut:
        ax.bar(x_fut, demand[n_hist:], color=C_DEM, alpha=0.18,
               hatch='//', zorder=2, label='Forecast demand')
    ax.fill_between(x, inventory_after_arrival, alpha=0.15, color=C_INV, zorder=4)
    ax.plot(x, inventory_after_arrival, color=C_INV, linewidth=2.2, zorder=5,
            label='Inventory (after arrival)')
    ax.set_ylabel('Units'); ax.set_title('Inventory Level vs Demand')
    ax.legend(loc='upper right', ncol=4 if x_fut else 3)
    ax.yaxis.set_major_formatter(fmt_k)

    # ── Panel 2: Order Quantities ─────────────────────────────────────────────
    ax = ax1
    shade_future(ax)
    ax.bar(x_hist, orders[:n_hist], color=C_ORD, alpha=0.85, label='Order qty')
    if x_fut:
        ax.bar(x_fut, orders[n_hist:], color=C_ORD, alpha=0.35,
               hatch='//', label='Projected order qty')
    ax.set_ylabel('Units ordered'); ax.set_title('Weekly Order Quantities')
    ax.legend(loc='upper right', ncol=2 if x_fut else 1)
    ax.yaxis.set_major_formatter(fmt_k)

    # ── Panel 3: Cost Breakdown ───────────────────────────────────────────────
    ax = ax2
    shade_future(ax)
    bh = hold_c[:n_hist]
    bo = hold_c[:n_hist] + ord_c[:n_hist]
    ax.bar(x_hist, hold_c[:n_hist], color=C_HLD, alpha=0.90, label='Holding')
    ax.bar(x_hist, ord_c[:n_hist],  color=C_ORC, alpha=0.90, bottom=bh,  label='Ordering')
    ax.bar(x_hist, lost_c[:n_hist], color=C_LST, alpha=0.90, bottom=bo,  label='Lost sales')
    if x_fut:
        bf = hold_c[n_hist:]
        bof= hold_c[n_hist:] + ord_c[n_hist:]
        ax.bar(x_fut, hold_c[n_hist:], color=C_HLD, alpha=0.35, hatch='//')
        ax.bar(x_fut, ord_c[n_hist:],  color=C_ORC, alpha=0.35, hatch='//', bottom=bf)
        ax.bar(x_fut, lost_c[n_hist:], color=C_LST, alpha=0.35, hatch='//', bottom=bof)
    ax.set_ylabel('Cost (€)'); ax.set_title('Weekly Cost Breakdown')
    ax.legend(loc='upper right', ncol=3)
    ax.yaxis.set_major_formatter(fmt_e)

    # ── Panel 4: Cumulative Cost ──────────────────────────────────────────────
    ax = ax3
    shade_future(ax)
    ax.fill_between(x[:n_hist], cum_cost[:n_hist], alpha=0.12, color=C_LST)
    ax.plot(x[:n_hist], cum_cost[:n_hist], color=C_LST,
            linewidth=2.2, label='Cumulative cost')
    if x_fut:
        jx = x[n_hist - 1:]; jy = cum_cost[n_hist - 1:]
        ax.fill_between(jx, jy, alpha=0.06, color=C_LST)
        ax.plot(jx, jy, color=C_LST, linewidth=2.2,
                linestyle='--', alpha=0.55, label='Projected')
    ax.set_ylabel('Cumulative cost (€)'); ax.set_title('Cumulative Cost')
    ax.legend(loc='upper left', ncol=2 if x_fut else 1)
    ax.yaxis.set_major_formatter(fmt_e)

    # ── x-axis ticks ──────────────────────────────────────────────────────────
    tick_step = max(1, len(weeks) // 24)
    ax3.set_xticks(x[::tick_step])
    ax3.set_xticklabels([str(w) for w in weeks[::tick_step]],
                        rotation=40, ha='right', fontsize=7.5)
    ax3.set_xlabel('Week', color=MUTED)
    ax3.set_xlim(-0.5, len(weeks) - 0.5)

    out_path = 'results.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"\nVisualization saved to {out_path}")

    try:
        plt.get_current_fig_manager().window.state('zoomed')   # TkAgg / Windows
    except Exception:
        try:
            plt.get_current_fig_manager().window.showMaximized()  # Qt backends
        except Exception:
            pass

    plt.show()


def main():
    # Configuration via CLI
    parser = argparse.ArgumentParser(description='Inventory Optimization using PPO')
    parser.add_argument('--file-path', type=str, default='Sample Data RL4IM UPDATED.xlsx',
                        help='Path to the Excel data file')
    parser.add_argument('--product', type=str, default='Ice Cream Strawberry Flavor',
                        help='Name of the product')
    parser.add_argument('--location', type=str, default='Logistics Hub Lissabon',
                        help='Location of the warehouse')
    parser.add_argument('--timesteps', type=int, default=10000,
                        help='Number of training timesteps')
    
    args = parser.parse_args()

    FILE_PATH = args.file_path
    PRODUCT = args.product
    LOCATION = args.location
    TRAIN_TIMESTEPS = args.timesteps
    
    print(f"--- Inventory Optimization for {PRODUCT} at {LOCATION} ---")
    
    # Load Real Data
    try:
        (demand_data, forecast_data, lead_time, initial_inventory, week_labels,
         future_forecast, future_week_labels) = load_data(FILE_PATH, PRODUCT, LOCATION)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize Custom Environment
    env = SingleEchelonEnv(
        demand_data=demand_data,
        forecast_data=forecast_data,
        lead_time=lead_time,
        initial_inventory=initial_inventory,
        max_order_qty=200 # Max order capped at 200 units (200,000 KG)
    )
    
    # Optional: check_env(env)
    
    # Train PPO Agent
    print(f"Starting training for {TRAIN_TIMESTEPS} steps...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    print("Training complete.")

    # Evaluate the trained model
    print("\n--- Evaluation on Historical Data --- Lead Time: {} weeks ---".format(lead_time))
    obs, _ = env.reset()
    total_cost = 0
    records = []

    # Combined label list so arrival week lookup works across the forecast horizon too
    all_week_labels = week_labels + future_week_labels

    def arrival_label(order_step_idx):
        idx = order_step_idx + lead_time
        if idx < len(all_week_labels):
            return str(all_week_labels[idx])
        return f"+{idx - len(all_week_labels) + 1}wk"

    header = (f"{'Week':<7} | {'Demand':<6} | {'Arrived':<7} | {'Order':<5} | "
              f"{'Due':<7} | {'Unmet':<5} | {'Inv':<5} | "
              f"{'Hold':<7} | {'OrdC':<7} | {'LostC':<8} | {'Cost':<9}")
    print(header)
    print("-" * len(header))

    terminated = False
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        step_idx = env.current_step
        obs, scaled_reward, terminated, truncated, info = env.step(action)
        total_cost -= info['reward']

        week_val = week_labels[step_idx] if step_idx < len(week_labels) else f"W{step_idx}"
        due_val   = arrival_label(step_idx)
        records.append({'week': week_val, 'due': due_val, **info})

        print(f"{week_val:<7} | {info['actual_demand']:<6} | {info['arriving_qty']:<7} | "
              f"{info['order_qty']:<5} | {due_val:<7} | {info['unmet_demand']:<5} | "
              f"{info['inventory']:<5} | {info['holding_cost_total']:<7.0f} | "
              f"{info['ordering_cost_total']:<7.0f} | {info['lost_sales_cost_total']:<8.0f} | "
              f"{-info['reward']:<9.0f}")

    print("-" * len(header))
    print(f"Total Episode Cost: €{total_cost:,.2f}")
    print("--------------------------------------")

    # Capture final state for forward projection
    final_inventory = env.inventory
    final_pipeline  = list(env.pipeline)

    # Project into future weeks using forecast as demand proxy
    if len(future_forecast) > 0:
        print(f"\n--- Forward Projection ({len(future_forecast)} future weeks) --- Lead Time: {lead_time} weeks ---")
    future_records = run_future_projection(
        model, final_inventory, final_pipeline,
        future_forecast, future_week_labels,
        lead_time=lead_time,
        n_forecast_weeks=env.n_forecast_weeks,
        holding_cost=env.holding_cost,
        ordering_cost=env.ordering_cost,
        lost_sales_cost=env.lost_sales_cost,
        max_order_qty=env.max_order_qty,
    )
    if future_records:
        fut_header = (f"{'Week':<7} | {'Forecast':<8} | {'Arrived':<7} | {'Order':<5} | "
                      f"{'Due':<7} | {'Inv':<5} | {'Hold':<7} | {'OrdC':<7} | {'LostC':<8}")
        print(fut_header)
        print("-" * len(fut_header))
        for i, r in enumerate(future_records):
            # step index in the future block = n_hist + i
            due_val = arrival_label(len(records) + i)
            print(f"{r['week']:<7} | {r['actual_demand']:<8} | {r['arriving_qty']:<7} | "
                  f"{r['order_qty']:<5} | {due_val:<7} | {r['inventory']:<5} | "
                  f"{r['holding_cost_total']:<7.0f} | {r['ordering_cost_total']:<7.0f} | "
                  f"{r['lost_sales_cost_total']:<8.0f}")
        print("-" * len(fut_header))

    visualize_results(records, PRODUCT, LOCATION, future_records=future_records)


if __name__ == "__main__":
    main()
