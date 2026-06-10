import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
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
                 max_order_qty=500, n_forecast_weeks=8):
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
    Returns: demand (array), forecast (array), lead_time (int), initial_inventory (int), week_labels (list)
    """
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    
    # Helper to find sheet name regardless of trailing spaces
    def get_sheet_name(base_name):
        for name in sheet_names:
            if name.strip() == base_name:
                return name
        return base_name

    df_demand = pd.read_excel(file_path, sheet_name=get_sheet_name('Demand'))
    df_inventory = pd.read_excel(file_path, sheet_name=get_sheet_name('Current Inventory'))
    df_lead_time = pd.read_excel(file_path, sheet_name=get_sheet_name('Lead Time'))
    df_forecast = pd.read_excel(file_path, sheet_name=get_sheet_name('Forecast'))

    # Strip whitespace from columns for consistent access
    for df in [df_demand, df_inventory, df_lead_time, df_forecast]:
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

    week_labels = df_demand.columns[2:].tolist()

    # Filter for the specific product/location
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

    return demand, forecast, lead_time, inventory, week_labels

import argparse

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
        demand_data, forecast_data, lead_time, initial_inventory, week_labels = load_data(FILE_PATH, PRODUCT, LOCATION)
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
    print("\n--- Evaluation on Historical Data ---")
    obs, _ = env.reset()
    total_reward = 0
    
    header = (f"{'Week':<7} | {'Demand':<6} | {'Order':<5} | {'Unmet':<5} | {'Inv':<4} | "
              f"{'Hold':<7} | {'OrdC':<7} | {'LostC':<8} | {'Reward':<9}")
    print(header)
    print("-" * len(header))
    
    terminated = False
    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        # Store current step to index week_labels
        step_idx = env.current_step
        obs, scaled_reward, terminated, truncated, info = env.step(action)
        total_reward += info['reward']
        
        week_val = week_labels[step_idx] if step_idx < len(week_labels) else f"W{step_idx}"
        
        print(f"{week_val:<7} | {info['actual_demand']:<6} | {info['order_qty']:<5} | "
              f"{info['unmet_demand']:<5} | {info['inventory']:<4} | "
              f"{info['holding_cost_total']:<7.0f} | {info['ordering_cost_total']:<7.0f} | "
              f"{info['lost_sales_cost_total']:<8.0f} | {info['reward']:<9.0f}")

    print("-" * len(header))
    print(f"Total Episode Reward: {total_reward:.2f}")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
