#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from inventory_ppo import (
    SingleEchelonEnv, PPO, run_base_stock_policy,
    compute_kpis
)

print("=== Trivialfall-Sanity-Check: konstante Nachfrage ===")

# Trivial problem: constant demand = 50 units/week for 52 weeks
demand = np.array([50] * 52, dtype=int)
lead_time = 2
initial_inventory = 0
holding_cost, ordering_cost, lost_sales_cost = 13, 60, 2500
max_order_qty = 150  # 3× peak demand = 3×50

# Create environment
env = SingleEchelonEnv(
    demand_data=demand, forecast_data=demand, lead_time=lead_time,
    initial_inventory=initial_inventory, holding_cost=holding_cost,
    ordering_cost=ordering_cost, lost_sales_cost=lost_sales_cost,
    max_order_qty=max_order_qty, n_forecast_weeks=4
)

print(f"Env: {len(demand)}-week episodes, lead_time={lead_time}, max_order_qty={max_order_qty}")
print(f"Training PPO for 50k timesteps...")

# Train PPO for ~50k timesteps
model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3,
            gamma=0.99, n_steps=2048, batch_size=64)
model.learn(total_timesteps=50000)

print("Training complete. Evaluating...")

# Evaluate on same 52 weeks
obs, _ = env.reset()
records_rl = []
terminated = False
while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, _, info = env.step(action)
    records_rl.append(info)

kpis_rl = compute_kpis(records_rl)

# Baseline: Base-stock with S = 50*2 = 100 (avg_demand * lead_time)
S_baseline = int(50 * lead_time)
records_baseline = run_base_stock_policy(
    S_baseline, demand, list(range(52)), lead_time, initial_inventory,
    holding_cost=holding_cost, ordering_cost=ordering_cost,
    lost_sales_cost=lost_sales_cost
)
kpis_baseline = compute_kpis(records_baseline)

print(f"\n--- Results ---")
print(f"RL:       Cost={kpis_rl['total_cost']:.0f}, SL={kpis_rl['service_level']:.1f}%")
print(f"Baseline: Cost={kpis_baseline['total_cost']:.0f}, SL={kpis_baseline['service_level']:.1f}%")

# Save result
result = {
    "rl_total_cost": float(kpis_rl['total_cost']),
    "rl_service_level": float(kpis_rl['service_level']),
    "baseline_s_value": S_baseline,
    "baseline_total_cost": float(kpis_baseline['total_cost']),
    "baseline_service_level": float(kpis_baseline['service_level']),
    "verdict": "RL better" if kpis_rl['total_cost'] < kpis_baseline['total_cost']
              else "Baseline better (setup issue?)"
}

output_path = Path(__file__).parent / "trivial_result.json"
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Result saved to {output_path}")
print(f"Verdict: {result['verdict']}")
