#!/usr/bin/env python3
"""Ablation auf dem Trivialfall (konstante Nachfrage): welcher Hebel loest den Setup-Bug?

Variants:
  A control       - wie trivial_test (kein Fix)
  B vecnorm       - VecNormalize(norm_reward=True) um den Env
  C warmup        - initial_inventory=100 (kein Cold-Start-Stockout)
  D vecnorm+warm  - beides kombiniert
  E vecnorm+more  - VecNormalize + 100k Timesteps
Alle gegen Base-Stock S=100 (avg_demand*lead_time) auf demselben 52-Wochen-Fall.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from inventory_ppo import SingleEchelonEnv, PPO, run_base_stock_policy, compute_kpis
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEMAND = np.array([50] * 52, dtype=int)
LEAD_TIME = 2
HOLD, ORDER, LOST = 13, 60, 2500
MAXQ = 150
NFC = 4


def make_env(initial_inventory=0):
    return SingleEchelonEnv(
        demand_data=DEMAND, forecast_data=DEMAND, lead_time=LEAD_TIME,
        initial_inventory=initial_inventory, holding_cost=HOLD,
        ordering_cost=ORDER, lost_sales_cost=LOST,
        max_order_qty=MAXQ, n_forecast_weeks=NFC,
    )


def eval_plain(model, initial_inventory=0):
    env = make_env(initial_inventory)
    obs, _ = env.reset()
    recs, done = [], False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(a)
        recs.append(info)
    return compute_kpis(recs)


def eval_vecnorm(model, vecnorm, initial_inventory=0):
    # Nutze die gelernten obs-Normalisierungs-Statistiken, reward-norm aus fuer Eval
    env = make_env(initial_inventory)
    obs, _ = env.reset()
    recs, done = [], False
    while not done:
        norm_obs = vecnorm.normalize_obs(obs)
        a, _ = model.predict(norm_obs, deterministic=True)
        obs, _, done, _, info = env.step(a)
        recs.append(info)
    return compute_kpis(recs)


def train_plain(timesteps=50000, initial_inventory=0):
    env = make_env(initial_inventory)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3,
                gamma=0.99, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=timesteps)
    return eval_plain(model, initial_inventory)


def train_vecnorm(timesteps=50000, initial_inventory=0, norm_obs=True):
    venv = DummyVecEnv([lambda: make_env(initial_inventory)])
    vecnorm = VecNormalize(venv, norm_obs=norm_obs, norm_reward=True,
                           clip_reward=10.0, gamma=0.99)
    model = PPO("MlpPolicy", vecnorm, verbose=0, learning_rate=1e-3,
                gamma=0.99, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=timesteps)
    vecnorm.training = False
    if norm_obs:
        return eval_vecnorm(model, vecnorm, initial_inventory)
    return eval_plain(model, initial_inventory)


# Baseline (fix) auf demselben Fall
bs = compute_kpis(run_base_stock_policy(
    int(50 * LEAD_TIME), DEMAND, list(range(52)), LEAD_TIME, 0,
    holding_cost=HOLD, ordering_cost=ORDER, lost_sales_cost=LOST))
BASE = {"cost": float(bs['total_cost']), "sl": float(bs['service_level'])}

results = {"baseline_S100": BASE, "variants": {}}

variants = {
    "A_control":       lambda: train_plain(50000, 0),
    "B_vecnorm":       lambda: train_vecnorm(50000, 0, norm_obs=True),
    "C_warmup":        lambda: train_plain(50000, 100),
    "D_vecnorm_warm":  lambda: train_vecnorm(50000, 100, norm_obs=True),
    "E_vecnorm_more":  lambda: train_vecnorm(100000, 0, norm_obs=True),
}

print(f"Baseline S=100: cost={BASE['cost']:.0f}, SL={BASE['sl']:.1f}%\n")
for name, fn in variants.items():
    print(f"Training {name} ...", flush=True)
    k = fn()
    cost, sl = float(k['total_cost']), float(k['service_level'])
    beats = cost < BASE['cost'] and sl >= BASE['sl']
    results["variants"][name] = {"cost": cost, "sl": sl, "beats_baseline": beats}
    print(f"  {name}: cost={cost:.0f}, SL={sl:.1f}%  {'<<< BEATS BASELINE' if beats else ''}", flush=True)

out = Path(__file__).parent / "ablation_trivial_result.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
