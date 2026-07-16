#!/usr/bin/env python3
"""Zweite Runde: Warmup ist gesetzt, wie schliessen wir den ~19% Rest-Gap?
Teste warmup + {mehr Timesteps, mehr Updates}. Trusted Pipeline-Eval, RL/Baseline gematcht.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from inventory_ppo import TrainingConfig, run_training_pipeline, load_csv_scenarios, DEFAULT_CSV_PATH

PRODUCT = "Ice Cream Strawberry Flavor"
LOCATION = "Logistics Hub Lissabon"
LEAD_TIME = 2

scen, _ = load_csv_scenarios(DEFAULT_CSV_PATH, PRODUCT, LOCATION)
avg_demand = float(np.mean([np.mean(s['demand']) for s in scen]))
WARMUP = int(round(avg_demand * LEAD_TIME))
print(f"warmup_inv={WARMUP}\n")


def best_baseline(rr):
    b = min(rr.base_stock_results, key=lambda x: x['kpis']['Total Cost (€)'])
    return b['variant'], float(b['kpis']['Total Cost (€)']), float(b['kpis']['Service Level (%)'])


def run(label, timesteps, n_steps, lr=1e-3):
    cfg = TrainingConfig(
        product=PRODUCT, location=LOCATION, lead_time=LEAD_TIME,
        initial_inventory=WARMUP, timesteps=timesteps, n_steps=n_steps,
        learning_rate=lr, verbose=0,
    )
    rr = run_training_pipeline(cfg, run_dir=None, verbose=False)
    bvar, bcost, bsl = best_baseline(rr)
    rl_cost, rl_sl = float(rr.total_cost), float(rr.service_level)
    beats = rl_cost < bcost and rl_sl >= bsl
    gap = (rl_cost - bcost) / bcost * 100
    print(f"[{label}] ts={timesteps} n_steps={n_steps}")
    print(f"    RL:       cost={rl_cost:>10.0f}  SL={rl_sl:5.1f}%")
    print(f"    baseline: cost={bcost:>10.0f}  SL={bsl:5.1f}%  ({bvar})")
    print(f"    gap={gap:+.1f}%  -> {'RL BEATS' if beats else 'baseline better'}\n", flush=True)
    return {"label": label, "timesteps": timesteps, "n_steps": n_steps,
            "rl_cost": rl_cost, "rl_sl": rl_sl, "baseline_cost": bcost,
            "baseline_sl": bsl, "gap_pct": gap, "beats_baseline": beats}


res = {"warmup_inv": WARMUP, "runs": []}
res["runs"].append(run("warmup_100k",        100000, 2048))
res["runs"].append(run("warmup_100k_upd",    100000, 512))
res["runs"].append(run("warmup_200k_upd",    200000, 512))

out = Path(__file__).parent / "validate_real2_result.json"
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(f"Saved -> {out}")
