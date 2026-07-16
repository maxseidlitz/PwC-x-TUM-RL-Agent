#!/usr/bin/env python3
"""Entscheidende Validierung auf dem ECHTEN Problem (Ice Cream Strawberry / Lissabon).

Vergleicht control (initial_inventory=0, aktuelles Verhalten) gegen warmup
(initial_inventory = round(avg_demand * lead_time)). In BEIDEN Faellen bewertet die
Pipeline RL und Base-Stock unter identischen Startbedingungen (fair).
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
TIMESTEPS = 30000

# avg demand fuer Warmup-Level bestimmen
scen, weeks = load_csv_scenarios(DEFAULT_CSV_PATH, PRODUCT, LOCATION)
demand_arrays = [s['demand'] for s in scen]
avg_demand = float(np.mean([np.mean(d) for d in demand_arrays]))
warmup_inv = int(round(avg_demand * LEAD_TIME))
print(f"avg_demand={avg_demand:.1f}, warmup_inv={warmup_inv}, n_scenarios={len(scen)}\n")


def best_baseline(rr):
    best = min(rr.base_stock_results, key=lambda b: b['kpis']['Total Cost (€)'])
    return best['variant'], float(best['kpis']['Total Cost (€)']), float(best['kpis']['Service Level (%)'])


def run(initial_inventory, label):
    cfg = TrainingConfig(
        product=PRODUCT, location=LOCATION, lead_time=LEAD_TIME,
        initial_inventory=initial_inventory, timesteps=TIMESTEPS, verbose=0,
    )
    rr = run_training_pipeline(cfg, run_dir=None, verbose=False)
    bvar, bcost, bsl = best_baseline(rr)
    rl_cost, rl_sl = float(rr.total_cost), float(rr.service_level)
    beats = rl_cost < bcost and rl_sl >= bsl
    print(f"[{label}] init_inv={initial_inventory}")
    print(f"    RL:            cost={rl_cost:>10.0f}  SL={rl_sl:5.1f}%")
    print(f"    best baseline: cost={bcost:>10.0f}  SL={bsl:5.1f}%  ({bvar})")
    print(f"    -> {'RL BEATS BASELINE' if beats else 'baseline better'}\n", flush=True)
    return {"init_inv": initial_inventory, "rl_cost": rl_cost, "rl_sl": rl_sl,
            "baseline_variant": bvar, "baseline_cost": bcost, "baseline_sl": bsl,
            "beats_baseline": beats}


results = {"avg_demand": avg_demand, "warmup_inv": warmup_inv,
           "timesteps": TIMESTEPS, "runs": {}}
results["runs"]["control"] = run(0, "CONTROL")
results["runs"]["warmup"] = run(warmup_inv, "WARMUP")

out = Path(__file__).parent / "validate_real_result.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved -> {out}")
