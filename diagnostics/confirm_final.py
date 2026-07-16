#!/usr/bin/env python3
"""Finaler Bestätigungslauf: Warmup + 200k + n_steps=512, leicht hoeherer Puffer,
gespeichert nach runs/. Ziel: RL_cost < baseline UND RL_sl >= baseline_sl.
"""
import json
from datetime import datetime
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
warmup = int(round(avg_demand * (LEAD_TIME + 0.4)))  # kleiner Extra-Puffer fuer SL
print(f"avg_demand={avg_demand:.1f}, warmup_inv={warmup}\n", flush=True)

slug = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_rca-warmup-fix"
run_dir = Path(__file__).parent.parent / "runs" / slug

cfg = TrainingConfig(
    product=PRODUCT, location=LOCATION, lead_time=LEAD_TIME,
    initial_inventory=warmup, timesteps=200000, n_steps=512,
    learning_rate=1e-3, verbose=0,
)
rr = run_training_pipeline(cfg, run_dir=run_dir, verbose=False)

real = [x for x in rr.base_stock_results if not x.get('theoretical')]
b = min(real, key=lambda x: x['kpis']['Total Cost (€)'])
bvar = b['variant']; bcost = float(b['kpis']['Total Cost (€)']); bsl = float(b['kpis']['Service Level (%)'])
rl_cost, rl_sl = float(rr.total_cost), float(rr.service_level)
beats = rl_cost < bcost and rl_sl >= bsl

result = {
    "run_dir": str(run_dir), "warmup_inv": warmup,
    "rl_cost": rl_cost, "rl_sl": rl_sl,
    "baseline_variant": bvar, "baseline_cost": bcost, "baseline_sl": bsl,
    "cost_win_eur": bcost - rl_cost, "sl_diff": rl_sl - bsl,
    "beats_baseline": beats,
}
print(f"RL:       cost={rl_cost:.0f}  SL={rl_sl:.4f}%")
print(f"baseline: cost={bcost:.0f}  SL={bsl:.4f}%  ({bvar})")
print(f"cost_win={bcost-rl_cost:.0f}  sl_diff={rl_sl-bsl:+.4f}")
print(f"BEATS BASELINE (both metrics): {beats}")

out = Path(__file__).parent / "confirm_final_result.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved -> {out}\nRun artifacts -> {run_dir}")
