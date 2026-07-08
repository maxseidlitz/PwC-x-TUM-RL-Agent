# tune.py — Hyperparameter Search

Trains the PPO inventory agent across multiple parameter combinations and product-location pairs, then writes results to Excel.

## Quick start

```bash
python tune.py                   # run with PARAM_COMBOS (or PARAM_GRID if combos empty)
python tune.py --out my_run.xlsx # custom output file
python tune.py --grid            # force full Cartesian grid even if PARAM_COMBOS is set
```

## Two modes

### Mode 1 — Explicit combinations (recommended)

Fill `PARAM_COMBOS` with the exact runs you want:

```python
PARAM_COMBOS: list[dict] = [
    {"timesteps": 20_000,  "learning_rate": 1e-3, "n_steps": 128,  "gamma": 0.99, "batch_size": 64, "n_forecast_weeks": 4},
    {"timesteps": 100_000, "learning_rate": 1e-4, "n_steps": 512,  "gamma": 0.99, "batch_size": 64, "n_forecast_weeks": 4},
]
```

Each dict is one run. No Cartesian explosion — you get exactly as many param sets as entries in the list.

### Mode 2 — Full Cartesian grid (fallback)

Leave `PARAM_COMBOS = []` and fill `PARAM_GRID`:

```python
PARAM_GRID: dict[str, list] = {
    "timesteps":     [20_000, 100_000],
    "learning_rate": [1e-4, 1e-3],
    "n_steps":       [128, 512, 2048],
    ...
}
```

Every combination is expanded (e.g. 2 × 2 × 3 = 12 combos). Force this mode from the CLI with `--grid`.

## Product-location pairs

`PRODUCT_LOCATIONS` controls which product/location combinations are evaluated. Each param combo is run once per pair, so total runs = `len(PRODUCT_LOCATIONS) × len(combos)`.

## Fixed parameters

`FIXED` holds config fields that don't vary across the sweep (file path, cost parameters, etc.). These are merged with each param combo before training.

## Output

Results are written to `tuning_results.xlsx` (or `--out <path>`) with two sheets:

| Sheet | Contents |
|-------|----------|
| **Summary** | One row per run — all params + KPIs (total cost, service level, avg inventory). Sorted by total cost ascending. Failed runs appear at the bottom. |
| **Per-Week** | Flattened weekly step records for every successful run, keyed by `run_index`. |

The best run is printed to the console at the end.

## Key parameters

| Parameter | Effect |
|-----------|--------|
| `timesteps` | PPO training steps — more = longer training, potentially better policy |
| `learning_rate` | Adam LR for the policy network |
| `n_steps` | Rollout buffer size before each PPO update |
| `gamma` | Discount factor for future rewards |
| `batch_size` | Mini-batch size for policy gradient updates |
| `n_forecast_weeks` | How many weeks of demand forecast the agent observes |
