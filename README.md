# Inventory Optimization with Reinforcement Learning (PPO)

This project implements a **Single-Echelon Inventory Optimization** system using Proximal Policy Optimization (PPO). It trains a Reinforcement Learning agent to make optimal ordering decisions based on generated demand scenarios and lead times, with the goal of minimizing total costs (Holding, Ordering, and Lost Sales).

## Features

- **Custom Gymnasium Environment**: Tailored for inventory management with support for lead times and pipeline inventory. Supports multi-scenario training: the agent sees a randomly drawn demand scenario on each episode reset, enabling generalization across demand patterns.
- **PPO Agent**: Utilizes `stable-baselines3` for robust reinforcement learning.
- **Scenario-based Training**: Trains on generated demand scenarios from a CSV file instead of historical data. Each row in the CSV is one scenario; the agent is trained across all scenarios simultaneously.
- **CLI Interface**: Easily configure experiments via command-line arguments.
- **Streamlit UI**: Web interface for parameter configuration, training with progress/ETA, and an interactive Plotly dashboard.
- **Detailed Analytics**: Provides step-by-step (weekly) breakdown of costs and actions during evaluation.
- **Forward Projection**: Projects the trained policy through every scenario individually and averages results week-by-week.
- **Base-Stock Baselines**: Computes conservative/middle/aggressive base-stock policies for comparison.
- **Excel Export**: Saves all results to `results.xlsx` with three sheets — Summary, Historical, and Future Projection.
- **Visualization**: Generates a multi-panel dashboard (`results.png`) covering inventory levels, order quantities, cost breakdown, and cumulative cost.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PwC-x-TUM-RL-Agent
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit UI (recommended)

Launch the web interface from the project root:

```bash
streamlit run ui/app.py
```

The UI automatically loads the default scenario CSV (`demand_scenarios/generated_demand_scenarios.csv`). You can optionally upload a different CSV file via the sidebar file uploader.

Configure product, location, inventory parameters (lead time, initial inventory), training parameters, and cost model in the sidebar, then click **Start Training**. After training completes, explore KPIs and an interactive dashboard with togglable data series. Each run is saved under `runs/<timestamp>_<product-slug>/`.

Use the **Compare Runs** tab to overlay up to 5 saved runs: filter by product, location, and timesteps, review a KPI comparison table, and inspect overlaid charts (inventory, orders, weekly total cost, cumulative cost) to see how the policy improves across training runs.

### Command line (CSV mode)

```bash
python inventory_ppo.py \
  --csv-path "demand_scenarios/generated_demand_scenarios.csv" \
  --product "Ice Cream Strawberry Flavor" \
  --location "Logistics Hub Lissabon" \
  --lead-time 2 \
  --initial-inventory 0 \
  --timesteps 50000
```

### Command line (legacy Excel mode)

```bash
python inventory_ppo.py \
  --file-path "Sample Data RL4IM UPDATED_with_scenarios_v3.xlsx" \
  --product "Ice Cream Strawberry Flavor" \
  --location "Logistics Hub Lissabon" \
  --timesteps 10000
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--csv-path` | Path to scenario CSV file (CSV mode) | `demand_scenarios/generated_demand_scenarios.csv` |
| `--file-path` | Path to Excel data file (legacy mode, used when `--csv-path` is empty) | `Sample Data RL4IM UPDATED_with_scenarios_v3.xlsx` |
| `--product` | Name of the product to optimize | `Ice Cream Strawberry Flavor` |
| `--location` | Warehouse location | `Logistics Hub Lissabon` |
| `--lead-time` | Lead time in weeks (CSV mode) | `2` |
| `--initial-inventory` | Starting inventory level in units (CSV mode) | `0` |
| `--timesteps` | Number of training iterations | `10000` |

## Data Structure

### Scenario CSV (primary)

The file at `demand_scenarios/generated_demand_scenarios.csv` drives training. It must have exactly three metadata columns followed by week columns:

| Col 1 | Col 2 | Col 3 | `18.2026` | `19.2026` | … |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Product | Location | Scenario_Number | 120 | 180 | … |

- The header names of the first three columns are flexible (e.g. `Product` or `Product_Name` both work).
- Week columns use the format `KW.YYYY` (e.g. `18.2026`).
- Each row is one independent demand scenario. All scenarios for the same product/location combination are used together during training.
- Lead time and initial inventory are **not** in the CSV — they are supplied via the UI sidebar or the `--lead-time` / `--initial-inventory` CLI flags.

### Legacy Excel (optional)

For backwards compatibility the original Excel format is still supported. It requires the sheets: **Demand**, **Current Inventory**, **Lead Time**, and **Forecast**. This mode is used automatically when no CSV path is provided.

## Outputs

After each run, artifacts are written to a dedicated folder:

`runs/<YYYY-MM-DD_HHMMSS>_<product-slug>/`

| File | Contents |
| :--- | :--- |
| `config.json` | All run parameters, timing, and summary KPIs |
| `model.zip` | Trained PPO model (Stable-Baselines3 format) |
| `results.png` | Static multi-panel dashboard (matplotlib) |
| `results.xlsx` | **Summary**, **Historical**, and **Future Projection** sheets |
| `records.json` | Serialized weekly records for UI reload and run comparison |

### Excel Sheet Details

**Summary**

| Metric | Description |
| :--- | :--- |
| Total Cost (€) | Sum of all costs over the projection period |
| Service Level (%) | `(1 − unmet demand / total demand) × 100` |
| Total Ordered (units) | Cumulative units ordered |
| Avg Inventory (units) | Mean end-of-week inventory |
| Forecast Weeks | Number of weeks in the projection |

**Historical & Future Projection columns**

`Week`, `Due Week`, `Demand` (or `Forecast Demand`), `Arrived Qty`, `Order Qty`, `Unmet Demand`, `Inventory (End)`, `Holding Cost (€)`, `Ordering Cost (€)`, `Lost Sales Cost (€)`, `Total Cost (€)`

## Cost Model

The agent's reward is the negative of the total costs:
- **Holding Cost**: 13 € per unit in stock.
- **Ordering Cost**: 60 € per unit ordered.
- **Lost Sales Cost**: 2,500 € per unmet unit (lost-sales model, no backlogging).

All cost parameters are configurable via the UI sidebar or `TrainingConfig`.

## License

[MIT](LICENSE)
