# Inventory Optimization with Reinforcement Learning (PPO)

This project implements a **Single-Echelon Inventory Optimization** system using Proximal Policy Optimization (PPO). It trains a Reinforcement Learning agent to make optimal ordering decisions based on historical demand, forecasts, and lead times, with the goal of minimizing total costs (Holding, Ordering, and Lost Sales).

## Features

- **Custom Gymnasium Environment**: Tailored for inventory management with support for lead times and pipeline inventory.
- **PPO Agent**: Utilizes `stable-baselines3` for robust reinforcement learning.
- **CLI Interface**: Easily configure experiments via command-line arguments.
- **Streamlit UI**: Web interface for parameter configuration, training with progress/ETA, and an interactive Plotly dashboard.
- **Data-Driven**: Loads demand, inventory, and forecast data directly from Excel files.
- **Detailed Analytics**: Provides step-by-step (weekly) breakdown of costs and actions during evaluation.
- **Forward Projection**: Extends the trained policy beyond the historical period using forecast data as a demand proxy.
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

Configure product, location, training parameters, and cost model in the sidebar, then click **Start Training**. After training completes, explore KPIs and an interactive dashboard with togglable data series. Each run is saved under `runs/<timestamp>_<product-slug>/`.

Use the **Compare Runs** tab to overlay up to 5 saved runs: filter by product, location, and timesteps, review a KPI comparison table, and inspect overlaid charts (inventory, orders, weekly total cost, cumulative cost) to see how the policy improves across training runs.

### Command line

Run the main script using the command line. You can customize the run using various flags:

```bash
python inventory_ppo.py --file-path "Sample Data RL4IM UPDATED.xlsx" --product "Ice Cream Strawberry Flavor" --location "Logistics Hub Lissabon" --timesteps 10000
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--file-path` | Path to the Excel data file | `Sample Data RL4IM UPDATED.xlsx` |
| `--product` | Name of the product to optimize | `Ice Cream Strawberry Flavor` |
| `--location` | Warehouse location | `Logistics Hub Lissabon` |
| `--timesteps` | Number of training iterations | `10000` |

## Data Structure

The Excel file should contain the following sheets:

- **Demand**: Weekly actual demand per product/location. Columns: `Product`, `Location`, `ww.yyyy`, ...
- **Current Inventory**: Initial inventory levels. Columns: `Product`, `Location`, `Current Inventory`.
- **Lead Time**: Lead time in weeks. Columns: `Product`, `Location`, `Lead Time in weeks`.
- **Forecast**: Weekly forecast data. Columns: `Product`, `Location`, `ww.yyyy`, ...

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
| Total Cost (€) | Sum of all costs over the historical period |
| Service Level (%) | `(1 − unmet demand / total demand) × 100` |
| Total Ordered (units) | Cumulative units ordered |
| Avg Inventory (units) | Mean end-of-week inventory |
| Historical / Projected Weeks | Number of weeks in each period |

**Historical & Future Projection columns**

`Week`, `Due Week`, `Demand` (or `Forecast Demand`), `Arrived Qty`, `Order Qty`, `Unmet Demand`, `Inventory (End)`, `Holding Cost (€)`, `Ordering Cost (€)`, `Lost Sales Cost (€)`, `Total Cost (€)`

## Cost Model

The agent's reward is the negative of the total costs:
- **Holding Cost**: 13 € per unit in stock.
- **Ordering Cost**: 60 € per unit ordered.
- **Lost Sales Cost**: 2,500 € per unmet unit (lost-sales model, no backlogging).

## License

[MIT](LICENSE)
