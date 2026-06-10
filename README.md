# Inventory Optimization with Reinforcement Learning (PPO)

This project implements a **Single-Echelon Inventory Optimization** system using Proximal Policy Optimization (PPO). It trains a Reinforcement Learning agent to make optimal ordering decisions based on historical demand, forecasts, and lead times, with the goal of minimizing total costs (Holding, Ordering, and Lost Sales).

## Features

- **Custom Gymnasium Environment**: Tailored for inventory management with support for lead times and pipeline inventory.
- **PPO Agent**: Utilizes `stable-baselines3` for robust reinforcement learning.
- **CLI Interface**: Easily configure experiments via command-line arguments.
- **Data-Driven**: Loads demand, inventory, and forecast data directly from Excel files.
- **Detailed Analytics**: Provides step-by-step (weekly) breakdown of costs and actions during evaluation.

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

## Cost Model

The agent's reward is the negative of the total costs:
- **Holding Cost**: 13 € per unit in stock.
- **Ordering Cost**: 60 € per unit ordered.
- **Lost Sales Cost**: 2500 € per unmet unit (No backlogging).

## License

[MIT](LICENSE)
