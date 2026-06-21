# PPO Algorithm for Inventory Optimization

## Overview

This project applies **Proximal Policy Optimization (PPO)** to a single-echelon inventory management problem. The agent learns a weekly ordering policy that minimizes total operational costs — holding, ordering, and lost sales — by interacting with a custom simulation environment built on [Gymnasium](https://gymnasium.farama.org/) and trained via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## 1. The Inventory Problem

The agent controls a single warehouse stocking a single SKU. Each week it must decide **how many units to order** before knowing the period's demand. Orders arrive after a fixed lead time, and unmet demand is lost (no backlogging).

The three cost drivers the agent must trade off:

| Cost | Default rate | Drives agent to… |
|---|---|---|
| Holding cost | €13 / unit in stock | Keep inventory lean |
| Ordering cost | €60 / unit ordered | Avoid unnecessary orders |
| Lost-sales cost | €2,500 / unit unmet | Avoid stockouts |

The asymmetry between costs (lost sales >> ordering >> holding) means the agent should err on the side of ordering rather than running out of stock.

---

## 2. Gymnasium Environment (`SingleEchelonEnv`)

### 2.1 Observation Space

Each week the agent receives a state vector of dimension `1 + lead_time + n_forecast_weeks`:

```
obs = [ current_inventory,
        pipeline[0], pipeline[1], ..., pipeline[lead_time-1],
        forecast[t], forecast[t+1], ..., forecast[t+n-1] ]
```

- **Current inventory** — units on hand after arrivals.
- **Pipeline inventory** — units in transit, one slot per week until arrival. `pipeline[0]` arrives next week, `pipeline[lead_time-1]` is the most recently placed order.
- **Demand forecast** — the next `n_forecast_weeks` (default: 4) forecast values from the data.

All components are divided by `max_order_qty` before being returned, keeping observations in a `[0, ~1]` range regardless of the product's scale. This prevents near-zero gradients for low-demand SKUs.

### 2.2 Action Space

The action space is continuous: a single scalar in `[0, 1]`. Inside `step()` this is scaled and rounded to an integer order quantity:

```python
order_qty = round(action[0] * max_order_qty)
```

Using a normalized continuous action space lets PPO operate on a fixed `[0, 1]` range while still producing discrete integer orders, and the `max_order_qty` cap is auto-corrected at runtime to `3× peak demand` if the user-specified cap is more than 5× that heuristic (preventing needle-in-a-haystack action targets).

### 2.3 Step Dynamics

Each call to `env.step(action)` simulates one week in the following order:

1. **Receive arrivals** — the oldest pipeline slot arrives and is added to inventory.
2. **Observe demand** — the actual demand for this week is revealed.
3. **Satisfy demand** — inventory is decremented. Any shortfall is recorded as unmet demand (lost, not backordered).
4. **Place order** — the agent's action is decoded and appended to the end of the pipeline.
5. **Compute reward** — `reward = -(holding_cost × inventory + ordering_cost × order_qty + lost_sales_cost × unmet_demand)`.
6. **Return scaled reward** — the raw reward is divided by 1,000 before being returned to the PPO trainer, keeping gradient magnitudes stable during learning.

### 2.4 Episode Structure

One episode spans all historical weeks in the data. `reset()` reinstates the initial inventory from the spreadsheet and clears the pipeline. The environment terminates when the last week with demand data is consumed (`current_step >= max_steps`).

---

## 3. PPO Algorithm

PPO is an on-policy, actor-critic algorithm that constrains how much the policy changes between gradient updates. This prevents large destabilizing updates that can occur with vanilla policy gradient methods.

### 3.1 Key Idea

PPO optimizes the **clipped surrogate objective**:

```
L_CLIP(θ) = E[ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]
```

Where:
- `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio between the new and old policy.
- `A_t` is the **Generalized Advantage Estimate** (GAE) — a low-variance estimate of how much better the action taken was compared to the baseline.
- `ε` (clip range, default 0.2 in SB3) limits how far the ratio can deviate from 1, preventing overly aggressive policy updates.

### 3.2 Actor-Critic Architecture (MlpPolicy)

The `MlpPolicy` in Stable-Baselines3 uses two separate multi-layer perceptrons sharing no weights:

- **Actor (policy network)** — maps observation → action distribution parameters. For a continuous action space this outputs a mean and log-std for a Gaussian distribution. The final action is sampled from this distribution during training; at evaluation time `deterministic=True` takes the mean directly.
- **Critic (value network)** — maps observation → scalar value estimate `V(s)`, used to compute advantages.

### 3.3 Training Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `learning_rate` | `1e-3` | Step size for the Adam optimizer |
| `gamma` | `0.99` | Discount factor — downweights future costs |
| `n_steps` | `2048` (auto-capped) | Rollout buffer size per update cycle |
| `batch_size` | `64` | Mini-batch size for gradient updates |
| `timesteps` | `10,000` | Total environment steps to train for |

**Auto-correction of `n_steps`**: if `n_steps` would be so large relative to `timesteps` that fewer than 10 PPO updates would occur, it is reduced automatically. It is also rounded down to a multiple of `episode_len` so rollout boundaries align with episode endings, avoiding value-estimate artifacts at truncated transitions.

### 3.4 Training Loop

```
for each rollout:
    collect n_steps transitions using current policy π_θ
    compute returns and GAE advantages
    for each mini-batch (repeated over several epochs):
        compute clipped surrogate loss
        compute value function loss
        compute entropy bonus
        backpropagate and update θ
```

The entropy bonus encourages exploration, which is especially valuable early in training when the agent has not yet discovered reliable ordering patterns.

---

## 4. Training Pipeline (`run_training_pipeline`)

The full pipeline executed by `run_training_pipeline()`:

1. **Load data** — demand, forecast, lead time, and initial inventory are read from the Excel file via `load_data()`.
2. **Auto-correct `max_order_qty`** — capped to `3× peak demand` if the configured value is too large.
3. **Build environment** — a `SingleEchelonEnv` is instantiated with the loaded data and cost parameters.
4. **Auto-correct `n_steps`** — reduced if needed to guarantee ≥10 PPO update cycles.
5. **Train** — `model.learn(total_timesteps=...)` runs the PPO training loop. A `ProgressCallback` optionally reports progress to the UI.
6. **Evaluate** — `evaluate_model()` runs the trained policy deterministically over the full historical period and records every weekly decision.
7. **Forward projection** — `run_future_projection()` continues the trained policy into future weeks using forecast values as the demand proxy.
8. **Save artifacts** — model weights, results Excel, visualization PNG, and JSON records are written to a timestamped run directory under `runs/`.

---

## 5. Evaluation

After training, `evaluate_model()` runs the policy with `deterministic=True` (greedy action — no exploration noise). It returns:

- **Records** — one dict per week containing demand, arrivals, order quantity, unmet demand, inventory, and all cost components.
- **KPIs** — aggregated from records via `compute_kpis()`:

| KPI | Formula |
|---|---|
| Total Cost (€) | `Σ (-reward)` over historical weeks |
| Service Level (%) | `(1 − Σ unmet / Σ demand) × 100` |
| Total Ordered (units) | `Σ order_qty` |
| Avg Inventory (units) | `mean(end-of-week inventory)` |

---

## 6. Forward Projection

`run_future_projection()` extends the evaluation beyond the historical data using forecast values. The pipeline and inventory state at the end of the historical run are carried forward, so the projection seamlessly continues from where history ends. Future weeks appear with hatched bars in the visualization to distinguish them from observed data.

---

## 7. Baseline Comparison

The Streamlit UI can run one or more **base-stock (order-up-to) policies** alongside PPO. A base-stock policy with level `S` orders `max(0, S − inventory_position)` each week, where `inventory_position = on_hand + pipeline`. Comparing PPO against several `S` values shows whether and by how much the learned policy improves on the classical benchmark.

---

## 8. Design Decisions

| Decision | Rationale |
|---|---|
| Continuous `[0,1]` action space | Fixed range works across all SKUs without re-tuning action bounds |
| Observation normalized by `max_order_qty` | Keeps all inputs in `[0,~1]`, preventing gradient scale issues for low-demand SKUs |
| Reward divided by 1,000 | Keeps gradient magnitudes in a numerically stable range during backprop |
| Lost-sales (no backlog) | Matches the real-world constraint that unmet demand is gone, not deferred |
| Lead-time pipeline as part of state | Gives the agent full visibility into committed future receipts, enabling proactive ordering |
| Forecast window in state | Enables the agent to anticipate demand spikes and order ahead of lead time |
