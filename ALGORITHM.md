# Algorithm: PPO-Based Inventory Optimization

## 1. Problem Statement

The system solves a **single-echelon inventory control problem**: given weekly demand,
a fixed replenishment lead time, and a cost structure, find an ordering policy that
minimises total operational cost over a planning horizon while avoiding lost sales.

---

## 2. Markov Decision Process Formulation

The environment is modelled as a discrete-time, finite-horizon MDP:

```
M = (S, A, P, R, γ, T)
```

| Symbol | Meaning |
|--------|---------|
| `S` | State space |
| `A` | Action space |
| `P` | Transition kernel |
| `R` | Reward function |
| `γ` | Discount factor (default 0.99) |
| `T` | Episode length = number of demand periods |

---

## 3. State Space

At decision epoch `t`, the agent observes:

```
s_t = ( I_t,  p_t^(1), …, p_t^(L),  f_t^(1), …, f_t^(N) )  /  Q_max
```

where:

- `I_t` — on-hand inventory at the start of period `t` (after the arriving shipment has
  been added but before demand is realised)
- `p_t^(ℓ)` — quantity in the pipeline that will arrive in `ℓ` periods
  (the *pipeline vector*, length = lead time `L`)
- `f_t^(n)` — point forecast for demand `n` periods ahead (lookahead of `N = 4` weeks)
- `Q_max` — maximum order quantity (auto-calibrated to `3 × peak demand`)

The full observation vector has dimension `d = 1 + L + N`.

The division by `Q_max` normalises all components to roughly `[0, 1]`, preventing
scale-induced gradient problems when products have very different demand magnitudes.

---

## 4. Action Space

The action is a **continuous scalar** representing the fraction of the maximum order
quantity to place:

```
a_t ∈ [0, 1]    →    q_t = round( a_t · Q_max )
```

The agent outputs a real number; the environment discretises it by rounding to the
nearest integer unit. This design allows the policy network to exploit smooth
gradient information while the environment enforces integer orders.

---

## 5. Environment Dynamics (Transition Kernel)

Each period `t` executes the following sequence:

```
1.  Receive arriving shipment:
        I_t  ←  I_{t-1}  +  p_{t-1}^(1)

2.  Observe actual demand  d_t

3.  Satisfy demand (lost-sales model — no backlogging):
        if I_t ≥ d_t:   I_t ←  I_t - d_t,   u_t = 0
        else:            u_t =  d_t - I_t,   I_t = 0

4.  Agent places order  q_t = round( a_t · Q_max )

5.  Advance pipeline:
        pipeline ← [ p_{t-1}^(2), …, p_{t-1}^(L), q_t ]

6.  t ← t + 1
```

The **lost-sales** assumption means unfulfilled demand is permanently lost
(no backorders). This is realistic for perishable or time-sensitive goods.

---

## 6. Reward Function

The reward at each step is the **negative total cost** scaled by 1,000:

```
R_t = -( c_h · I_t  +  c_o · q_t  +  c_l · u_t )  /  1000
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `c_h`     | 13 €    | Holding cost per unit per week |
| `c_o`     | 60 €    | Ordering cost per unit ordered |
| `c_l`     | 2,500 € | Lost-sales penalty per unit unmet |

The `/1000` scaling keeps rewards in a numerically comfortable range for PPO's
value function. The agent's objective is to maximise cumulative discounted reward:

```
J(π) = E_π [ Σ_{t=0}^{T-1} γ^t · R_t ]
```

which is equivalent to minimising total discounted operational cost.

---

## 7. Policy Optimisation: Proximal Policy Optimisation (PPO)

PPO is an actor-critic on-policy algorithm that learns a stochastic policy
`π_θ(a|s)` and a value function `V_φ(s)`.

### 7.1 Policy Network (Actor)

```
π_θ : S → N(μ_θ(s), σ_θ(s))   (Gaussian policy on [0,1])
```

Implemented as a multi-layer perceptron with two hidden layers of 64 units each
(Stable-Baselines3 `MlpPolicy` default).

### 7.2 Value Function (Critic)

```
V_φ : S → R
```

A separate MLP estimating the expected return from state `s`.

### 7.3 Advantage Estimation (GAE)

Generalised Advantage Estimation with parameter `λ`:

```
δ_t   = R_t + γ · V_φ(s_{t+1}) - V_φ(s_t)

Â_t  = Σ_{k=0}^{T-t-1} (γλ)^k · δ_{t+k}
```

### 7.4 Clipped Surrogate Objective

```
r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)

L^{CLIP}(θ) = E_t [ min( r_t(θ) · Â_t,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]
```

The clip parameter `ε = 0.2` (SB3 default) prevents excessively large policy
updates that could destabilise training on a small, finite episode.

### 7.5 Total Loss

```
L(θ, φ) = -L^{CLIP}(θ)  +  c_v · L^{VF}(φ)  -  c_e · H[π_θ]
```

| Term | Role |
|------|------|
| `L^{CLIP}` | Policy gradient (actor) |
| `L^{VF}`   | Value function MSE (critic) |
| `H[π_θ]`  | Entropy bonus for exploration |

### 7.6 Hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| `learning_rate` | 1e-3 | Adam step size |
| `gamma` (γ) | 0.99 | Discount factor |
| `n_steps` | 2048 (auto-capped) | Rollout buffer length |
| `batch_size` | 64 | Mini-batch size per gradient step |
| `timesteps` | 10,000 | Total environment interactions |

**Auto-capping of `n_steps`:** to guarantee at least 10 PPO updates in any run,
the code enforces `n_steps ≤ total_timesteps / 10`, rounded to a multiple of
the episode length for clean rollout boundaries.

---

## 8. Benchmark Policies

Three classical policies are evaluated for comparison.

### 8.1 Base Stock Policy

Order quantity is chosen to bring the **inventory position** (on-hand + pipeline)
up to a fixed target level `S`:

```
q_t  =  max( 0,  S - I_t - Σ_{ℓ=1}^{L} p_t^(ℓ) )
```

Three variants of `S` are tested automatically, based on average demand `μ̄`:

| Variant | S |
|---------|---|
| Conservative | `round( μ̄ · L )` |
| Middle | `round( μ̄ · (L+1) )` |
| Aggressive | `round( μ̄ · (L+2) )` |

### 8.2 Static (s, S) Policy

A classical two-bin reorder policy. Parameters are derived from observed demand:

```
s  =  μ̄ · L          (reorder point)
S  =  s  +  μ̄  +  σ  (order-up-to level)

q_t = S - IP_t   if  IP_t ≤ s,   else  0
```

where `IP_t = I_t + Σ p_t^(ℓ)` is the inventory position and `σ` is the demand
standard deviation.

### 8.3 Forecast-Based Order-Up-To

A dynamic policy that sets the target each period based on the forward forecast:

```
Target_t  =  Σ_{n=0}^{L} f_t^(n)

q_t  =  max( 0,  Target_t - IP_t )
```

This orders enough to cover total expected demand over the lead time plus one
review period.

---

## 9. Scenario Averaging

When multiple demand scenarios are provided, the agent is trained on the shared
historical demand series, then **evaluated independently on each scenario**. The
reported planning records are the **week-by-week arithmetic mean** across scenarios:

```
ŷ_t  =  (1/K) · Σ_{k=1}^{K}  y_t^(k)
```

for each numeric KPI `y` (demand, orders, inventory, costs). This produces a
single expected-case plan under scenario uncertainty.

---

## 10. Key Performance Indicators

| KPI | Formula |
|-----|---------|
| Total Cost (€) | `Σ_t ( c_h · I_t + c_o · q_t + c_l · u_t )` |
| Service Level (%) | `100 · (1 - Σ u_t / Σ d_t)` |
| Total Ordered | `Σ_t q_t` |
| Avg Inventory | `(1/T) · Σ_t I_t` |

---

## 11. Limitations

**1. Single-echelon only.**
The model captures one warehouse-to-customer link. Multi-tier supply chains
(supplier → DC → store) with upstream variability are not represented.

**2. Deterministic lead time.**
Lead time `L` is fixed and known. Stochastic lead times (e.g., supplier delays)
are not modelled, which can cause underperformance in practice.

**3. Lost-sales only, no backorders.**
The environment penalises unmet demand but does not carry it forward. If the
real system does backfill orders, the cost model is mis-specified.

**4. Short episodes and limited data.**
PPO is an on-policy algorithm: it discards experience after each update. With
typical weekly data covering 1–3 years (~52–156 steps per episode), the agent
has very few distinct episodes to learn from. This limits generalisation
compared to off-policy methods (DQN, SAC) that reuse experience.

**5. Single SKU, single location.**
One agent is trained per product-location pair. There is no cross-product
knowledge transfer, so training costs scale linearly with the number of SKUs.

**6. No supplier constraints.**
The model assumes unlimited supply. Minimum order quantities, supplier capacity
caps, and quantity discounts are not encoded.

**7. Point forecasts only.**
The agent receives point forecasts, not distributional ones. It cannot reason
about forecast uncertainty or hedge against forecast error explicitly.

**8. Reward scaling heuristic.**
Dividing the reward by 1,000 is a fixed heuristic. If cost parameters change
dramatically (e.g., `c_l` increases to 50,000), gradient magnitudes shift and
the agent may learn slowly or diverge without re-tuning this constant.

**9. Continuous-to-discrete action rounding.**
Rounding `a_t · Q_max` to the nearest integer creates a non-differentiable
discontinuity that the policy gradient must approximate through sampling noise.
For large `Q_max` the effective action resolution is fine, but for small
ranges the gradient signal is noisy.

---

## 12. Potential Improvements

**A. Use an off-policy algorithm (SAC or TD3).**
Soft Actor-Critic maintains a replay buffer and can reuse every past transition.
This is especially valuable on short episode horizons where on-policy PPO
has little data per update.

**B. Distributional state representation.**
Replace the point forecast vector with a forecast mean + standard deviation pair
per week. The agent can then explicitly hedge against forecast uncertainty,
which is often the dominant source of cost variance.

**C. Stochastic lead time.**
Sample lead time from a distribution (e.g., Poisson) during training so the
policy becomes robust to supplier variability. The pipeline vector then needs
to be replaced by a probability mass function over arrival times.

**D. Multi-product / multi-echelon environment.**
Train a single agent across many SKUs using product-level features (demand
scale, lead time, cost ratios) as additional state dimensions. This enables
knowledge transfer and reduces per-SKU training cost.

**E. Curriculum learning.**
Start training on easy scenarios (low demand variability, short lead time)
and gradually increase difficulty. This can substantially speed up convergence
when real data has high variance.

**F. Normalise reward by cost scale.**
Compute a dynamic reward scale factor (e.g., `max(c_l · peak_demand, 1)`)
rather than the hardcoded `/1000`, so the algorithm remains well-conditioned
when cost parameters vary across products.

**G. Explicit backorder modelling.**
Add a backlog state variable and a backorder penalty `c_b · backlog_t` so the
environment can represent systems where unmet demand is deferred rather than
permanently lost.

**H. Hyperparameter search by cost outcome.**
The current `tune.py` grid-searches over PPO hyperparameters independently for
each product. A Bayesian optimisation loop (e.g., Optuna) with the total cost
as the objective would find good configurations in fewer evaluations.

**I. Ensemble policy under scenario uncertainty.**
Instead of averaging scenario outcomes post-hoc, train a separate policy for
each scenario and combine their order recommendations (e.g., via a weighted
average based on scenario probabilities). This produces a stochastic-robust plan.

**J. Action masking for minimum order quantities.**
If supplier constraints impose a minimum order quantity `q_min`, mask actions
below `q_min / Q_max` during policy rollout so the agent never places
infeasibly small orders.
