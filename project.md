Act as an expert Machine Learning Engineer specializing in Deep Reinforcement Learning (DRL) and supply chain optimization. 

I need you to write a clean, well-documented Python script that trains an RL agent to manage a single-echelon inventory system using the Proximal Policy Optimization (PPO) algorithm. 

Please use `gymnasium` for the custom environment and `stable-baselines3` for the PPO implementation. 

### 1. Business & Data Context
The agent will act on a single-stage environment: one specific product at one specific location (e.g., "Ice Cream Strawberry Flavor" at "Logistics Hub Cologne"). You can assume products are produced in factories and shipped to these hubs.

**Unit Definition:** All quantities (demand, inventory, forecast, actions) are strictly integer values representing units of 1,000 KG (e.g., a demand of 70 means 70,000 KG).

Data is given in `Sample Data RL4IM UPDATED.xlsx` in the same folder.

The data structures (assume loaded via pandas) include:
* **Demand:** Historical actual demand per week (integer).
* **Current Inventory:** Starting stock level for the product-location (integer).
* **Lead Time:** A single, fixed delivery delay in weeks for this product-location.
* **Forecast:** Future predicted demand per week (integer).
* **Forecast Accuracy:** Percentage accuracy of past forecasts.

### 2. Custom Environment Specification (`SingleEchelonEnv`)
Create a custom `gymnasium.Env` representing this single node:

* **Observation Space (State):** A continuous 1D vector consisting of:
    1.  Current Inventory Level.
    2.  Pipeline Inventory (orders placed but not yet arrived, represented as a list or array of length equal to `Lead_Time`).
    3.  Forecasted demand for the next `N` weeks.
    *(Note: Ensure observations are properly scaled/normalized for the PPO neural network).*

* **Action Space:** A continuous space `Box(low=0.0, high=1.0)` which gets scaled up by a `MAX_ORDER_QUANTITY` and rounded to the nearest integer inside the environment step to reflect the 1,000 KG unit requirement.

* **Economic Parameters & Reward Function:** The goal is to minimize total costs (maximize negative costs).
    * **Holding Cost:** 13 € per unit (per 1,000 KG) per week.
    * **Ordering Cost:** 60 € per unit (per 1,000 KG) ordered.
    * **Lost Sales Cost:** 2,500 € per unit (per 1,000 KG) of unmet demand.
    * *Reward formula:* `-( (Holding_Cost * end_of_week_inventory) + (Ordering_Cost * order_quantity) + (Lost_Sales_Cost * unmet_demand) )`

* **Step Transition Logic (Lost Sales Model):**
    1.  Receive incoming order from the pipeline (based on Lead Time) and add to Current Inventory.
    2.  Observe actual demand for the current week (from the historical Demand data).
    3.  Satisfy demand: 
        * If `Current Inventory >= demand`, unmet demand is 0.
        * If `Current Inventory < demand`, calculate `unmet_demand = demand - Current Inventory`. Current Inventory then drops to 0 (Lost Sales model, no backlogging).
    4.  Read the agent's Action (new order quantity, rounded to integer). 
    5.  Calculate the Reward using the costs defined above.
    6.  Add the new order to the end of the Pipeline Inventory and shift the pipeline forward by 1 time step.
    7.  Update the step counter and return `obs, reward, terminated, truncated, info`.

### 3. Code Requirements
* **Keep it simple and modular:** Focus on a clean `__init__`, `reset`, and `step` function. Keep in mind that this will later be expanded to a multi-stage network.
* **Dummy Data Loading:** Write a small mock function to simulate extracting a 1D numpy array for demand, forecast, starting inventory, and fixed lead time for ONE product/location combination so the code is runnable out of the box. Ensure the dummy data uses integers.
* **Training Loop:** Initialize `model = PPO('MlpPolicy', env, verbose=1)` and call `model.learn(total_timesteps=20000)`.
* **Evaluation:** Show a quick loop evaluating the trained model for one full episode, printing the Step, Actual Demand, Action (Order Qty), Unmet Demand, Final Inventory, and Reward.