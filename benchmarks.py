import math
import pandas as pd

FILE_PATH = "Sample Data RL4IM UPDATED.xlsx"

PRODUCT = "Ice Cream Strawberry Flavor"
LOCATION = "Logistics Hub Stuttgart"

START_INVENTORY_WEEK = "17.2026"
START_FORECAST_WEEK = "18.2026"

# Same cost parameters as PPO UI
HOLDING_COST_PER_UNIT = 13
ORDERING_COST_PER_UNIT = 60
LOST_SALES_COST_PER_UNIT = 2500

# PPO result from the 10,000 timestep run
PPO_TOTAL_COST = 7053
PPO_SERVICE_LEVEL = 100.00
PPO_TOTAL_ORDERED = 74
PPO_AVG_INVENTORY = 3.14
PPO_FORECAST_WEEKS = 64

# Benchmark parameters
REVIEW_PERIOD = 1
SAFETY_FACTOR = 0.20
MAX_ORDER_QTY = 201


def week_key(week_label):
    week, year = str(week_label).split(".")
    return int(year), int(week)


def read_clean_sheet(file_path, sheet_name_clean):
    """
    Reads an Excel sheet even if the real sheet name has extra spaces.
    Also cleans column names.
    """
    excel_file = pd.ExcelFile(file_path)
    sheet_map = {sheet.strip(): sheet for sheet in excel_file.sheet_names}

    real_sheet_name = sheet_map[sheet_name_clean]
    df = pd.read_excel(file_path, sheet_name=real_sheet_name)

    df.columns = df.columns.astype(str).str.strip()

    return df


def get_row(df, product, location):
    """
    Gets one row for the selected product-location pair.
    """
    row = df[
        (df["Product"] == product) &
        (df["Location"] == location)
    ]

    if row.empty:
        available = df[["Product", "Location"]].drop_duplicates()
        raise ValueError(
            f"No data found for Product='{product}' and Location='{location}'.\n"
            f"Available combinations:\n{available}"
        )

    return row.iloc[0]


def calculate_kpis(records):
    """
    Calculates the same KPIs used for PPO comparison.
    """
    total_demand = sum(r["Demand"] for r in records)
    total_unmet_demand = sum(r["Unmet Demand"] for r in records)
    total_ordered = sum(r["Order Qty"] for r in records)
    total_cost = sum(r["Weekly Cost"] for r in records)
    avg_inventory = sum(r["Ending Inventory"] for r in records) / len(records)

    service_level = 100 * (1 - total_unmet_demand / max(total_demand, 1))

    return {
        "Total Cost (€)": round(total_cost, 2),
        "Service Level (%)": round(service_level, 2),
        "Total Ordered (units)": int(total_ordered),
        "Avg Inventory (units)": round(avg_inventory, 2),
        "Forecast Weeks": len(records),
    }


def simulate_static_sS_policy(
    forecast_values,
    initial_inventory,
    lead_time,
    holding_cost_per_unit,
    ordering_cost_per_unit,
    lost_sales_cost_per_unit,
    max_order_qty,
):
    """
    Static (s, S) inventory heuristic.

    Aligned with PPO step timing:
    1. Receive arriving order
    2. Satisfy current demand
    3. Decide order quantity
    4. Add order to pipeline
    5. Calculate costs
    """

    forecast_values = pd.to_numeric(forecast_values)

    avg_weekly_demand = forecast_values.mean()

    # Static reorder point and order-up-to level
    s = math.ceil(avg_weekly_demand * lead_time)
    S = math.ceil(avg_weekly_demand * (lead_time + REVIEW_PERIOD) * (1 + SAFETY_FACTOR))

    # Make sure S is higher than s
    S = max(S, s + 1)

    inventory = int(initial_inventory)
    pipeline_orders = {}
    records = []

    weeks = list(forecast_values.index)

    for t, week in enumerate(weeks):
        demand = int(forecast_values.loc[week])

        # 1. Receive orders that arrive this week
        arriving_order = pipeline_orders.pop(t, 0)
        inventory += arriving_order
        inventory_after_arrival = inventory

        # 2. Satisfy current demand
        satisfied_demand = min(inventory, demand)
        unmet_demand = max(0, demand - inventory)
        inventory = max(0, inventory - demand)

        # 3. Inventory position after demand
        inventory_position = inventory + sum(pipeline_orders.values())

        # 4. Static (s, S) ordering rule
        if inventory_position <= s:
            order_qty = S - inventory_position
            order_qty = max(0, min(order_qty, max_order_qty))
        else:
            order_qty = 0

        # 5. Costs, same cost structure as PPO
        holding_cost = inventory * holding_cost_per_unit
        ordering_cost = order_qty * ordering_cost_per_unit
        lost_sales_cost = unmet_demand * lost_sales_cost_per_unit
        weekly_cost = holding_cost + ordering_cost + lost_sales_cost

        # 6. Add new order to pipeline
        arrival_time = t + lead_time
        pipeline_orders[arrival_time] = pipeline_orders.get(arrival_time, 0) + order_qty

        records.append({
            "Week": week,
            "Demand": demand,
            "Arriving Order": arriving_order,
            "Inventory After Arrival": inventory_after_arrival,
            "Inventory Position Before Order": inventory_position,
            "Order Qty": order_qty,
            "Satisfied Demand": satisfied_demand,
            "Ending Inventory": inventory,
            "Unmet Demand": unmet_demand,
            "Holding Cost": holding_cost,
            "Ordering Cost": ordering_cost,
            "Lost Sales Cost": lost_sales_cost,
            "Weekly Cost": weekly_cost,
            "s": s,
            "S": S,
        })

    kpis = calculate_kpis(records)

    summary = {
        "Method": "Static (s, S) Heuristic",
        "Product": PRODUCT,
        "Location": LOCATION,
        **kpis,
        "s": s,
        "S": S,
        "Policy Detail": f"Reorder if inventory position <= {s}; order up to {S}",
    }

    records_df = pd.DataFrame(records)

    return summary, records_df


def simulate_forecast_order_up_to_policy(
    forecast_values,
    initial_inventory,
    lead_time,
    holding_cost_per_unit,
    ordering_cost_per_unit,
    lost_sales_cost_per_unit,
    max_order_qty,
):
    """
    Forecast-based Order-up-to Policy.

    Aligned with PPO step timing:
    1. Receive arriving order
    2. Satisfy current demand
    3. Calculate dynamic target stock from future forecast
    4. Decide order quantity
    5. Add order to pipeline
    6. Calculate costs
    """

    forecast_values = pd.to_numeric(forecast_values)

    inventory = int(initial_inventory)
    pipeline_orders = {}
    records = []

    weeks = list(forecast_values.index)

    for t, week in enumerate(weeks):
        demand = int(forecast_values.loc[week])

        # 1. Receive orders that arrive this week
        arriving_order = pipeline_orders.pop(t, 0)
        inventory += arriving_order
        inventory_after_arrival = inventory

        # 2. Satisfy current demand
        satisfied_demand = min(inventory, demand)
        unmet_demand = max(0, demand - inventory)
        inventory = max(0, inventory - demand)

        # 3. Inventory position after demand
        inventory_position = inventory + sum(pipeline_orders.values())

        # 4. Coverage period starts from next week
        # because current demand has already been served.
        coverage_start = t + 1
        coverage_end = min(coverage_start + lead_time + REVIEW_PERIOD, len(weeks))
        coverage_weeks = weeks[coverage_start:coverage_end]

        if coverage_weeks:
            expected_demand = int(forecast_values.loc[coverage_weeks].sum())
        else:
            expected_demand = 0

        target_stock = math.ceil(expected_demand * (1 + SAFETY_FACTOR))

        order_qty = max(0, target_stock - inventory_position)
        order_qty = min(order_qty, max_order_qty)

        # 5. Costs, same cost structure as PPO
        holding_cost = inventory * holding_cost_per_unit
        ordering_cost = order_qty * ordering_cost_per_unit
        lost_sales_cost = unmet_demand * lost_sales_cost_per_unit
        weekly_cost = holding_cost + ordering_cost + lost_sales_cost

        # 6. Add new order to pipeline
        arrival_time = t + lead_time
        pipeline_orders[arrival_time] = pipeline_orders.get(arrival_time, 0) + order_qty

        records.append({
            "Week": week,
            "Demand": demand,
            "Arriving Order": arriving_order,
            "Inventory After Arrival": inventory_after_arrival,
            "Inventory Position Before Order": inventory_position,
            "Expected Demand Coverage": expected_demand,
            "Target Stock": target_stock,
            "Order Qty": order_qty,
            "Satisfied Demand": satisfied_demand,
            "Ending Inventory": inventory,
            "Unmet Demand": unmet_demand,
            "Holding Cost": holding_cost,
            "Ordering Cost": ordering_cost,
            "Lost Sales Cost": lost_sales_cost,
            "Weekly Cost": weekly_cost,
        })

    kpis = calculate_kpis(records)

    summary = {
        "Method": "Forecast-based Order-up-to Policy",
        "Product": PRODUCT,
        "Location": LOCATION,
        **kpis,
        "s": None,
        "S": None,
        "Policy Detail": "Dynamic target stock based on future forecast demand over lead time + review period",
    }

    records_df = pd.DataFrame(records)

    return summary, records_df


# Load data
current_inventory_df = read_clean_sheet(FILE_PATH, "Current Inventory")
lead_time_df = read_clean_sheet(FILE_PATH, "Lead Time")
forecast_df = read_clean_sheet(FILE_PATH, "Forecast")

# Select product-location rows
inventory_row = get_row(current_inventory_df, PRODUCT, LOCATION)
lead_time_row = get_row(lead_time_df, PRODUCT, LOCATION)
forecast_row = get_row(forecast_df, PRODUCT, LOCATION)

# Extract input values
initial_inventory = int(inventory_row[START_INVENTORY_WEEK])
lead_time = int(lead_time_row["Lead Time in weeks"])

forecast_week_columns = [
    col for col in forecast_df.columns
    if col not in ["Product", "Location"]
]

forecast_week_columns = sorted(forecast_week_columns, key=week_key)

forecast_weeks = [
    week for week in forecast_week_columns
    if week_key(week) >= week_key(START_FORECAST_WEEK)
]

forecast_values = forecast_row[forecast_weeks]

# Run Static (s, S) benchmark
sS_summary, sS_records = simulate_static_sS_policy(
    forecast_values=forecast_values,
    initial_inventory=initial_inventory,
    lead_time=lead_time,
    holding_cost_per_unit=HOLDING_COST_PER_UNIT,
    ordering_cost_per_unit=ORDERING_COST_PER_UNIT,
    lost_sales_cost_per_unit=LOST_SALES_COST_PER_UNIT,
    max_order_qty=MAX_ORDER_QTY,
)

# Run Forecast-based Order-up-to benchmark
order_up_to_summary, order_up_to_records = simulate_forecast_order_up_to_policy(
    forecast_values=forecast_values,
    initial_inventory=initial_inventory,
    lead_time=lead_time,
    holding_cost_per_unit=HOLDING_COST_PER_UNIT,
    ordering_cost_per_unit=ORDERING_COST_PER_UNIT,
    lost_sales_cost_per_unit=LOST_SALES_COST_PER_UNIT,
    max_order_qty=MAX_ORDER_QTY,
)

# PPO summary row
ppo_summary = {
    "Method": "PPO / RL (10,000 timesteps)",
    "Product": PRODUCT,
    "Location": LOCATION,
    "Total Cost (€)": PPO_TOTAL_COST,
    "Service Level (%)": PPO_SERVICE_LEVEL,
    "Total Ordered (units)": PPO_TOTAL_ORDERED,
    "Avg Inventory (units)": PPO_AVG_INVENTORY,
    "Forecast Weeks": PPO_FORECAST_WEEKS,
    "s": None,
    "S": None,
    "Policy Detail": "Learned PPO replenishment policy",
}

comparison_df = pd.DataFrame([
    ppo_summary,
    sS_summary,
    order_up_to_summary,
])

print("\nBenchmark comparison")
print("--------------------")
print(comparison_df)

# Export to Excel
OUTPUT_FILE = "benchmark_comparison.xlsx"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    comparison_df.to_excel(writer, sheet_name="Comparison", index=False)
    sS_records.to_excel(writer, sheet_name="Static_sS_Weekly", index=False)
    order_up_to_records.to_excel(writer, sheet_name="Order_up_to_Weekly", index=False)

print(f"\nSaved benchmark comparison to: {OUTPUT_FILE}")