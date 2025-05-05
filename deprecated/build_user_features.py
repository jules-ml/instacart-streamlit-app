import pandas as pd
import os

# Load data
orders = pd.read_csv("data/orders.csv")
train = pd.read_csv("data/order_products__train.csv")

# --- Create basic user features ---
user_features = orders.groupby("user_id").agg(
    user_total_orders=("order_number", "max"),
    user_avg_days_between_orders=("days_since_prior_order", "mean"),
    user_median_days_between_orders=("days_since_prior_order", "median"),
    user_avg_order_hour=("order_hour_of_day", "mean")
).reset_index()

# --- New feature: user_reorder_ratio ---
# Attach user_id to each row in the training set
orders_subset = orders[["order_id", "user_id"]]
train = train.merge(orders_subset, on="order_id", how="left")

# Calculate reorder ratio
user_reorder_ratio = (
    train.groupby("user_id")["reordered"]
    .mean()
    .reset_index()
    .rename(columns={"reordered": "user_reorder_ratio"})
)

# Merge into user_features
user_features = user_features.merge(user_reorder_ratio, on="user_id", how="left")

# Save updated user_features
os.makedirs("data", exist_ok=True)
user_features.to_csv("data/user_features.csv", index=False)
print("✅ user_features.csv updated and saved to /data")
