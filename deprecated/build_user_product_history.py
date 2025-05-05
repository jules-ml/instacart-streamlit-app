import pandas as pd
import os

print("🔧 Rebuilding user_product_history.csv...")

# Load necessary data
orders = pd.read_csv("data/orders.csv")
prior = pd.read_csv("data/order_products__prior.csv")

# Merge prior orders with users
prior_orders = prior.merge(orders[["order_id", "user_id", "order_number"]],
                           on="order_id", how="left")

# Compute reorder count and last order number for user-product pair
user_product_history = prior_orders.groupby(['user_id', 'product_id']).agg(
    user_product_reorder_count=('reordered', 'sum'),
    user_product_last_order_number=('order_number', 'max')
).reset_index()

# Save output
os.makedirs("data", exist_ok=True)
user_product_history.to_csv("data/user_product_history.csv", index=False)

print("✅ user_product_history.csv successfully rebuilt with both reorder count and last order number.")
