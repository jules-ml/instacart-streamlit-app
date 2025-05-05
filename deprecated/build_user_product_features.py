# build_user_product_features.py
import pandas as pd
import os

# Load data
orders = pd.read_csv("data/orders.csv")
prior = pd.read_csv("data/order_products__prior.csv")

# Merge to get user_id
prior_merged = prior.merge(orders[['order_id', 'user_id']], on='order_id')

# Create user-product reorder count feature
user_prod_stats = prior_merged.groupby(['user_id', 'product_id']) \
    .agg(user_product_reorder_count=('reordered', 'sum')) \
    .reset_index()

# Save to data folder
os.makedirs("data", exist_ok=True)
user_prod_stats.to_csv("data/user_product_reorder_count.csv", index=False)

print("✅ user_product_reorder_count.csv saved to /data")
