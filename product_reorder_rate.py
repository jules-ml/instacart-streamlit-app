import pandas as pd
import os

# Load data
prior = pd.read_csv("data/order_products__prior.csv")

# Calculate reorder rate per product
product_reorder_stats = prior.groupby("product_id").agg(
    total_orders=('order_id', 'count'),
    total_reorders=('reordered', 'sum')
).reset_index()

product_reorder_stats["product_reorder_rate"] = (
    product_reorder_stats["total_reorders"] / product_reorder_stats["total_orders"]
)

# Save to file
os.makedirs("data", exist_ok=True)
product_reorder_stats[["product_id", "product_reorder_rate"]].to_csv("data/product_reorder_rate.csv", index=False)

print("✅ Saved product_reorder_rate.csv to /data")
