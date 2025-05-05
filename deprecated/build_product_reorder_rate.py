import pandas as pd

# Load prior orders
prior = pd.read_csv("data/order_products__prior.csv")

# Compute reorder rate per product
product_reorder_rate = (
    prior.groupby("product_id")["reordered"]
    .mean()
    .reset_index()
    .rename(columns={"reordered": "product_reorder_rate"})
)

# Save
product_reorder_rate.to_csv("data/product_reorder_rate.csv", index=False)
print("✅ product_reorder_rate.csv saved to data/")
