import pandas as pd
import os

# Load train set (with labels)
orders = pd.read_csv("data/orders.csv")
order_products_train = pd.read_csv("data/order_products__train.csv")

# Merge to get user_id for each order in train
orders_train = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]
train_labeled = order_products_train.merge(orders_train, on="order_id", how="left")

# Build user-product label data
user_product_labels = train_labeled.groupby(["user_id", "product_id"]).agg(
    reordered=("reordered", "max")  # 1 if reordered at least once in train order
).reset_index()

# Save it
os.makedirs("data", exist_ok=True)
user_product_labels.to_csv("data/user_product_history.csv", index=False)

print("✅ user_product_history.csv rebuilt with reorder labels.")
