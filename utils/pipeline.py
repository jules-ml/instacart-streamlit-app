import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/final_model.pkl")

# Load data
user_features = pd.read_csv("data/user_features.csv")
products_df = pd.read_csv("data/products.csv")
user_product_history = pd.read_csv("data/user_product_history.csv")
product_reorder_rate = pd.read_csv("data/product_reorder_rate.csv")

# Rename columns if needed
user_product_history = user_product_history.rename(columns={
    "user_product_last_order": "user_product_last_order_number"
})

# Prediction function
def predict_user_top_bottom_products(user_id):
    # Filter user
    user_df = user_features[user_features["user_id"] == user_id]
    if user_df.empty:
        return [], []

    # Build input
    input_df = products_df.drop(columns=["product_name"], errors="ignore").copy()
    input_df["user_id"] = user_id
    input_df = input_df.merge(user_df, on="user_id", how="left")
    input_df = input_df.merge(user_product_history, on=["user_id", "product_id"], how="left")
    input_df = input_df.merge(product_reorder_rate, on="product_id", how="left")

    # Fill missing features
    input_df["user_product_last_order_number"] = input_df.get("user_product_last_order_number", 0)
    input_df["user_product_reorder_count"] = input_df.get("user_product_reorder_count", 0)
    input_df["product_reorder_rate"] = input_df["product_reorder_rate"].fillna(0)

    # Fill remaining NaNs
    input_df.fillna(0, inplace=True)

    # Predict
    feature_cols = model.get_booster().feature_names
    X = input_df.reindex(columns=feature_cols, fill_value=0)
    input_df["score"] = model.predict(X)

    # Adjust score: penalize globally popular items (if available)
    if "product_reorder_prob" in input_df.columns:
        input_df["adjusted_score"] = input_df["score"] - 0.4 * input_df["product_reorder_prob"]
    else:
        input_df["adjusted_score"] = input_df["score"]

    # Boost for user-specific reorder history
    if "user_product_reorder_count" in input_df.columns:
        input_df["adjusted_score"] += 0.3 * input_df["user_product_reorder_count"]

    # Add slight random noise to break ties
    input_df["adjusted_score"] += np.random.normal(0, 0.01, size=len(input_df))

    # Add product names back
    input_df = input_df.merge(products_df[["product_id", "product_name"]], on="product_id", how="left")

    # Sort using adjusted score
    sorted_df = input_df.sort_values("adjusted_score", ascending=False)

    top5 = sorted_df.head(5)["product_name"].tolist()
    bottom5 = sorted_df.tail(5)["product_name"].tolist()
    return top5, bottom5
