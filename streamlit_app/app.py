import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets from parent folders
model = joblib.load("models/final_model.pkl")
user_features = pd.read_csv("../data/user_features.csv")
products_df = pd.read_csv("../data/products.csv")
user_product_history = pd.read_csv("../data/user_product_history.csv")
product_reorder_rate = pd.read_csv("../data/product_reorder_rate.csv")

# Rename columns if needed
user_product_history = user_product_history.rename(columns={
    "user_product_last_order": "user_product_last_order_number"
})

@st.cache_data
def predict_user_top_bottom_products(user_id):
    user_df = user_features[user_features["user_id"] == user_id]
    if user_df.empty:
        return [], []

    input_df = products_df.drop(columns=["product_name"], errors="ignore").copy()
    input_df["user_id"] = user_id
    input_df = input_df.merge(user_df, on="user_id", how="left")
    input_df = input_df.merge(user_product_history, on=["user_id", "product_id"], how="left")
    input_df = input_df.merge(product_reorder_rate, on="product_id", how="left")

    input_df["user_product_last_order_number"] = input_df.get("user_product_last_order_number", 0)
    input_df["user_product_reorder_count"] = input_df.get("user_product_reorder_count", 0)
    input_df["product_reorder_rate"] = input_df["product_reorder_rate"].fillna(0)
    input_df.fillna(0, inplace=True)

    feature_cols = model.get_booster().feature_names
    X = input_df.reindex(columns=feature_cols, fill_value=0)
    input_df["score"] = model.predict(X)

    # Adjust scoring
    if "product_reorder_prob" in input_df.columns:
        input_df["adjusted_score"] = input_df["score"] - 0.4 * input_df["product_reorder_prob"]
    else:
        input_df["adjusted_score"] = input_df["score"]

    if "user_product_reorder_count" in input_df.columns:
        input_df["adjusted_score"] += 0.3 * input_df["user_product_reorder_count"]

    input_df["adjusted_score"] += np.random.normal(0, 0.01, size=len(input_df))

    input_df = input_df.merge(products_df[["product_id", "product_name"]], on="product_id", how="left")
    sorted_df = input_df.sort_values("adjusted_score", ascending=False)

    top5 = sorted_df.head(5)["product_name"].tolist()
    bottom5 = sorted_df.tail(5)["product_name"].tolist()
    return top5, bottom5

# --- Streamlit UI ---
st.set_page_config(page_title="Instacart Reorder Predictor", layout="centered")

st.title("🛒 Instacart Reorder Prediction App")
st.write("Enter a `user_id` to view their top and bottom predicted reorders.")

user_id_input = st.number_input("User ID", min_value=1, value=100, step=1)

if st.button("Predict"):
    top5, bottom5 = predict_user_top_bottom_products(user_id_input)

    st.subheader("🥇 Top 5 Recommendations")
    for item in top5:
        st.markdown(f"- {item}")

    st.subheader("🚫 Bottom 5 Recommendations")
    for item in bottom5:
        st.markdown(f"- {item}")
