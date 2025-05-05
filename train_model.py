import pandas as pd
import joblib
from xgboost import XGBClassifier
import os

# Load data
print("📦 Loading data...")
user_features = pd.read_csv("data/user_features.csv")
user_product_history = pd.read_csv("data/user_product_history.csv")
products_df = pd.read_csv("data/products.csv")
product_reorder_rate = pd.read_csv("data/product_reorder_rate.csv")

# Merge user and product-level features
print("🔗 Merging features...")
train_df = user_product_history.merge(user_features, on="user_id", how="left")
train_df = train_df.merge(product_reorder_rate, on="product_id", how="left")

# Drop rows with missing target
train_df = train_df.dropna(subset=["reordered"])

# Separate features and label
X = train_df.drop(columns=["user_id", "product_id", "reordered"])
y = train_df["reordered"]

# Train model
print("🧠 Training model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
print("✅ Model trained and saved to models/final_model.pkl")
