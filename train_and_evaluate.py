import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

# --- Load Data ---
user_features = pd.read_csv("data/user_features.csv")
user_product_history = pd.read_csv("data/user_product_history.csv")

# Make sure 'reordered' exists
if "reordered" not in user_product_history.columns:
    raise ValueError("❌ 'reordered' column not found in user_product_history.csv")

# --- Merge features ---
train_df = user_product_history.merge(user_features, on="user_id", how="left")
train_df = train_df.fillna(0)

print("Unique training pairs:", train_df[["user_id", "product_id"]].nunique())

# --- Split X/y ---
X = train_df.drop(columns=["user_id", "product_id", "reordered"])
y = train_df["reordered"]

# --- Train-test split ---
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Model with Regularization to prevent overfitting ---
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    eval_metric="logloss",
    use_label_encoder=False
)

# --- Train Model ---
model.fit(X_train, y_train)

xgb.plot_importance(model)
import matplotlib.pyplot as plt
plt.show()

# --- Evaluate ---
y_pred = model.predict(X_valid)
print("\n📊 Classification Report:\n")
print(classification_report(y_valid, y_pred))

# --- AUC Score ---
y_proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_proba)
print(f"\n🧠 ROC AUC: {auc:.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
print("\n✅ Model trained and saved to models/final_model.pkl")
