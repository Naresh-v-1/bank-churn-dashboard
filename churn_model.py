import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# ---------------------------
# 1. LOAD DATA
# ---------------------------
df = pd.read_csv("bank_churn.csv")
print("Data shape:", df.shape)
print(df.head())

# ---------------------------
# 2. CLEAN DATA
# ---------------------------
# Drop ID-like columns if they exist
for col in ["RowNumber", "CustomerId", "Surname"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Target column is usually "Exited"
target_col = "Exited"
X = df.drop(columns=target_col)
y = df[target_col]

# ---------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 4. HANDLE CLASS IMBALANCE (SMOTE)
# ---------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())

# ---------------------------
# 5. TRAIN MODELS
# ---------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", round(auc, 4))
    results[name] = {"model": model, "auc": auc, "preds": preds, "probs": probs}

# Pick best model by AUC
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ---------------------------
# 6. CONFUSION MATRIX (best model)
# ---------------------------
cm = confusion_matrix(y_test, results[best_name]["preds"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# ---------------------------
# 7. SHAP EXPLAINABILITY (best model)
# ---------------------------
try:
    explainer = shap.Explainer(best_model, X_train_scaled, feature_names=X.columns)
    shap_values = explainer(X_test_scaled[:200], check_additivity=False)  # sample for speed
    shap.summary_plot(shap_values, X_test.iloc[:200], show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("Saved shap_summary.png")
except Exception as e:
    print("SHAP failed (optional step):", e)

# ---------------------------
# 8. CUSTOMER SEGMENTATION + BUSINESS LAYER
# ---------------------------
X_test_copy = X_test.copy()
X_test_copy["churn_prob"] = results[best_name]["probs"]
X_test_copy["actual_exited"] = y_test.values

# Value proxy: Balance if available, else EstimatedSalary
value_col = "Balance" if "Balance" in X_test_copy.columns else "EstimatedSalary"

def segment(row):
    high_risk = row["churn_prob"] >= 0.5
    high_value = row[value_col] >= X_test_copy[value_col].median()
    if high_risk and high_value:
        return "High Risk - High Value"
    elif high_risk and not high_value:
        return "High Risk - Low Value"
    else:
        return "Low Risk"

X_test_copy["segment"] = X_test_copy.apply(segment, axis=1)
print("\nSegment counts:\n", X_test_copy["segment"].value_counts())

action_map = {
    "High Risk - High Value": "Priority retention call + personalized offer (e.g., fee waiver, rate upgrade)",
    "High Risk - Low Value": "Automated email/SMS retention campaign with small incentive",
    "Low Risk": "No action needed, monitor periodically"
}
X_test_copy["recommended_action"] = X_test_copy["segment"].map(action_map)
X_test_copy.to_csv("churn_predictions_with_segments.csv", index=False)
print("Saved churn_predictions_with_segments.csv")

# ---------------------------
# 9. REVENUE IMPACT ESTIMATE
# ---------------------------
AVG_CUSTOMER_VALUE = X_test_copy[value_col].mean() * 0.05  # assume 5% annual revenue yield
high_risk_high_value_count = (X_test_copy["segment"] == "High Risk - High Value").sum()
ASSUMED_RETENTION_SUCCESS_RATE = 0.30  # 30% of targeted high-value at-risk customers retained

revenue_protected = (
    high_risk_high_value_count * AVG_CUSTOMER_VALUE * ASSUMED_RETENTION_SUCCESS_RATE
)
print(f"\nEstimated High Risk-High Value customers: {high_risk_high_value_count}")
print(f"Estimated annual revenue protected (assuming {int(ASSUMED_RETENTION_SUCCESS_RATE*100)}% retention success): ${revenue_protected:,.2f}")

print("\nDONE. Check confusion_matrix.png, shap_summary.png, and churn_predictions_with_segments.csv")