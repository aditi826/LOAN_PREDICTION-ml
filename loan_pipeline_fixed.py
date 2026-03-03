# ============================================================
# LOAN APPROVAL PREDICTION PIPELINE
# Target: loan_status (Approved/Rejected)
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# ─────────────────────────────────────────
# STEP 0: LOAD DATA
# ─────────────────────────────────────────
try:
    df = pd.read_csv("loan_prediction.csv")
except FileNotFoundError:
    print("ERROR: 'loan_prediction.csv' not found!")
    exit(1)

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()

# Strip whitespace from all string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

print("Dataset columns:", df.columns.tolist())
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["loan_status"].value_counts())

# Drop ID column (not a feature)
df.drop(columns=["loan_id"], inplace=True, errors="ignore")

# ─────────────────────────────────────────
# STEP 1: FEATURE ENGINEERING (loan-specific)
# ─────────────────────────────────────────
def engineer_loan_features(df):
    df = df.copy()

    # Total household income
    df["total_income"] = df["income_annum"] + df.get("coapplicant_income", 0)

    # Income to loan ratio (affordability signal)
    df["income_loan_ratio"] = df["total_income"] / (df["loan_amount"] + 1)

    # EMI estimate (monthly payment burden)
    df["emi"] = df["loan_amount"] / (df["loan_term"] + 1)

    # How much income is left after EMI
    df["balance_income"] = df["total_income"] - (df["emi"] * 1000)

    # Total assets
    df["total_assets"] = df["residential_assets_value"] + df["commercial_assets_value"] + df["luxury_assets_value"] + df["bank_asset_value"]

    # Log transform to reduce skewness
    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    df["log_total_income"] = np.log1p(df["total_income"])

    return df

df = engineer_loan_features(df)
print("\nNew features added. Shape:", df.shape)


# ─────────────────────────────────────────
# STEP 2: DEFINE FEATURES & TARGET
# ─────────────────────────────────────────
TARGET = "loan_status"

# Encode target: Approved → 1, Rejected → 0
df[TARGET] = (df[TARGET] == "Approved").astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Auto-detect column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"\nNumeric features ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

# ─────────────────────────────────────────
# STEP 3: SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Approval rate (train): {y_train.mean():.2%}")

# ─────────────────────────────────────────
# STEP 4: BUILD PIPELINE
# ─────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Using Gradient Boosting (best for tabular loan data)
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ─────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────
pipeline.fit(X_train, y_train)
print("\nModel trained successfully!")

# ─────────────────────────────────────────
# STEP 6: EVALUATE
# ─────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"])
plt.title("Loan Approval — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved to confusion_matrix.png")

# Feature Importance
feature_names = num_cols + cat_cols
importances = pipeline.named_steps["model"].feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n── Top 10 Features ──")
print(feat_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df.head(10), x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Features for Loan Approval")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance saved to feature_importance.png")

# ─────────────────────────────────────────
# STEP 7: SAVE PIPELINE
# ─────────────────────────────────────────
joblib.dump(pipeline, "loan_approval_pipeline.joblib")
print("\nPipeline saved → loan_approval_pipeline.joblib")

# ─────────────────────────────────────────
# STEP 8: PREDICT ON NEW APPLICANTS
# ─────────────────────────────────────────
def predict_loan(applicant_data: dict) -> dict:
    """
    Pass a single applicant's raw data as a dict.
    Returns approval decision + probability.
    """
    pipeline = joblib.load("loan_approval_pipeline.joblib")

    df_input = pd.DataFrame([applicant_data])
    df_input = engineer_loan_features(df_input)

    prediction = pipeline.predict(df_input)[0]
    probability = pipeline.predict_proba(df_input)[0][1]

    return {
        "decision": "APPROVED" if prediction == 1 else "REJECTED",
        "approval_probability": f"{probability:.2%}",
        "confidence": "High" if abs(probability - 0.5) > 0.3 else "Low"
    }

# ── Example Usage ──
new_applicant = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000,
}

result = predict_loan(new_applicant)
print("\n── Loan Decision ──")
for k, v in result.items():
    print(f"  {k}: {v}")

print("\n✓ Pipeline completed successfully!")
