import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

# Setup
os.makedirs("static", exist_ok=True)

# Load Data
df = pd.read_excel("data/Fraud_Transaction_Detection.xlsx")
df.drop(columns=['Customer Name', 'Customer Email', 'Customer Phone', 'Transaction ID'], inplace=True, errors='ignore')

encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['Is_Fraud'])
y = df['Is_Fraud']

plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette="pastel", legend=False)
plt.title("Original Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/original_class_dist.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### ============================
### WITHOUT SMOTE
### ============================
clf_original = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_original.fit(X_train, y_train)
y_pred_original = clf_original.predict(X_test)

# Metrics Without SMOTE
acc_original = accuracy_score(y_test, y_pred_original)
report_original = classification_report(y_test, y_pred_original, zero_division=0)
cm_original = confusion_matrix(y_test, y_pred_original)
y_proba_original = clf_original.predict_proba(X_test)[:, 1]
fpr_original, tpr_original, _ = roc_curve(y_test, y_proba_original)
roc_auc_original = auc(fpr_original, tpr_original)

# Save Confusion Matrix Without SMOTE
plt.figure(figsize=(6, 4))
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix (Without SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix_no_smote.png")
plt.close()

# Save ROC Curve Without SMOTE
plt.figure(figsize=(6, 4))
plt.plot(fpr_original, tpr_original, label=f"AUC = {roc_auc_original:.2f}", color='purple')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (Without SMOTE)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("static/roc_curve_no_smote.png")
plt.close()

print("="*50)
print("❌ Model WITHOUT SMOTE")
print("="*50)
print(f"Accuracy: {acc_original:.4f}")
print(report_original)

### ============================
### WITH SMOTE
### ============================
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Visualize SMOTE Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_resampled, hue=y_train_resampled, palette="muted", legend=False)
plt.title("SMOTE Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/smote_class_dist.png")
plt.close()

clf_smote = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = clf_smote.predict(X_test)

# Metrics With SMOTE
acc_smote = accuracy_score(y_test, y_pred_smote)
report_smote = classification_report(y_test, y_pred_smote, zero_division=0)
cm_smote = confusion_matrix(y_test, y_pred_smote)
y_proba_smote = clf_smote.predict_proba(X_test)[:, 1]
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_proba_smote)
roc_auc_smote = auc(fpr_smote, tpr_smote)

# Save Confusion Matrix With SMOTE
plt.figure(figsize=(6, 4))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (With SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

# Save ROC Curve With SMOTE
plt.figure(figsize=(6, 4))
plt.plot(fpr_smote, tpr_smote, label=f"AUC = {roc_auc_smote:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve (With SMOTE)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("static/roc_curve.png")
plt.close()

print("="*50)
print("✅ Model WITH SMOTE")
print("="*50)
print(f"Accuracy: {acc_smote:.4f}")
print(report_smote)

# ===========================
# Comparison Table to CSV
# ===========================
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)', 'AUC-ROC'],
    'Without SMOTE': [
        acc_original,
        precision_score(y_test, y_pred_original, average='macro', zero_division=0),
        recall_score(y_test, y_pred_original, average='macro', zero_division=0),
        f1_score(y_test, y_pred_original, average='macro', zero_division=0),
        roc_auc_original
    ],
    'With SMOTE': [
        acc_smote,
        precision_score(y_test, y_pred_smote, average='macro', zero_division=0),
        recall_score(y_test, y_pred_smote, average='macro', zero_division=0),
        f1_score(y_test, y_pred_smote, average='macro', zero_division=0),
        roc_auc_smote
    ]
})

results_df.to_csv("static/smote_vs_no_smote_comparison.csv", index=False)
print("\n📊 Comparison saved to: static/smote_vs_no_smote_comparison.csv")
print("\n📁 Open /static folder to view output graphs and report files.")
