# run_resume_screener.py — robust demo-ready resume screener
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

DATA_PATH = "resumes.csv"
MODEL_OUT = "resume_screening_model.pkl"
FIG_OUT = "resume_confusion_matrix.png"

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Dataset not found: {DATA_PATH} — put your CSV in the project folder.")

print("📂 Loading resume dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['Resume', 'Category']).reset_index(drop=True)

# Basic validation
if 'Resume' not in df.columns or 'Category' not in df.columns:
    raise SystemExit("CSV must contain 'Resume' and 'Category' columns.")

# Ensure every class has at least 2 samples — duplicate rows for demo datasets
counts = df['Category'].value_counts()
print("\nCategory counts before fix:\n", counts.to_string())

if (counts < 2).any():
    print("\nSmall classes detected — duplicating rows to ensure at least 2 samples per class (demo only).")
    for cat, cnt in counts.items():
        if cnt < 2:
            rows = df[df['Category'] == cat]
            needed = 2 - cnt
            for _ in range(needed):
                df = pd.concat([df, rows], ignore_index=True)
    df = df.reset_index(drop=True)
    counts = df['Category'].value_counts()
    print("\nCategory counts after fix:\n", counts.to_string())

# Vectorize
print("\n🔹 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['Resume'].astype(str))
y = df['Category'].astype(str)

# If we have lots of classes but few samples, evaluate with stratified CV (k=min(5, min_count))
min_count = y.value_counts().min()
cv_splits = min(5, max(2, min_count))  # at least 2, up to 5
print(f"\nUsing Stratified K-Fold CV with {cv_splits} splits for robust evaluation.")

skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
clf = MultinomialNB()

# cross-validated predictions (works better for small datasets)
print("\n🔁 Running cross-validated predictions...")
y_pred_cv = cross_val_predict(clf, X, y, cv=skf)

acc = accuracy_score(y, y_pred_cv)
print(f"\n✅ Cross-validated Accuracy (all data): {acc:.3f}\n")
print("Classification Report (CV):")
print(classification_report(y, y_pred_cv, zero_division=0))

# Confusion matrix for CV predictions (aggregate)
labels = sorted(df['Category'].unique())
cm = confusion_matrix(y, y_pred_cv, labels=labels)
plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) * 0.5)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("Resume Screening — Confusion Matrix (CV)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
plt.close()
print(f"✅ Saved confusion matrix as '{FIG_OUT}'")

# Train final model on full data and save pipeline
print("\n🟦 Training final model on full dataset...")
clf.fit(X, y)
joblib.dump((vectorizer, clf), MODEL_OUT)
print(f"✅ Model pipeline saved to '{MODEL_OUT}'")
print("\n🎯 Done.")