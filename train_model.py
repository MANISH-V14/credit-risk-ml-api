import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

n = 5000

annual_income = np.random.randint(20000, 200000, n)
loan_amount = np.random.randint(1000, 100000, n)
credit_score = np.random.randint(300, 850, n)
years_employed = np.random.randint(0, 30, n)
existing_debt = np.random.randint(0, 100000, n)
num_credit_cards = np.random.randint(0, 10, n)

# Risk logic (realistic)
risk = (
    (credit_score < 550) |
    (existing_debt > annual_income * 0.8) |
    (loan_amount > annual_income * 0.7) |
    (years_employed < 2)
)

target = risk.astype(int)

df = pd.DataFrame({
    "annual_income": annual_income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "years_employed": years_employed,
    "existing_debt": existing_debt,
    "num_credit_cards": num_credit_cards,
    "target": target
})

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1),
    df["target"],
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")

joblib.dump(model, "credit_model.pkl")
print("Model saved as credit_model.pkl")