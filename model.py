import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset (CSV must be in same folder)
df = pd.read_csv("bike_loan_decision_tree.csv")

X = df[['income', 'age', 'credit_score']]
y = df['loan_approved']

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

def predict_loan(income, age, credit):
    return int(model.predict([[income, age, credit]])[0])
