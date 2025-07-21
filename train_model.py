import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the ID column if it exists
df = df.drop('Id', axis=1, errors='ignore')

# Split into features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'Iris_model.joblib')
print("Model saved as Iris_model.joblib")
