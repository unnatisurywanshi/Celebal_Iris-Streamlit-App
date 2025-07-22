# iris_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Iris.csv")

# Clean dataset
df.drop(columns=["Id"], inplace=True)

# Encode labels
df['Species'] = df['Species'].astype('category').cat.codes

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)
