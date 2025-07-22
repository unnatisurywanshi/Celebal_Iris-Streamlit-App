import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)

# Encode species labels
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor")
st.markdown("Enter the measurements below to predict the species.")

# Sidebar for user inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()), 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()), 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()), 1.5)
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()), 0.2)

# Predict
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_features)
predicted_species = le.inverse_transform(prediction)[0]

st.success(f"🌼 Predicted Species: **{predicted_species}**")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Pairplot visualization
st.subheader("📊 Data Visualization")
if st.checkbox("Show Pairplot (Takes a moment)"):
    fig = sns.pairplot(df, hue="Species", palette="bright", diag_kind="hist")
    st.pyplot(fig)
