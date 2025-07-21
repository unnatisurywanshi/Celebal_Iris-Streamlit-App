import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set page config
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('Iris_model.joblib')

model = load_model()

# Load the dataset for visualization
@st.cache_data
def load_data():
    df = pd.read_csv('Iris.csv')
    df = df.drop('Id', axis=1, errors='ignore')
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Home", "Prediction", "Visualization", "About"])

# Home Page
if options == "Home":
    st.title("Iris Flower Species Classification")
    st.image("https://archive.ics.uci.edu/ml/assets/MLimages/Large53.jpg", width=500)
    st.write("""
    ### About the Iris Dataset
    The Iris dataset contains measurements for 150 iris flowers from three different species:
    - Iris-setosa
    - Iris-versicolor
    - Iris-virginica
    
    The measurements include:
    - Sepal length (cm)
    - Sepal width (cm)
    - Petal length (cm)
    - Petal width (cm)
    
    This app uses a machine learning model to predict the species of an iris flower based on these measurements.
    """)

# Prediction Page
elif options == "Prediction":
    st.title("Iris Species Prediction")
    st.write("Enter the measurements of the iris flower to predict its species.")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 
                                min_value=4.0, 
                                max_value=8.0, 
                                value=5.8, 
                                step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", 
                               min_value=2.0, 
                               max_value=4.5, 
                               value=3.5, 
                               step=0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 
                                min_value=1.0, 
                                max_value=7.0, 
                                value=4.3, 
                                step=0.1)
        petal_width = st.slider("Petal Width (cm)", 
                               min_value=0.1, 
                               max_value=2.5, 
                               value=1.3, 
                               step=0.1)
    
    # Make prediction
    if st.button("Predict Species"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Result")
        st.success(f"The predicted species is: **{prediction[0]}**")
        
        # Display probabilities
        st.write("Prediction probabilities:")
        prob_df = pd.DataFrame({
            'Species': model.classes_,
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Species'))

# Visualization Page
elif options == "Visualization":
    st.title("Data Visualization")
    st.write("Explore the Iris dataset through visualizations.")
    
    # Pairplot
    st.subheader("Pairplot of Features")
    fig = sns.pairplot(df, hue="Species", height=2.5)
    st.pyplot(fig)
    
    # Box plots
    st.subheader("Feature Distributions by Species")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, feature in enumerate(df.columns[:-1]):
        sns.boxplot(x='Species', y=feature, data=df, ax=axes[i//2, i%2])
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.drop('Species', axis=1).corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# About Page
elif options == "About":
    st.title("About")
    st.write("""
    ### Iris Species Classification App
    This app uses a Random Forest Classifier to predict the species of an iris flower based on its measurements.
    
    **Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
    
    **Model:** Random Forest Classifier (accuracy: ~96%)
    
    **Features Used:**
    - Sepal Length (cm)
    - Sepal Width (cm)
    - Petal Length (cm)
    - Petal Width (cm)
    
    **Target Variable:** Species (Iris-setosa, Iris-versicolor, Iris-virginica)
    """)
