import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    try:
        # Try to load pre-trained model
        model = joblib.load('Iris_model.joblib')
    except:
        # If file doesn't exist, train and save a new model
        iris = load_iris()
        X = iris.data
        y = iris.target
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, 'Iris_model.joblib')
    return model

model = load_model()
iris = load_iris()

# Rest of your Streamlit app code...
