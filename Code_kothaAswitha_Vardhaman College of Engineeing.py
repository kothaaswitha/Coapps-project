import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    return data

def train_model(algorithm, X_train, y_train):
    if algorithm == 'Random Forest':
        model = RandomForestRegressor()
    elif algorithm == 'Gradient Boosting':
        model = GradientBoostingRegressor()
    elif algorithm == 'Linear Regression':
        model = LinearRegression()
    else:
        raise ValueError("Invalid algorithm selected")

    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)

def main():
    st.title('Energy Demand Forecasting with IoT Data')
    algorithm = st.sidebar.selectbox('Select Algorithm', ['Random Forest', 'Gradient Boosting', 'Linear Regression'])
    file_path = st.sidebar.file_uploader('Upload IoT Data', type=['csv'])
    if file_path:
        data = load_data(file_path)
        data = preprocess_data(data)
        st.subheader('Loaded Data')
        st.write(data)
        X = data.drop(columns=['serial', 'Time_stamp','kVAR'])
        y = data['kVAR']
        model = train_model(algorithm, X, y)
        st.subheader('Predict on Test Dataset')
        serial = st.number_input('Serial', value=0)
        kWh = st.number_input('kWh', value=0.000 )
        kW = st.number_input('kW', value=0.000)
        kVARh = st.number_input('kVARh', value=0.000)
        if st.button('Predict'):
            test_data = pd.DataFrame({'serial': [serial], 'kWh': [kWh], 'kW': [kW], 'kVARh': [kVARh]})
            test_X = test_data.drop(columns=['serial'])
            prediction = predict(model, test_X)
            st.subheader('Prediction')
            st.write(prediction)
if __name__ == '__main__':
    main()
