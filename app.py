import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title('SARIMA Model Forecasting (ATLANTA)')

# Fixed SARIMA model parameters
p, d, q = 1, 1, 1
P, D, Q, s = 0, 1, 1, 12

# The coefficients from JMP
ar_coefficient = 0.512021615939496 # Your AR1 coefficient here
ma_coefficient = 0.728544796962296 # Your MA1 coefficient here
seasonal_ma_coefficient = 0.674267607143122 # Your MA1,12 coefficient here

# You may also have an intercept or other parameters; include them as needed
intercept = 973.78596155231 # Your intercept here

# Sidebar for user input to predict periods
st.sidebar.header('Forecast Parameters')
n_periods = st.sidebar.number_input('Enter number of periods to forecast:', min_value=1, value=12, step=1)

# File uploader for new data
uploaded_file = st.file_uploader("Upload your input CSV file (time series data)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.write('Data Preview:')
    st.write(data.head())

    if st.button('Run Forecast'):
        # Construct the SARIMA model with the fixed orders and the JMP coefficients
        mod = sm.tsa.statespace.SARIMAX(
            data['value'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Start with the coefficients from JMP and set them in the model
        params = np.array([ar_coefficient, ma_coefficient, seasonal_ma_coefficient, intercept])
        
        # Fit the model with the specified parameters
        results = mod.filter(params)
        
        # Make forecast
        forecast = results.get_forecast(steps=n_periods)
        forecast_index = pd.date_range(start=data.index[-1], periods=n_periods+1, closed='right')
        forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Predicted'])

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='Historical')
        plt.plot(forecast_df, label='Forecast', color='r')
        plt.fill_between(forecast_index,
                         forecast.conf_int().iloc[:, 0],
                         forecast.conf_int().iloc[:, 1],
                         color='pink', alpha=0.3)
        plt.legend()
        plt.title('SARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        st.pyplot(plt)

        # Show forecast data
        st.write('Forecast Data:')
        st.write(forecast_df)
        st.write('95% Prediction Interval:')
        st.write(forecast.conf_int())
else:
    st.info('Awaiting for CSV file to be uploaded.')
