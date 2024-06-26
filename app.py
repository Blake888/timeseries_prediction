import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title('SARIMA Model Forecasting (ATLANTA)')

# Assuming the names of your columns are 'Date' for the date column
# and 'Passengers' for the passenger counts/time series data
date_column = 'Date'
ts_column = 'Passengers'

# Fixed SARIMA model parameters
p, d, q = 1, 1, 1
P, D, Q, s = 0, 1, 1, 12

# The coefficients from JMP
ar_coefficient = 0.512021615939496 # Your AR1 coefficient
ma_coefficient = -0.728544796962296 # Your MA1 coefficient
seasonal_ma_coefficient = -0.674267607143122 # Your MA1,12 coefficient

# You may also have an intercept or other parameters; include them as needed
intercept = 973.78596155231 # Your intercept

# Sidebar 
st.sidebar.header('Forecast Parameters')
# Place the file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file (time series data)", type=["csv"])
# Add some space after the number input
st.sidebar.markdown('---')  # This adds a horizontal line
st.sidebar.markdown('<br>', unsafe_allow_html=True)  # This adds a line break (space)
#for user input to predict periods
n_periods = st.sidebar.number_input('Enter number of months to forecast:', min_value=1, value=12, step=1)
# Add some space after the number input
st.sidebar.markdown('---')  # This adds a horizontal line
st.sidebar.markdown('<br>', unsafe_allow_html=True)  # This adds a line break (space)
# Place the button on the sidebar
run_forecast = st.sidebar.button('Run Forecast')

# Data preview section
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=date_column, parse_dates=[date_column])
    st.write('Data Preview:')
    st.write(data.head())

    if run_forecast:
        # Construct the SARIMA model with the fixed orders and the JMP coefficients
        mod = sm.tsa.statespace.SARIMAX(
            data[ts_column],
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
        forecast_df = pd.DataFrame({
            'Predicted': forecast.predicted_mean,
            'Lower CI': forecast.conf_int().iloc[:, 0],
            'Upper CI': forecast.conf_int().iloc[:, 1]
        })
        
        # Generate the forecast index
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='ME')
        forecast_df.index = forecast_index

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data[ts_column], label='Historical', color='k')
        plt.plot(forecast_df.index, forecast_df['Predicted'], label='Forecast', color='b')
        plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='blue', alpha=0.3)
        plt.legend()
        plt.title('SARIMA Forecast (ATLANTA)')
        plt.xlabel('Date')
        plt.ylabel('Passengers')
        st.pyplot(plt)

        # Plotting the forecast only
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_df.index, forecast_df['Predicted'], label='Forecast', color='b',marker='o')
        plt.xticks(forecast_index, forecast_index.strftime('%Y-%m'), rotation=45) # Format the x-axis to show all months
        plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='blue', alpha=0.3)
        plt.legend()
        plt.title('SARIMA Forecast Only (ATLANTA)')
        plt.xlabel('Date')
        plt.ylabel('Passengers')
        st.pyplot(plt)
        

        # Show forecast data
        st.write('Forecast Data:')
        st.write(forecast_df)
        st.write('95% Prediction Interval:')
        st.write(forecast.conf_int())
else:
    st.info('Awaiting for CSV file to be uploaded.')
