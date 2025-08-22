# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
from prophet import Prophet
import numpy as np

app = Flask(__name__)


# --- Data Processing and Model Training ---
# This part runs once when the Flask app starts.
# For a production app, you would save and load a pre-trained model.
def get_forecast_data():
    # 1. Load the dataset
    df = pd.read_csv('delhi_aqi.csv')

    # 2. Convert to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 3. Resample to daily averages
    daily_df = df.resample('D').mean()
    daily_df.reset_index(inplace=True)

    # 4. Prepare for Prophet (using pm2_5 as the target)
    prophet_df = daily_df[['date', 'pm2_5']].copy()
    prophet_df.rename(columns={'date': 'ds', 'pm2_5': 'y'}, inplace=True)
    prophet_df.dropna(inplace=True)

    # 5. Define Indian Holidays
    holidays_df = pd.DataFrame({
        'holiday': 'Major Holiday',
        'ds': pd.to_datetime(['2020-11-14', '2021-11-04', '2022-10-24',  # Diwali
                              '2021-03-29', '2022-03-18']),  # Holi
        'lower_window': -2,
        'upper_window': 3,
    })

    # 6. Initialize and Fit the Prophet model
    model = Prophet(holidays=holidays_df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)

    # 7. Create future dataframe (forecast for the rest of 2023)
    future_df = model.make_future_dataframe(periods=340)  # Forecast until end of 2023
    forecast = model.predict(future_df)

    # 8. Prepare data for the calendar
    forecast_data = forecast[['ds', 'yhat']].copy()
    forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
    forecast_data['yhat'] = forecast_data['yhat'].round(2)

    return forecast_data.to_dict('records')


# --- Global variable to hold the forecast data ---
FORECAST_DATA = get_forecast_data()


# --- Flask Route ---
@app.route('/')
def home():
    # Pass the pre-calculated forecast data to the HTML template
    #return "Hello getting started"
    return render_template('index.html', forecast_data=FORECAST_DATA)


if __name__ == '__main__':
    app.run(debug=True)