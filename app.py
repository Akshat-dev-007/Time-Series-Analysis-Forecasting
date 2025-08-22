# app.py
from flask import Flask, render_template
import pandas as pd
from prophet import Prophet
import numpy as np
import os

# --- IMPORTANT FIX FOR FLASK ---
# Use a non-interactive backend for Matplotlib to prevent GUI errors in the server environment.
# This must be done BEFORE importing pyplot.
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------

app = Flask(__name__)


def get_model_outputs():
    """
    This function now performs several key tasks:
    1. Loads and prepares data.
    2. Trains the Prophet model on the entire dataset.
    3. Generates a future forecast up to the end of 2025.
    4. Creates and saves the forecast and components plots as image files.
    5. Merges actual and forecast data for the calendar.
    6. Returns a dictionary with all the necessary data and filenames for the web app.
    """
    # --- 1. Load and Prepare Data ---
    df = pd.read_csv('delhi_aqi.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    daily_df = df.resample('D').mean()
    daily_df.reset_index(inplace=True)

    prophet_df = daily_df[['date', 'pm2_5']].copy()
    prophet_df.rename(columns={'date': 'ds', 'pm2_5': 'y'}, inplace=True)
    prophet_df.dropna(inplace=True)

    # --- 2. Define Holidays ---
    holidays_df = pd.DataFrame({
        'holiday': 'Major Holiday',
        'ds': pd.to_datetime(
            ['2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12', '2024-11-01', '2025-10-21',  # Diwali
             '2021-03-29', '2022-03-18', '2023-03-07', '2024-03-25', '2025-03-14']),  # Holi
        'lower_window': -2,
        'upper_window': 3,
    })

    # --- 3. Train Model on ALL Data ---
    model = Prophet(holidays=holidays_df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)

    # --- 4. Create Future DataFrame until end of 2025 ---
    last_date = prophet_df['ds'].max()
    days_to_forecast = (pd.to_datetime('2025-12-31') - last_date).days
    future_df = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future_df)

    # --- 5. Generate and Save Plots ---
    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    forecast_plot_path = os.path.join('static', 'forecast_plot.png')
    components_plot_path = os.path.join('static', 'components_plot.png')

    # Create and save the forecast plot
    fig1 = model.plot(forecast, figsize=(12, 6))
    plt.title('Delhi PM2.5 Forecast', fontsize=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Predicted PM2.5 (µg/m³)', fontsize=14)
    plt.savefig(forecast_plot_path, bbox_inches='tight')
    plt.close(fig1)

    # Create and save the components plot
    fig2 = model.plot_components(forecast, figsize=(12, 8))
    plt.savefig(components_plot_path, bbox_inches='tight')
    plt.close(fig2)

    # --- 6. Merge Actuals and Forecasts for Calendar ---
    combined_df = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='right')
    combined_df['yhat'] = combined_df['yhat'].round(2)
    combined_df['ds'] = combined_df['ds'].dt.strftime('%Y-%m-%d')
    combined_df['y'] = combined_df['y'].replace({np.nan: None})

    return {
        "calendar_data": combined_df.to_dict('records'),
        "forecast_plot_filename": "forecast_plot.png",
        "components_plot_filename": "components_plot.png"
    }


# --- Global variable to hold all outputs ---
MODEL_OUTPUTS = get_model_outputs()


@app.route('/')
def home():
    # Pass the data and plot filenames to the template
    return render_template(
        'index.html',
        combined_data=MODEL_OUTPUTS["calendar_data"],
        forecast_plot_filename=MODEL_OUTPUTS["forecast_plot_filename"],
        components_plot_filename=MODEL_OUTPUTS["components_plot_filename"]
    )


if __name__ == '__main__':
    app.run(debug=True)
