Delhi Air Quality (PM2.5) Forecasting Web App
Overview
This is an end-to-end data science project that forecasts the daily PM2.5 air quality levels for Delhi, India. The project involves analyzing historical air quality data, training a time series forecasting model, and deploying the results in an interactive web application built with Flask.

The final application displays a calendar showing the predicted PM2.5 levels for 2023, 2024, and 2025. For past dates where historical data is available, users can click on a day to see a direct comparison between the model's prediction and the actual recorded PM2.5 value. The application also visualizes the overall forecast and the model's learned components (trend, seasonality, and holidays).

Features
Interactive Calendar: Navigate through months to see daily PM2.5 predictions, color-coded by severity.

Forecast vs. Actuals: Click on any past day in 2023 to view a comparison of the predicted value against the actual recorded data.

Long-Range Forecast: The model provides predictions up to the end of 2025.

Data-Driven Visualizations: Includes plots of the full forecast and the model's seasonal components, which are generated automatically.

Automated Data Pipeline: The Flask backend handles data loading, preprocessing, model training, and prediction generation when the server starts.

Tech Stack
Backend: Python, Flask

Data Science & Modeling: Pandas, Prophet (by Facebook), Matplotlib

Frontend: HTML, Tailwind CSS, JavaScript

Environment: PyCharm, Jupyter/Colab (for initial analysis)

Project Structure
The project is organized in a standard Flask application structure:

Time-Series-Analysis-Forecast/
│
├── app.py              # Main Flask application logic, data processing, and model training
├── delhi_aqi.csv       # The historical air quality dataset
├── requirements.txt    # List of Python dependencies
├── README.md           # This file
│
├── static/
│   ├── forecast_plot.png       # Saved image of the forecast plot
│   └── components_plot.png     # Saved image of the components plot
│
└── templates/
    └── index.html      # The frontend HTML, CSS, and JavaScript

Setup and Installation
To run this project locally, follow these steps:

Clone the Repository (or create the project):
Make sure you have all the files (app.py, delhi_aqi.csv, templates/index.html) in the correct structure as shown above.

Install Dependencies:
Install all the required Python libraries using the requirements.txt file.

How It Works
Data Loading: The delhi_aqi.csv file, containing hourly pollution data, is loaded.

Preprocessing: The data is resampled into daily averages to create a stable time series for PM2.5 levels.

Model Training: A Prophet forecasting model is trained on the entire historical dataset. The model is configured to recognize yearly and weekly seasonality, as well as the specific effects of major Indian holidays like Diwali and Holi.

Forecasting: The trained model generates a forecast for all dates from the beginning of the dataset through to the end of 2025.

Plot Generation: The application generates two plots using Matplotlib—the full forecast and its components—and saves them as images in the static/ directory.

Backend Server: The Flask app serves the index.html file, passing the forecast data and plot filenames to it.

Frontend Rendering: The JavaScript in the index.html file dynamically builds the calendar, color-codes each day based on the forecast, and creates the interactive pop-up modals to display detailed information.
