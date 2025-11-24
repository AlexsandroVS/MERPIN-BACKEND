import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import os
import json

# Assuming etl_pipeline is in the same directory or accessible via PYTHONPATH
from .etl_pipeline import run_etl_pipeline

def train_and_evaluate_model(df_ts: pd.DataFrame, periods_to_forecast: int = 90, test_size_months: int = 3):
    """
    Trains a Prophet model, evaluates it, and generates a forecast.

    Args:
        df_ts (pd.DataFrame): The processed time series DataFrame with 'ds' and 'y' columns.
        periods_to_forecast (int): Number of future days to forecast.
        test_size_months (int): Number of recent months to use for the test set.

    Returns:
        tuple: A tuple containing:
            - Prophet model object
            - pd.DataFrame: The forecast DataFrame
            - dict: Evaluation metrics (MAPE, RMSE)
            - pd.DataFrame: The test set used for evaluation
    """
    if df_ts.empty:
        raise ValueError("Input DataFrame for model training is empty.")

    # Sort data by date to ensure correct splitting
    df_ts = df_ts.sort_values(by='ds').reset_index(drop=True)

    # Determine split point for training and testing
    # Calculate the date 'test_size_months' months ago from the last date in the dataset
    split_date = df_ts['ds'].max() - pd.DateOffset(months=test_size_months)
    
    train_df = df_ts[df_ts['ds'] <= split_date]
    test_df = df_ts[df_ts['ds'] > split_date]

    if train_df.empty:
        raise ValueError("Training DataFrame is empty. Not enough historical data for the specified test size.")
    if test_df.empty:
        print("Warning: Test DataFrame is empty. Evaluation metrics will not be calculated.")

    # Define custom holidays
    holidays = pd.DataFrame({
        'holiday': 'shopping_event',
        'ds': pd.to_datetime([
            '2019-11-29', '2019-12-02', '2019-12-25',  # 2019 dates
            '2020-11-27', '2020-11-30', '2020-12-25',  # 2020 dates (if applicable)
        ]),
        'lower_window': 0,
        'upper_window': 1,
    })

    # Initialize Prophet model
    # Using seasonality_mode='multiplicative' as recommended for e-commerce sales
    # Pass holidays dataframe here to include custom events like Black Friday
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Daily seasonality might be too granular or not present in daily aggregated data
        holidays=holidays if not holidays.empty else None
    )

    if not holidays.empty:
        model.add_country_holidays(country_name='US') # Add standard US holidays as well

    model.fit(train_df)

    # Create future DataFrame for forecasting
    future = model.make_future_dataframe(periods=periods_to_forecast, include_history=True)
    
    # Generate forecast for the entire period (history + future)
    forecast = model.predict(future)

    # Evaluate model if test_df is not empty
    metrics = {}
    if not test_df.empty:
        # Filter the forecast to only include the dates present in the test_df
        # This ensures we only compare actuals with forecasts for the test period
        df_comparison = pd.merge(test_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
        
        if not df_comparison.empty:
            y_true = df_comparison['y']
            y_pred = df_comparison['yhat']

            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            metrics = {
                "MAPE": mape,
                "RMSE": rmse
            }
        else:
            print("Warning: No overlapping dates between test data and forecast for evaluation. This might indicate an issue with data splitting or forecast range.")
            metrics = {"MAPE": None, "RMSE": None}
    else:
        metrics = {"MAPE": None, "RMSE": None}

    return model, forecast, metrics, test_df

if __name__ == '__main__':
    # Example usage:
    current_dir = os.path.dirname(__file__)
    csv_data_path = os.path.join(current_dir, '..', 'CSV') # Path to actual CSVs
    
    try:
        print("Running ETL pipeline...")
        processed_df = run_etl_pipeline(csv_data_path)
        print("ETL pipeline completed.")

        print("\nTraining and evaluating model...")
        # Use a smaller periods_to_forecast for quick testing if needed, e.g., 30
        # Use a smaller test_size_months if the dataset is small, e.g., 1
        model, forecast, metrics, test_df = train_and_evaluate_model(processed_df, periods_to_forecast=90, test_size_months=3)
        print("Model training and evaluation completed.")

        print("\nForecast Head:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
        print("\nForecast Tail:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        print("\nEvaluation Metrics:")
        print(json.dumps(metrics, indent=4))

        # You can save the model for later use
        # from prophet.serialize import model_to_json, model_from_json
        # with open('serialized_model.json', 'w') as fout:
        #     fout.write(model_to_json(model))
        # print("\nModel saved to serialized_model.json")

    except ValueError as e:
        print(f"Error during model pipeline: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")