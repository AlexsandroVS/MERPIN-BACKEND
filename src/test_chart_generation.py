import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import io
import os
import numpy as np

def generate_dummy_forecast_data():
    """Generates dummy data for Prophet model and forecast."""
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=365 * 2, freq='D'))
    sales = np.random.rand(365 * 2) * 100 + np.sin(np.arange(365 * 2) / 30) * 50 + 100
    sales = sales + np.arange(365 * 2) * 0.1
    df_ts = pd.DataFrame({'ds': dates, 'y': sales})

    model = Prophet(seasonality_mode='multiplicative')
    model.fit(df_ts)

    future = model.make_future_dataframe(periods=30, include_history=True)
    forecast = model.predict(future)
    return model, forecast

def test_chart_generation():
    model, forecast = generate_dummy_forecast_data()

    fig = model.plot(forecast)
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    
    # Check if the buffer contains data
    if len(img_buf.getvalue()) > 1000: # A typical PNG image should be larger than 1KB
        print(f"Chart generated successfully. Size: {len(img_buf.getvalue())} bytes.")
        # Optionally save to a file to visually inspect
        with open("test_forecast_chart.png", "wb") as f:
            f.write(img_buf.getvalue())
        print("Saved test_forecast_chart.png for inspection.")
        return True
    else:
        print(f"Chart generation failed or produced a very small file. Size: {len(img_buf.getvalue())} bytes.")
        return False

if __name__ == "__main__":
    if test_chart_generation():
        print("Isolated chart generation test PASSED.")
    else:
        print("Isolated chart generation test FAILED.")
