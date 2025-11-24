from fastapi import FastAPI, HTTPException, Response, Path, File, UploadFile, BackgroundTasks, Depends
from typing import List
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg') # Must be called before import pyplot as plt

import matplotlib.pyplot as plt
import base64
import os
import json
import shutil
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import ETL and Model Training functions
from .etl_pipeline import run_etl_pipeline
from .model_training import train_and_evaluate_model

# Caching and Logging
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from loguru import logger

# Configure Loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Initialize FastAPI app
app = FastAPI(
    title="Sales Prediction API",
    description="API for predicting sales of technological equipment and providing analytical insights.",
    version="1.1.0"
)

# CORS Middleware Configuration
# CORS Middleware Configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = allowed_origins_env.split(",")
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*" # Allow all by default for easier deployment, restrict in production via env var
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables to hold processed data and trained model
# These will be loaded on startup
# Global variables to hold processed data and trained model
# These will be loaded on startup
processed_data_df: pd.DataFrame = pd.DataFrame()
trained_prophet_model = None
# Placeholder for the full historical data (before aggregation) for bestsellers analysis
full_historical_df: pd.DataFrame = pd.DataFrame() 

# Processing Status
is_processing: bool = False
last_processed_time: str = None
processing_error: str = None 

# Define Pydantic models for request/response
class PredictionResponse(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float

class ForecastResponse(BaseModel):
    forecast: list[PredictionResponse]
    metrics: dict | None = None

class Bestseller(BaseModel):
    product: str
    total_sales_revenue: float

class BestsellersResponse(BaseModel):
    bestsellers: list[Bestseller]

class MetricsResponse(BaseModel):
    total_revenue: float
    total_orders: int
    total_products_sold: int
    average_order_value: float

class ChartBase64Response(BaseModel):
    image_base64: str
    media_type: str = "image/png"
    metrics: dict | None = None

class MonthlyGrowth(BaseModel):
    month: str
    revenue: float
    growth_pct: float

class CategorySales(BaseModel):
    category: str
    revenue: float

class CitySales(BaseModel):
    city: str
    revenue: float

class HourlySales(BaseModel):
    hour: int
    orders: int

class ProductPair(BaseModel):
    product1: str
    product2: str
    frequency: int

class ProductPairsResponse(BaseModel):
    pairs: List[ProductPair]

@app.on_event("startup")
async def startup_event():
    """
    Load data, run ETL, train model, and initialize cache on startup.
    """
    global processed_data_df, trained_prophet_model, full_historical_df
    
    # Initialize In-Memory Cache
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logger.info("In-Memory Cache initialized.")

    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'CSV')

    try:
        logger.info("API Startup: Running ETL pipeline...")
        processed_data_df, full_historical_df = run_etl_pipeline(data_path)
        logger.info("API Startup: ETL pipeline completed.")

        logger.info("API Startup: Training model...")
        # Train model with a reasonable forecast period and test size
        trained_prophet_model, _, _, _ = train_and_evaluate_model(processed_data_df, periods_to_forecast=90, test_size_months=3)
        logger.info("API Startup: Model training completed.")

    except Exception as e:
        logger.error(f"API Startup Error: Failed to load data or train model: {e}")
        # Depending on criticality, you might want to exit or set a flag
        # indicating the API is not fully functional.
        # raise HTTPException(status_code=500, detail=f"Failed to initialize API: {e}")

@app.get("/predict/sales/{days}", response_model=ForecastResponse)
@cache(expire=3600)
async def predict_sales(days: int = Path(..., gt=0, description="Number of days to forecast")):
    """
    Predicts sales for the next 'days' using the trained Prophet model.
    """
    if trained_prophet_model is None or processed_data_df.empty:
        raise HTTPException(status_code=503, detail="Model not loaded or data not processed yet.")

    logger.info(f"Generating forecast for {days} days.")
    # Create future dataframe for prediction
    future = trained_prophet_model.make_future_dataframe(periods=days, include_history=False)
    forecast = trained_prophet_model.predict(future)

    # Extract relevant forecast columns
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    
    # Convert datetime objects to string for JSON serialization
    for p in predictions:
        p['ds'] = p['ds'].strftime('%Y-%m-%d')

    return ForecastResponse(forecast=predictions)

@app.get("/analysis/bestsellers/{top_n}", response_model=BestsellersResponse)
@cache(expire=3600)
async def get_bestsellers(top_n: int = Path(..., gt=0, description="Number of top-selling products to retrieve")):
    """
    Identifies the top N best-selling products based on historical sales revenue.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet for analysis.")

    logger.info(f"Retrieving top {top_n} bestsellers.")
    bestsellers_df = full_historical_df.groupby('Product')['Sales Revenue'].sum().nlargest(top_n).reset_index()
    bestsellers = bestsellers_df.rename(columns={'Product': 'product', 'Sales Revenue': 'total_sales_revenue'}).to_dict(orient='records')

    return BestsellersResponse(bestsellers=bestsellers)


@app.get("/metrics", response_model=MetricsResponse)
@cache(expire=3600)
async def get_metrics():
    """
    Returns key performance indicators (KPIs) from the processed data.
    """
    if processed_data_df.empty:
        raise HTTPException(status_code=503, detail="Data not processed yet.")

    total_revenue = float(processed_data_df['y'].sum())
    total_orders = int(full_historical_df['Order ID'].nunique())
    total_products_sold = int(full_historical_df['Quantity Ordered'].sum())
    average_order_value = total_revenue / total_orders if total_orders > 0 else 0.0

    logger.info(f"DEBUG: Computing metrics. Total Revenue: {total_revenue}")

    return MetricsResponse(
        total_revenue=total_revenue,
        total_orders=total_orders,
        total_products_sold=total_products_sold,
        average_order_value=average_order_value,
        last_updated=datetime.now().isoformat()
    )

@app.get("/analysis/monthly_growth", response_model=List[MonthlyGrowth])
@cache(expire=3600)
async def get_monthly_growth():
    """
    Calculates month-over-month sales growth.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet.")

    logger.info("Calculating monthly growth.")
    df = full_historical_df.copy()
    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Sales Revenue'].sum().reset_index()
    monthly_sales['Growth_Pct'] = monthly_sales['Sales Revenue'].pct_change() * 100
    monthly_sales['Growth_Pct'] = monthly_sales['Growth_Pct'].fillna(0)
    
    result = []
    for _, row in monthly_sales.iterrows():
        result.append({
            "month": str(row['Month']),
            "revenue": float(row['Sales Revenue']),
            "growth_pct": float(row['Growth_Pct'])
        })
    return result

@app.get("/analysis/category_sales", response_model=List[CategorySales])
@cache(expire=3600)
async def get_category_sales():
    """
    Calculates sales by product category.
    Since we don't have an explicit category column, we'll infer it or just use Product for now if categories aren't defined.
    For this dataset (tech products), we can try to group by the first word or known categories.
    Let's assume 'Product' is granular enough, or we can group by a simple heuristic.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet.")

    logger.info("Calculating category sales.")
    # Simple heuristic: Group by first word of product (e.g., "iPhone", "Macbook", "Google", "Wired")
    # Or just return top products as categories if they are distinct enough.
    # Let's try to map some common items to categories for better visualization.
    
    def get_category(product_name):
        product_name = str(product_name).lower()
        if 'iphone' in product_name: return 'Phones'
        if 'samsung' in product_name or 'phone' in product_name: return 'Phones'
        if 'macbook' in product_name or 'laptop' in product_name: return 'Laptops'
        if 'monitor' in product_name: return 'Monitors'
        if 'headphone' in product_name or 'airpods' in product_name or 'soundsport' in product_name: return 'Audio'
        if 'tv' in product_name: return 'TVs'
        if 'cable' in product_name: return 'Accessories'
        if 'batteries' in product_name: return 'Consumables'
        return 'Other'

    df = full_historical_df.copy()
    df['Category'] = df['Product'].apply(get_category)
    
    category_sales = df.groupby('Category')['Sales Revenue'].sum().reset_index()
    
    return [
        {"category": row['Category'], "revenue": row['Sales Revenue']}
        for _, row in category_sales.iterrows()
    ]

@app.get("/analysis/sales_by_city", response_model=List[CitySales])
@cache(expire=3600)
async def get_sales_by_city():
    """
    Calculates sales revenue by city.
    Extracts city from 'Purchase Address'.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet.")

    logger.info("Calculating sales by city.")
    df = full_historical_df.copy()
    
    # Helper to extract city and state
    def get_city(address):
        return address.split(',')[1].strip() + ' ' + address.split(',')[2].split(' ')[1]

    df['City'] = df['Purchase Address'].apply(lambda x: f"{get_city(x)}")
    city_sales = df.groupby('City')['Sales Revenue'].sum().reset_index()
    
    return [
        {"city": str(row['City']), "revenue": float(row['Sales Revenue'])}
        for _, row in city_sales.iterrows()
    ]

@app.get("/analysis/hourly_sales", response_model=List[HourlySales])
@cache(expire=3600)
async def get_hourly_sales():
    """
    Calculates number of orders by hour of the day.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet.")

    logger.info("Calculating hourly sales.")
    df = full_historical_df.copy()
    df['Hour'] = df['Order Date'].dt.hour
    hourly_counts = df.groupby('Hour')['Order ID'].nunique().reset_index()
    
    return [
        {"hour": int(row['Hour']), "orders": int(row['Order ID'])}
        for _, row in hourly_counts.iterrows()
    ]

@app.get("/analysis/product_pairs", response_model=ProductPairsResponse)
@cache(expire=3600)
async def get_product_pairs():
    """
    Identifies top 5 pairs of products frequently bought together.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="Historical data not loaded yet.")

    logger.info("Calculating product pairs.")
    from itertools import combinations
    from collections import Counter

    df = full_historical_df[full_historical_df['Order ID'].duplicated(keep=False)]
    df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
    df = df[['Order ID', 'Grouped']].drop_duplicates()

    count = Counter()
    for row in df['Grouped']:
        row_list = row.split(',')
        count.update(Counter(combinations(row_list, 2)))

    top_pairs = count.most_common(5)
    
    result = []
    for pair, frequency in top_pairs:
        result.append({
            "product1": pair[0],
            "product2": pair[1],
            "frequency": frequency
        })
    
    return ProductPairsResponse(pairs=result)

@app.get("/chart/forecast/{days}", response_class=Response)
async def get_forecast_chart(days: int = Path(..., description="Number of days to forecast for the chart")):
    """
    Generates and returns a PNG image of the sales forecast chart.
    """
    if trained_prophet_model is None or processed_data_df.empty:
        raise HTTPException(status_code=503, detail="Model not loaded or data not processed yet.")

    logger.info(f"Generating forecast chart for {days} days.")
    future = trained_prophet_model.make_future_dataframe(periods=days, include_history=True)
    forecast = trained_prophet_model.predict(future)

    fig, ax = plt.subplots(figsize=(10, 6))
    trained_prophet_model.plot(forecast, ax=ax)
    ax.set_title('Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Revenue')
    fig.canvas.draw() # Explicitly draw the canvas

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    plt.close(fig) # Close the figure to free memory
    img_buf.seek(0)
    return StreamingResponse(img_buf, media_type="image/png")

@app.get("/chart/forecast_base64", response_model=ChartBase64Response)
@cache(expire=3600)
async def get_forecast_chart_base64(days: int = 90): # Default to 90 days if not specified
    """
    Generates a sales forecast chart and returns it as a Base64 encoded string.
    """
    if trained_prophet_model is None or processed_data_df.empty:
        raise HTTPException(status_code=503, detail="Model not loaded or data not processed yet.")

    if days <= 0:
        raise HTTPException(status_code=400, detail="Days must be a positive integer.")

    logger.info(f"Generating base64 forecast chart for {days} days.")
    future = trained_prophet_model.make_future_dataframe(periods=days, include_history=True)
    forecast = trained_prophet_model.predict(future)

    fig, ax = plt.subplots(figsize=(10, 6))
    trained_prophet_model.plot(forecast, ax=ax)
    ax.set_title('Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales Revenue')
    fig.canvas.draw() # Explicitly draw the canvas

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    plt.close(fig) # Close the figure to free memory
    img_buf.seek(0)

    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')

    return ChartBase64Response(image_base64=img_base64, media_type="image/png")

async def reload_data_and_model():
    """
    Asynchronously re-runs the ETL pipeline and retrains the model.
    This function is intended to be called after new data is uploaded.
    """
    global processed_data_df, trained_prophet_model, full_historical_df, is_processing, last_processed_time, processing_error
    
    is_processing = True
    processing_error = None
    
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'CSV')

    try:
        logger.info("Data Reload: Running ETL pipeline...")
        processed_data_df, full_historical_df = run_etl_pipeline(data_path)
        logger.info("Data Reload: ETL pipeline completed.")

        logger.info("Data Reload: Retraining model...")
        trained_prophet_model, _, _, _ = train_and_evaluate_model(processed_data_df, periods_to_forecast=90, test_size_months=3)
        logger.info("Data Reload: Model retraining completed.")
        
        # Clear cache after reload
        await FastAPICache.clear()
        logger.info("Cache cleared.")
        
        last_processed_time = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Data Reload Error: {e}")
        processing_error = str(e)
    finally:
        is_processing = False

@app.get("/status")
async def get_processing_status():
    """
    Returns the current status of data processing.
    """
    return {
        "processing": is_processing,
        "last_updated": last_processed_time,
        "error": processing_error
    }

@app.post("/upload/")
async def upload_data_files(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Uploads new data files (CSV or JSON), saves them, and triggers a data reload and model retraining.
    Returns immediately and runs processing in background.
    """
    # Ensure the data directory exists
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'CSV')
    os.makedirs(data_path, exist_ok=True)

    filenames = []
    for file in files:
        file_path = os.path.join(data_path, file.filename)
        filenames.append(file.filename)

        # Basic validation for file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.json')):
             raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only CSV and JSON are allowed.")

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}: {e}")
        finally:
            file.file.close()

    logger.info(f"Files {filenames} uploaded. Starting background reload...")
    
    # Reload data and model in background
    background_tasks.add_task(reload_data_and_model)

    return {"message": f"Successfully uploaded {len(files)} files. Processing started in background."}

@app.get("/export/", response_class=StreamingResponse)
async def export_data():
    """
    Exports the full historical data to an XLSX file.
    """
    if full_historical_df.empty:
        raise HTTPException(status_code=503, detail="No data available to export.")

    logger.info("Exporting data to Excel.")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        full_historical_df.to_excel(writer, index=False, sheet_name='Sales_Data')
    output.seek(0)

    headers = {
        'Content-Disposition': 'attachment; filename="sales_report.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)