import pandas as pd
import glob
import os

def run_etl_pipeline(data_folder_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the ETL pipeline to process raw sales data from CSV and JSON files.

    Args:
        data_folder_path (str): The absolute path to the folder containing the data files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The processed time series data with 'ds' and 'y' columns.
            - The consolidated, cleaned, but non-aggregated full historical data.
    """
    # 1. Ingesta Masiva y Consolidación de Archivos
    csv_files = glob.glob(os.path.join(data_folder_path, "*.csv"))
    json_files = glob.glob(os.path.join(data_folder_path, "*.json"))
    all_files = csv_files + json_files
    
    print(f"DEBUG: ETL Pipeline found {len(all_files)} files in {data_folder_path}")
    for f in all_files:
        print(f"DEBUG: Found file: {os.path.basename(f)}")

    if not all_files:
        raise ValueError(f"No CSV or JSON files found in the specified folder: {data_folder_path}")

    df_list = []
    for f in all_files:
        if f.endswith('.csv'):
            df = pd.read_csv(f)
        elif f.endswith('.json'):
            # Assuming JSON is in a format that can be directly read into a DataFrame
            # e.g., an array of objects.
            df = pd.read_json(f)
        else:
            # Skip other files if any
            continue
        df_list.append(df)

    df_consolidated = pd.concat(df_list, ignore_index=True)
    print(f"DEBUG: Total rows after concatenation: {len(df_consolidated)}")

    # 2. Limpieza y Normalización de Datos
    # Remove rows that are just headers repeated in the data
    df_consolidated = df_consolidated[df_consolidated['Order Date'] != 'Order Date']
    
    # Drop rows with any NaN values that might interfere with conversion
    df_consolidated.dropna(how='all', inplace=True)

    # Convert 'Order Date' to datetime
    df_consolidated['Order Date'] = pd.to_datetime(df_consolidated['Order Date'], errors='coerce')
    df_consolidated.dropna(subset=['Order Date'], inplace=True) # Drop rows where date conversion failed

    # Clean and convert 'Price Each' and 'Quantity Ordered' to numeric
    df_consolidated['Price Each'] = pd.to_numeric(df_consolidated['Price Each'], errors='coerce')
    df_consolidated['Quantity Ordered'] = pd.to_numeric(df_consolidated['Quantity Ordered'], errors='coerce')

    # Drop rows where numeric conversion failed
    df_consolidated.dropna(subset=['Price Each', 'Quantity Ordered'], inplace=True)

    # Ensure Quantity Ordered is integer (as it represents count)
    df_consolidated['Quantity Ordered'] = df_consolidated['Quantity Ordered'].astype(int)

    # 3. Feature Engineering y Agregación Temporal
    # Calculate Sales Revenue
    df_consolidated['Sales Revenue'] = df_consolidated['Quantity Ordered'] * df_consolidated['Price Each']

    # Aggregate daily sales revenue
    df_ts = df_consolidated.groupby('Order Date')['Sales Revenue'].sum().reset_index()
    df_ts.columns = ['ds', 'y']

    # Ensure continuity of the time series by filling missing dates with y=0
    # First, set 'ds' as index to use resample
    df_ts = df_ts.set_index('ds')
    # Resample to daily frequency and fill missing values with 0
    df_ts = df_ts.resample('D').sum().fillna(0).reset_index()

    return df_ts, df_consolidated

if __name__ == '__main__':
    # Example usage (assuming CSVs are in a 'CSV' folder relative to this script)
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'CSV')
    
    try:
        processed_df, full_df = run_etl_pipeline(data_path)
        print("ETL Pipeline completed successfully. Head of the processed DataFrame:")
        print(processed_df.head())
        print("\nTail of the processed DataFrame:")
        print(processed_df.tail())
        print(f"\nTotal rows in processed DataFrame: {len(processed_df)}")
        print("\nHead of the full (non-aggregated) DataFrame:")
        print(full_df.head())
    except ValueError as e:
        print(f"Error during ETL pipeline: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 