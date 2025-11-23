# processing/options_processing.py

"""
This script extracts the daily option contracts out of the raw files provided within the OptionMetrics Dataset 
and saves them in daily csv files within the data directory.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import zipfile
from scipy.interpolate import CubicSpline
from typing import Optional, Dict, Tuple
import yfinance as yf

from rate_calculator import RateCalculator

REQUIRED_COLUMNS = {
    # Identification columns
    'id_columns': [
        'Date',
        'OptionID',
        'Strike',
        'DaysToExpiration',
        'Expiration'
    ],
    # Market data and Greeks
    'option_traits': [
        'ImpliedVolatility',
        'Delta',
        'Gamma',
        'Vega',
        'OpenInterest',
        'Volume'
    ],
    # Additional calculated/market columns
    'market_columns': [
        'SPX',
        'LogMoneyness',
        'RiskFreeRate',
        'ForwardPrice'
    ]
}
ALL_COLUMNS = (REQUIRED_COLUMNS['id_columns'] + 
               REQUIRED_COLUMNS['option_traits'] + 
               REQUIRED_COLUMNS['market_columns'])

def extract_options_data(raw_data_dir: str, date: datetime) -> Optional[pd.DataFrame]:
    """
    Extracts options data for a specific date from compressed zip files.

    This function searches for a zip file containing option contracts for the month of `date`,
    extracts the relevant file, and filters data for the exact date.

    Args:
        raw_data_dir (str): The directory containing raw options data in zip files.
        date (datetime): The date for which the options data should be extracted.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing options data for the specified date,
        or None if no relevant data is found.
    """
    zip_file_name = f"GI.NA.IVYOPPRCD_{date.strftime('%Y%m')}.zip" # taking the file corresponding to the month of our date
    zip_file_path = os.path.join(raw_data_dir, zip_file_name) # builds the full path
    
    if os.path.exists(zip_file_path):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for file_in_zip in zip_ref.namelist():
                    if file_in_zip.endswith('.txt'):
                        with zip_ref.open(file_in_zip) as f:
                            monthly_data = pd.read_csv(f, delimiter='\t')
                            monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], format='%Y%m%d')
                            monthly_data['Expiration'] = pd.to_datetime(monthly_data['Expiration'], format='%Y%m%d')
                            daily_data = monthly_data[monthly_data['Date'] == date]
                            
                            if not daily_data.empty:
                                return daily_data
                                
        except Exception as e:
            print(f"Error reading options zip file for {date}: {e}")
    
    return None


def process_and_save_individual_date(raw_data_dir: str, processed_data_dir: str, current_date: datetime) -> None:
    """
    Processes option data for a single date and saves the cleaned data.

    This function retrieves raw options data, calculates additional financial variables,
    and saves the processed data in the specified directory.

    Args:
        raw_data_dir (str): The directory containing raw options data.
        processed_data_dir (str): The directory where processed data should be saved.
        current_date (datetime): The date for which options data is processed.

    Returns:
        None
    """
    raw_options = extract_options_data(raw_data_dir, current_date)

    if raw_options is None:
        print(f"No data found for date {current_date}")
        return None
    
    # Add DaysToExpiration ------------------------------------
    raw_options['DaysToExpiration'] = (raw_options['Expiration'] - raw_options['Date']).dt.days

    # Add Risk Free Rates ------------------------------------
    rate_calculator = RateCalculator(raw_data_dir)
    spline = rate_calculator.get_rate_spline(current_date)
    if spline is None:
        print(f"No rates data found for date {current_date}")
        return # it means we cant calculate the rates for this date
    raw_options['RiskFreeRate'] = rate_calculator.calculate_rates(raw_options['DaysToExpiration'])

    # Add SPX Price ------------------------------------
    spx_data = pd.read_csv('data/spx_data/spx.csv', index_col=0)
    spx_price = spx_data.loc[current_date.strftime('%Y-%m-%d'), 'Close'] if current_date.strftime('%Y-%m-%d') in spx_data.index else np.nan
    if np.isnan(spx_price):
        print(f"No SPX price for {current_date}")
        return
    raw_options['SPX'] = spx_price

    # Adjust Strike Price ------------------------------------
    raw_options['Strike'] = raw_options['Strike'] / 1000

    # Calculate forward price and LogMoneyness ------------------------------------
    T = raw_options['DaysToExpiration'] / 365.0
    r = raw_options['RiskFreeRate']
    
    forward_price = spx_price * np.exp(r * T)
    raw_options['ForwardPrice'] = forward_price
    
    raw_options['LogMoneyness'] = np.log(raw_options['Strike'] / forward_price)

    # Keep only the columns we want in the final dataset + non-NAN values --------------------------
    processed_options = raw_options[ALL_COLUMNS].copy()
    processed_options = processed_options.dropna()

    # Save processed data
    output_file = os.path.join(processed_data_dir, f"{current_date.strftime('%Y-%m-%d')}.csv")
    print(f"Saving processed data for {current_date} to {output_file}")
    processed_options.to_csv(output_file, index=False)
    
        
def process_and_save_interval(raw_data_dir: str, processed_data_dir: str, start_date: datetime, end_date: datetime) -> None:
    """
    Processes options data over a given date range.

    Iterates through each date in the given interval and processes the options data
    for each day, saving the results in `processed_data_dir`.

    Args:
        raw_data_dir (str): The directory containing raw options data.
        processed_data_dir (str): The directory where processed data should be saved.
        start_date (datetime): The starting date for processing.
        end_date (datetime): The ending date for processing.

    Returns:
        None
    """
    current_date = start_date
    while current_date <= end_date:
        
        print(f"\nProcessing date: {current_date}")
        process_and_save_individual_date(raw_data_dir, processed_data_dir, current_date)
        current_date += timedelta(days=1)



def main() -> None:
    """
    Main function to initiate the options data processing pipeline.

    This function sets up file paths, ensures necessary directories exist,
    and processes options data over the defined date range.

    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, "../data/OptionMetrics_Raw_Data/IvyDBGI.NA.SPX.1996_01_01-2023_12_31")
    processed_data_dir = os.path.join(script_dir, "../data/Processed_Options_Data")
    os.makedirs(processed_data_dir, exist_ok=True)

    start_date = datetime(1996, 1, 4)
    end_date = datetime(2023, 12, 31)

    process_and_save_interval(raw_data_dir, processed_data_dir, start_date, end_date)

if __name__ == "__main__":
    main()


