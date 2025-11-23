# processing/rate_calculator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import zipfile
from scipy.interpolate import CubicSpline
from typing import Optional

class RateCalculator:
    """Handles rate calculations with caching of current spline"""
    
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self._current_date = None
        self._current_spline = None
        
    def get_rate_spline(self, date: datetime) -> Optional[CubicSpline]:
        """Get rate spline, using cached version if available"""
        if self._current_date == date and self._current_spline is not None:
            return self._current_spline
            
        rates_data = self._extract_rates_data(date)
        if rates_data is None:
            return None
            
        self._current_date = date
        self._current_spline = self._calculate_spline(rates_data)
        return self._current_spline
        
    def calculate_rates(self, days: pd.Series) -> pd.Series:
        if self._current_spline is None:
            raise ValueError("No rate spline loaded. Call get_rate_spline first.")
        # Evaluate spline on the entire array at once
        return pd.Series(self._current_spline(days.values), index=days.index)

    def _extract_rates_data(self, date: datetime) -> Optional[pd.DataFrame]:
        """Extract rates data from zip files"""
        current_date = date
        monthly_data = None
        
        zip_file_name = f"GI.NA.IVYZEROCD_{current_date.strftime('%Y%m')}.zip"
        zip_file_path = os.path.join(self.data_directory, zip_file_name)
        
        if os.path.exists(zip_file_path):
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    for file_in_zip in zip_ref.namelist():
                        if file_in_zip.endswith('.txt'):
                            with zip_ref.open(file_in_zip) as f:
                                monthly_data = pd.read_csv(f, delimiter='\t')
                                monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], format='%Y%m%d')
                                monthly_data = monthly_data[monthly_data['Currency'] == 333]
                                
                                daily_data = monthly_data[monthly_data['Date'] == current_date]
                                if not daily_data.empty:
                                    return daily_data
                                
                                valid_dates = monthly_data[monthly_data['Date'] <= current_date]['Date']
                                if not valid_dates.empty:
                                    most_recent = valid_dates.max()
                                    return monthly_data[monthly_data['Date'] == most_recent]
            except Exception as e:
                print(f"Error reading rates zip file for {current_date}: {e}")
        
        while current_date > datetime(1996, 1, 1):
            current_date -= timedelta(days=current_date.day)
            zip_file_name = f"GI.NA.IVYZEROCD_{current_date.strftime('%Y%m')}.zip"
            zip_file_path = os.path.join(self.data_directory, zip_file_name)
            
            if os.path.exists(zip_file_path):
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for file_in_zip in zip_ref.namelist():
                            if file_in_zip.endswith('.txt'):
                                with zip_ref.open(file_in_zip) as f:
                                    monthly_data = pd.read_csv(f, delimiter='\t')
                                    monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], format='%Y%m%d')
                                    monthly_data = monthly_data[monthly_data['Currency'] == 333]
                                    
                                    valid_dates = monthly_data['Date']
                                    if not valid_dates.empty:
                                        most_recent = valid_dates.max()
                                        return monthly_data[monthly_data['Date'] == most_recent]
                except Exception as e:
                    print(f"Error reading rates zip file for {current_date}: {e}")
        
        return None

    def _calculate_spline(self, rates_data: pd.DataFrame) -> CubicSpline:
        """Calculate spline from rates data"""
        x = rates_data['Days'].values
        y = rates_data['Rate'].values
        
        idx_sorted = np.argsort(x)
        x = x[idx_sorted]
        y = y[idx_sorted]
        
        return CubicSpline(x, y, bc_type='natural')