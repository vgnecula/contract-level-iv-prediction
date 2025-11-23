# processing/dataset_entries_processing.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def get_trading_files(options_dir, start_date: datetime, end_date: datetime) -> List[str]:
    """
    Retrieves the list of trading day files within a specified date range.

    This function scans the options directory, extracts file names corresponding to 
    valid trading days within the given period, and returns a sorted list of filenames.

    Args:
        options_dir (str): The directory containing daily options CSV files.
        start_date (datetime): The start date of the desired trading period.
        end_date (datetime): The end date of the desired trading period.

    Returns:
        List[str]: A list of filenames representing valid trading days within the date range.

    """ 
    all_files = sorted([f for f in os.listdir(options_dir) if f.endswith('.csv')])
    
    trading_files = []
    for f in all_files:
        try:
            file_date = datetime.strptime(f.replace('.csv', ''), '%Y-%m-%d')
            if start_date <= file_date <= end_date:
                trading_files.append(f)
        except ValueError:
            continue
    
    all_files_count = len(all_files)
    trading_files_count = len(trading_files)

    print(f"\nTotal number of trading days: {all_files_count}")
    print(f"Total number of trading days in the specified period: {trading_files_count}")
    return trading_files
        
def get_valid_contracts(options_dir: str, window_files: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Identifies option contracts that are consistently present throughout a given time window.

    This function:
    - Starts with the contracts available on the first day in `window_files`.
    - Filters out contracts missing from any subsequent days in the window.
    - Ensures data completeness by checking for missing values and placeholder values (`-99.99`).

    Args:
        options_dir (str): The directory containing daily options data.
        window_files (List[str]): A list of filenames representing a rolling window of trading days.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping valid contract IDs to their corresponding
        time-series DataFrame containing market features.

    """  
    # Initial Filtering -------------------------------------------------------
    
    # Dictionary, key-> contract id, value-> its temporal sequence
    initial_contract_data = {}

    # First, select all the contracts that are in the first file
    first_file = window_files[0]
    first_day_data = pd.read_csv(os.path.join(options_dir, first_file))
    first_day_contracts = set(first_day_data[first_day_data['DaysToExpiration'] <= 30]['OptionID'])
    for _, row in first_day_data.iterrows(): 
        initial_contract_data[row['OptionID']] = [row] # Adding all the day 1 contracts to the list

    # Iterate over all days, starting second
    for current_file_name in window_files[1:]:
        current_file_data = pd.read_csv(os.path.join(options_dir, current_file_name))

        # Filter out all contracts that were not present in day 1
        current_file_data = current_file_data[current_file_data['OptionID'].isin(first_day_contracts)]

        # Now go through all remaining contracts and add them to sequence corresponding to their ContractID 
        for _, row in current_file_data.iterrows():
            contract_id = row['OptionID'] # get the id
            initial_contract_data[contract_id].append(row) # append the new row   

    # Further Filtering -------------------------------------------------------
    # Now, we have all the contracts that are present on Day1, however not all available for all the remaining days
    # We need to do further filtering: present on all days + our chosen filters
    valid_contracts = {}
    required_days = len(window_files)

    for contract_id, data in initial_contract_data.items():

        if(len(data) == required_days): #first check
            # Transforming the sequence into a dataframe
            contract_df = pd.DataFrame(data)
            contract_df['Date'] = pd.to_datetime(contract_df['Date'])
            contract_df = contract_df.sort_values('Date').reset_index(drop=True) # Sort again to be sure everything is good

            has_nans = any(
                pd.isna(value) 
                for row in data 
                for value in row.values
            )
            
            if not has_nans:
                if (contract_df['Date'].nunique() == required_days and
                        not ((contract_df['ImpliedVolatility'] == -99.99).any() or
                            (contract_df['Delta'] == -99.99).any() or
                            (contract_df['Gamma'] == -99.99).any() or
                            (contract_df['Vega'] == -99.99).any())):
                    valid_contracts[contract_id] = contract_df

    print(f"Found {len(valid_contracts)} contracts with complete data for all {required_days} trading days")
    return valid_contracts

def process_single_dataset_entry(valid_contracts: Dict[str, pd.DataFrame], 
                                 current_start: datetime, 
                                 current_end: datetime) -> Dict[str, np.ndarray]:
    """
    Processes a single dataset entry by generating sequences and target labels.

    This function:
    - Extracts time-series features for each contract.
    - Removes unnecessary columns such as `OptionID` and `Date`.
    - Computes the target variables - direction or magnitude.
    - Formats sequences and targets into NumPy arrays.

    Args:
        valid_contracts (Dict[str, pd.DataFrame]): A dictionary of contract data with complete time-series.
        current_start (datetime): The starting date of the batch window.
        current_end (datetime): The ending date of the batch window.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - 'sequences':       (num_contracts, seq_len-1, num_features)
            - 'targets_dir':     (num_contracts,)  direction (0 = down, 1 = up)
            - 'targets_mag':     (num_contracts,)  |ΔIV|
            - 'log_moneyness_ref':  (num_contracts,) log moneyness reference (for bucket computation later)
            - 'contract_ids':    list of contract IDs
            - 'feature_columns': list of feature column names
            - 'start_date':      batch start date
            - 'end_date':        batch end date

    """
    sequences = []
    targets_dir = []
    targets_mag = []
    contract_ids = []
    log_moneyness_ref_list = []

    feature_columns = None

    # Separate the training sequence and the target
    for contract_id, contract_df in valid_contracts.items():
        
        # Drop date columns and ID (we dont want them in the input features)
        contract_df = contract_df.drop(['OptionID', 'Date', 'Expiration'], axis = 1)

        if feature_columns is None:
            feature_columns = contract_df.columns.to_numpy()
        else:
            # Sanity check: all contracts should have same columns in same order
            if not np.array_equal(feature_columns, contract_df.columns.to_numpy()):
                raise ValueError(f"Inconsistent feature columns for contract {contract_id}")


        # Calculate Change in IV Direction JUST between last two days ()
        last_day_iv_change = contract_df['ImpliedVolatility'].iloc[-1] - contract_df['ImpliedVolatility'].iloc[-2] 
       
        # direction
        dir_label = 1 if last_day_iv_change >= 0 else 0

        # magnitudes
        delta = float(last_day_iv_change)
        mag = abs(delta)

        # - LogMoneyness reference -
        # Use penultimate day (the last *input* day) as reference.
        log_m_ref = float(contract_df['LogMoneyness'].iloc[-2])
        log_moneyness_ref_list.append(log_m_ref)


        # IMPORTANT: Get rid of the last day from the sequence (for no leakage)
        sequence = contract_df[:-1].to_numpy(dtype=np.float32) 


        # APPEND TO THE LISTs
        sequences.append(sequence)
        targets_dir.append(dir_label)
        targets_mag.append(mag)
        contract_ids.append(contract_id)

    return {
        'sequences':       np.stack(sequences),
        'targets_dir':     np.array(targets_dir, dtype=np.float32),
        'targets_mag':     np.array(targets_mag, dtype=np.float32),
        'log_moneyness_ref':  np.array(log_moneyness_ref_list, dtype=np.float32),
        'contract_ids':    contract_ids,
        'feature_columns': feature_columns,
        'start_date':      current_start,
        'end_date':        current_end
    }


def process_and_save_dataset_entries(options_dir: str, 
                                     dataset_entries_dir: str, 
                                     start_date: datetime, 
                                     end_date: datetime, 
                                     window_size: int, 
                                     stride: int) -> None:
    """
    Processes options data by iterating through rolling windows and saves structured dataset entries.

    This function:
    - Iterates over the trading files within the date range using a rolling window.
    - Identifies valid option contracts for each batch.
    - Extracts time-series features and target variables.
    - Saves processed dataset entries as `.npz` files.

    Args:
        options_dir (str): Directory containing daily options CSV files.
        dataset_entries_dir (str): Directory where processed dataset entries will be saved.
        start_date (datetime): Start date for processing.
        end_date (datetime): End date for processing.
        window_size (int): Number of trading days included in each batch.
        stride (int): Step size for shifting the rolling window.

    Returns:
        None

    """   
    trading_files = get_trading_files(options_dir, start_date, end_date) 

    current_entry = 0

    for i in range(0, len(trading_files) - window_size + 1, stride):
        
        window_files = trading_files[i : i+window_size]
        
        if len(window_files) < window_size:
            break
        
        current_start = datetime.strptime(window_files[0].split('.')[0], '%Y-%m-%d')
        current_end = datetime.strptime(window_files[-1].split('.')[0], '%Y-%m-%d')
            
        print(f"\nProcessing batch period: {current_start.date()} to {current_end.date()}")
            
        valid_contracts = get_valid_contracts(options_dir, window_files) # this returns a dictionary {key: Contract_ID, value: pd.Dataframe (Date-Features)}

        if valid_contracts:
            entry = process_single_dataset_entry(valid_contracts, current_start, current_end)

            if entry is not None:
                entry_save_path = os.path.join(dataset_entries_dir, f"batch_{current_entry:04d}.npz")
                np.savez(
                    entry_save_path,
                    sequences       = entry['sequences'],
                    targets_dir     = entry['targets_dir'],
                    targets_mag     = entry['targets_mag'],
                    log_moneyness_ref = entry['log_moneyness_ref'],
                    contract_ids    = entry['contract_ids'],
                    feature_columns = entry['feature_columns'],
                    start_date      = entry['start_date'],
                    end_date        = entry['end_date']
                )
                print(f"Saved batch {current_entry} with {len(entry['contract_ids'])} contracts")
                current_entry += 1

def main() -> None:

    script_dir = os.path.dirname(os.path.abspath(__file__))
    options_dir = os.path.join(script_dir, "../data/Processed_Options_Data")
    dataset_entries_dir = os.path.join(script_dir, "../data/dataset_entries/")
    os.makedirs(dataset_entries_dir, exist_ok=True)

    start_date = datetime(1996, 1, 4)
    end_date = datetime(2023, 12, 31)

    window_size = 10
    stride = 5

    process_and_save_dataset_entries(options_dir, dataset_entries_dir, start_date, end_date, window_size, stride)

if __name__ == "__main__":
    main()