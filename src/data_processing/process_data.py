import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import deque


HISTORICAL_PRICE_WINDOW = 3600
TYPE_DICT = {
    'Pump V1': 0,
    'Raydium V4': 1,
    'Moonshot': 2,
}


def convert_time_string_to_utc(time_str):
    date_str = time_str.split(' (')[0]
    dt = datetime.strptime(date_str, "%a %b %d %Y %H:%M:%S GMT%z")
    utc_timestamp = int(dt.timestamp())
    return utc_timestamp

class DataProcessor:
    def __init__(self, raw_data_dir, processed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir

    def get_transaction_files(self):
        transactions_path = os.path.join(self.raw_data_dir, 'transactions')
        return [os.path.join(transactions_path, f) for f in os.listdir(transactions_path) if f.endswith('.csv')]

    def get_matching_pairs_file(self, transaction_file):

        transaction_filename = os.path.basename(transaction_file)

        date_str = transaction_filename.split('_')[
            1]

        date = pd.to_datetime(date_str, format='%Y-%m-%d')

        pairs_path = os.path.join(self.raw_data_dir, 'pairs')
        all_pairs_files = [f for f in os.listdir(pairs_path) if f.endswith('.csv')]

        for pairs_file in all_pairs_files:
            pairs_file_date_str = pairs_file.replace('.csv', '').split('_')[1]
            pairs_file_date = pd.to_datetime(pairs_file_date_str, format='%Y-%m-%d')

            if pairs_file_date + pd.Timedelta(days=-7) <= date < pairs_file_date:
                return os.path.join(pairs_path, pairs_file)

        return None

    def process_file_pair(self, transaction_file, pairs_file):
        print(f"Processing transaction file: {transaction_file} with pairs file: {pairs_file}")

        transactions_df = pd.read_csv(transaction_file, on_bad_lines='skip', low_memory=False)
        pairs_df = pd.read_csv(pairs_file, on_bad_lines='skip', low_memory=False)

        transactions_df = self.clean_data(transactions_df)
        pairs_df = self.clean_data(pairs_df)
        date_str = os.path.basename(transaction_file).split('_')[1]
        self.organize_as_timeseries(transactions_df, pairs_df, date_str)

    def clean_data(self, df):
        df = df.drop_duplicates()
        df = df.fillna(method='ffill')
        return df



    def organize_as_timeseries(self, transactions_df, pairs_df, date_str):

        for pair_address, group in transactions_df.groupby('pair_address'):
            if pair_address in pairs_df['pair_address'].unique():
                pair_info = pairs_df[pairs_df['pair_address'] == pair_address]
                pair_type = pair_info['protocol'].values[0]
            else:
                pair_type = 'Unknown'

            pair_type = TYPE_DICT.get(pair_type, -1)

            if len(group) > 86400:

                output = {
                    'pair_address': [],
                    'time_stamp': [],
                    'current_price': [],
                    'historical_price': [],
                    'type': [],
                    'active_trades': [],  # TODO: Define this
                    'num_transactions': [],
                    'label': []
                }

                group['utc_timestamp'] = group['created_at'].apply(convert_time_string_to_utc)

                group = group.sort_values(by='utc_timestamp')
                start_time = group['utc_timestamp'].values[0]
                end_time = group['utc_timestamp'].values[-1]
                historical_prices = deque(maxlen=HISTORICAL_PRICE_WINDOW)

                group['price_usd'] = group['price_usd'].astype(np.float64)
                prev_price = group['price_usd'].values[0]

                for time_pt in range(start_time, end_time, 1):
                    if time_pt in group['utc_timestamp'].values:
                        time_window = group[group['utc_timestamp'] == time_pt]
                        avg_price = time_window['price_usd'].mean()
                        if abs(avg_price - prev_price) > 0.0000003:
                            if avg_price > prev_price:
                                label = 1
                            else:
                                label = 0
                        else:
                            label = 0.5
                        historical_prices.append(avg_price)
                        num_transactions = len(time_window)
                        prev_price = avg_price
                    else:
                        avg_price = historical_prices[-1]
                        historical_prices.append(avg_price)
                        num_transactions = 0
                        label = np.nan

                    if time_pt - start_time > HISTORICAL_PRICE_WINDOW:
                        output['pair_address'].append(pair_address)
                        output['time_stamp'].append(time_pt)
                        output['current_price'].append(avg_price)
                        output['historical_price'].append(list(historical_prices))
                        output['type'].append(pair_type)
                        output['active_trades'].append(num_transactions)  # TODO: To be replaced with actual value
                        output['num_transactions'].append(num_transactions)
                        output['label'].append(label)

                time_series_data = pd.DataFrame(output)
                time_series_data['label'] = time_series_data['label'].fillna(method='bfill')
                time_series_data['label'] = time_series_data['label'].shift(-1)
                time_series_data = time_series_data.dropna()

                time_series_data.to_csv(f"{self.processed_data_dir}/time_series_{date_str}_{pair_address}.csv", index=False)
                print(f"Time series data saved for {pair_address} on {date_str}. Number of entries: {len(time_series_data)}")

    def preprocess(self):
        transaction_files = self.get_transaction_files()

        for transaction_file in tqdm(transaction_files, desc="Processing transaction files"):
            pairs_file = self.get_matching_pairs_file(transaction_file)

            if pairs_file:

                self.process_file_pair(transaction_file, pairs_file)

            else:
                print(f"No matching pairs file found for {transaction_file}")

        print(f"Time series data saved in {self.processed_data_dir}")

if __name__ == "__main__":

    RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed')

    processor = DataProcessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    processor.preprocess()
