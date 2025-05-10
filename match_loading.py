import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import math
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path("output_data")

# --- Column Mapping Dictionaries ---
# These define how to map various source column names to standardized names
# and how to prioritize them if multiple potential source columns exist.
RESULT_COLUMNS = {
    'FTHome': ['FTHG', 'HG', 'FTHome'],
    'FTAway': ['FTAG', 'AG', 'FTAway'],
    'FTResult': ['FTR', 'Res', 'FTResult'],
    'HTHome': ['HTHG', 'HTHome'],
    'HTAway': ['HTAG', 'HTAway'],
    'HTResult': ['HTR', 'HTResult'],
}

STATS_COLUMNS = {
    'HomeShots': ['HS', 'HomeShots'],
    'AwayShots': ['AS', 'AwayShots'],
    'HomeTarget': ['HST', 'HomeTarget'],
    'AwayTarget': ['AST', 'AwayTarget'],
    'HomeFouls': ['HF', 'HFKC', 'HomeFouls'],
    'AwayFouls': ['AF', 'AFKC', 'AwayFouls'],
    'HomeCorners': ['HC', 'HomeCorners'],
    'AwayCorners': ['AC', 'AwayCorners'],
    'HomeYellow': ['HY', 'HomeYellow'],
    'AwayYellow': ['AY', 'AwayYellow'],
    'HomeRed': ['HR', 'HomeRed'],
    'AwayRed': ['AR', 'AwayRed']
}

ODDS_HIERARCHY = {
    'OddHome': ['HomeOdds', 'AvgH', 'B365H', 'GBH', 'IWH', 'LBH', 'PSH', 'WHH', 'VCH', 'OddHome'],
    'OddDraw': ['DrawOdds', 'AvgD', 'B365D', 'GBD', 'IWD', 'LBD', 'PSD', 'WHD', 'VCD', 'OddDraw'],
    'OddAway': ['AwayOdds', 'AvgA', 'B365A', 'GBA', 'IWA', 'LBA', 'PSA', 'WHA', 'VCA', 'OddAway'],
    'MaxHome': ['MaxH', 'BbMxH', 'MaxHome'],
    'MaxDraw': ['MaxD', 'BbMxD', 'MaxDraw'],
    'MaxAway': ['MaxA', 'BbMxA', 'MaxAway'],
    'Over25': ['Over25', 'BbAv>2.5', 'Avg>2.5', 'P>2.5', 'B365>2.5'],
    'Under25': ['Under25', 'BbAv<2.5', 'Avg<2.5', 'P<2.5', 'B365<2.5'],
    'MaxOver25': ['MaxOver25', 'BbMx>2.5', 'Max>2.5'],
    'MaxUnder25': ['MaxUnder25', 'BbMx<2.5', 'Max<2.5'],
    'HandiSize': ['HandiSize', 'BbAHh', 'AHh', 'GBAH', 'LBAH', 'B365AH', 'PSAH'],
    'HandiHome': ['HandiHome', 'BbAvAHH', 'GBAHH', 'LBAHH', 'B365AHH', 'PSAHH', 'PAHH'],
    'HandiAway': ['HandiAway', 'BbAvAHA', 'GBAHA', 'LBAHA', 'B365AHA', 'PSAHA', 'PAHA']
}


# --- Helper Functions for Data Parsing and Cleaning ---

def parse_date(date_str: Any) -> Optional[str]:
    """Parse various date string formats to 'YYYY-MM-DD'."""
    if pd.isna(date_str):
        return None
    if isinstance(date_str, datetime):
        return date_str.strftime('%Y-%m-%d')

    date_str_cleaned = str(date_str).strip()
    if not date_str_cleaned:
        return None

    formats_to_try = [
        '%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%y-%m-%d',
        '%m/%d/%Y', '%m/%d/%y', '%d-%m-%Y', '%Y/%m/%d'
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str_cleaned, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    try:
        return pd.to_datetime(date_str_cleaned).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None


def parse_time(time_str: Any) -> Optional[str]:
    """Parse various time string formats to 'HH:MM:SS'."""
    from datetime import time as dt_time
    from pandas import Timestamp as pd_Timestamp

    if pd.isna(time_str):
        return None

    if isinstance(time_str, (dt_time, pd_Timestamp)):
        return time_str.strftime('%H:%M:%S')

    time_str_cleaned = str(time_str).strip()
    if not time_str_cleaned:
        return None

    try:
        float_val = float(time_str_cleaned)
        if 0 <= float_val < 1:
            total_seconds = int(float_val * 24 * 60 * 60)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except ValueError:
        pass

    formats_to_try = ['%H:%M', '%H.%M', '%H:%M:%S']
    for fmt in formats_to_try:
        try:
            return datetime.strptime(time_str_cleaned, fmt).strftime('%H:%M:%S')
        except ValueError:
            continue
    try:
        return pd.to_datetime(time_str_cleaned).strftime('%H:%M:%S')
    except (ValueError, TypeError):
        return None


def get_mapped_value(row: pd.Series, target_column_base: str, mapping_dict: Dict[str, List[str]]) -> Any:
    """Retrieves a value from a row based on a list of possible source column names."""
    if target_column_base in mapping_dict:
        for source_column_name in mapping_dict[target_column_base]:
            if source_column_name in row.index and pd.notna(row[source_column_name]):
                return row[source_column_name]
    return None


def process_result_value(value: Any) -> Optional[str]:
    """Standardizes match result values (H, D, A)."""
    if pd.isna(value): return None
    value_str = str(value).strip().upper()
    if value_str in ['H', 'HOME', 'WIN']: return 'H'
    if value_str in ['D', 'DRAW']: return 'D'
    if value_str in ['A', 'AWAY', 'LOSS']: return 'A'
    return None


def clean_numeric_value(value: Any) -> Optional[float]:
    """Cleans a value to be a float, or None if not possible."""
    if pd.isna(value): return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_value_from_series_by_candidates(row_series: pd.Series, column_candidates: List[str]) -> Any:
    """Attempts to get a value from a pandas Series using a list of candidate column names."""
    for col_name in column_candidates:
        if col_name in row_series.index and pd.notna(row_series[col_name]):
            return row_series[col_name]
    if column_candidates and '\ufeff' + column_candidates[0] in row_series.index and pd.notna(
            row_series['\ufeff' + column_candidates[0]]):
        return row_series['\ufeff' + column_candidates[0]]
    return None


# --- Core Data Processing per Row ---
def process_row(row: pd.Series) -> Dict[str, Any]:
    """Processes a single row of match data, standardizing and cleaning values."""
    processed_data = {}

    div_val = _get_value_from_series_by_candidates(row, ['Div', 'Division'])
    date_val = _get_value_from_series_by_candidates(row, ['Date', 'MatchDate', 'datetime', 'date'])
    time_val = _get_value_from_series_by_candidates(row, ['Time', 'time', 'Kickoff_Time', 'KO_Time', 'Kickoff'])
    home_team_val = _get_value_from_series_by_candidates(row, ['HomeTeam', 'Home', 'home_team', 'hometeam'])
    away_team_val = _get_value_from_series_by_candidates(row, ['AwayTeam', 'Away', 'away_team', 'awayteam'])

    processed_data['Division'] = str(div_val).strip() if pd.notna(div_val) else None
    processed_data['MatchDate'] = parse_date(date_val)
    processed_data['MatchTime'] = parse_time(time_val)
    processed_data['HomeTeam'] = str(home_team_val).strip() if pd.notna(home_team_val) else None
    processed_data['AwayTeam'] = str(away_team_val).strip() if pd.notna(away_team_val) else None

    for target_col, _ in RESULT_COLUMNS.items():
        raw_value = get_mapped_value(row, target_col, RESULT_COLUMNS)
        processed_data[target_col] = process_result_value(raw_value) if 'Result' in target_col else clean_numeric_value(
            raw_value)

    for target_col, _ in STATS_COLUMNS.items():
        raw_value = get_mapped_value(row, target_col, STATS_COLUMNS)
        processed_data[target_col] = clean_numeric_value(raw_value)

    for target_col, source_cols_hierarchy in ODDS_HIERARCHY.items():
        found_value = None
        for source_col in source_cols_hierarchy:
            if source_col in row.index and pd.notna(row[source_col]):
                found_value = row[source_col]
                break
        processed_data[target_col] = clean_numeric_value(found_value)

    return processed_data


# --- Chunk and File Processing ---
def process_chunk(df_chunk: pd.DataFrame) -> List[Dict[str, Any]]:
    """Processes a DataFrame chunk row by row."""
    processed_rows = []
    for _, row in df_chunk.iterrows():
        try:
            processed_row_data = process_row(row)
            if processed_row_data.get('HomeTeam') and processed_row_data.get('AwayTeam'):
                processed_rows.append(processed_row_data)
        except Exception:
            continue
    return processed_rows


def clean_column_name(col_name: str) -> str:
    """Cleans column name, including removing UTF-8 BOM."""
    name = str(col_name)
    if name.startswith('\ufeff'):  # Check for Byte Order Mark
        name = name[1:]
    return name.strip().replace('\n', ' ').replace('\r', '')


def read_csv_robustly(file_path: Path) -> Optional[pd.DataFrame]:
    """Reads a CSV file, trying various common encodings and separators."""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
    na_values_list = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                      '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>',
                      'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', on_bad_lines='warn',
                             dtype=str, keep_default_na=False, na_values=na_values_list)
            df.replace(na_values_list, np.nan, inplace=True)
            return df
        except Exception:
            pass  # Try next encoding / separator combination

        separators_to_try = [',', ';', '\t']
        for sep in separators_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine='python', on_bad_lines='warn',
                                 dtype=str, keep_default_na=False, na_values=na_values_list)
                df.replace(na_values_list, np.nan, inplace=True)
                return df
            except Exception:
                pass  # Try next
    return None


def load_and_process_data(file_path: Path, chunk_size: int = 2000) -> Optional[pd.DataFrame]:
    """Loads, processes file data in chunks, and returns a processed DataFrame."""
    try:
        na_values_list = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                          '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, dtype=str, keep_default_na=False, na_values=na_values_list)
            df.replace(na_values_list, np.nan, inplace=True)
        elif file_path.suffix.lower() == '.csv':
            df = read_csv_robustly(file_path)
        else:
            print(f"Error: Unsupported file type: {file_path.suffix} for {file_path}")
            return None

        if df is None or df.empty:
            print(f"Warning: No data read from {file_path}, or file is empty/unreadable.")
            return None

        df.columns = [clean_column_name(col) for col in df.columns]

        df.dropna(how='all', axis=1, inplace=True)
        df.dropna(how='all', axis=0, inplace=True)

        if df.empty:
            print(f"Warning: Data empty after initial cleaning for {file_path}.")
            return None

        num_chunks = math.ceil(len(df) / chunk_size)
        chunks = np.array_split(df, num_chunks) if num_chunks > 0 else []

        all_processed_rows: List[Dict[str, Any]] = []
        num_workers = max(1, multiprocessing.cpu_count() - 1)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk_idx = {
                executor.submit(process_chunk, chunk): i
                for i, chunk in enumerate(chunks) if not chunk.empty
            }
            for future in as_completed(future_to_chunk_idx):
                try:
                    chunk_result = future.result()
                    if chunk_result:
                        all_processed_rows.extend(chunk_result)
                except Exception as e:
                    print(f"Error processing a data chunk from {file_path}: {e}")

        if not all_processed_rows:
            print(f"Warning: No data rows successfully processed from {file_path} after all chunks.")
            return None

        processed_df = pd.DataFrame(all_processed_rows)
        return processed_df

    except Exception as e:
        print(f"Critical error during loading/processing of {file_path}: {e}")
        return None


# --- Main Execution ---
def main():
    """Main function to orchestrate data processing for a single file."""
    input_file_path = Path("example_data/E0.csv")

    print(f"--- Football Data Processing Script (match_loading.py) Started ---")
    print(f"Input file: {input_file_path}")

    if not input_file_path.exists():
        print(f"Error: Input file '{input_file_path}' not found.")
        return

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}: {e}")
        return

    final_processed_df = load_and_process_data(input_file_path)

    if final_processed_df is not None and not final_processed_df.empty:
        output_filename = f"processed_{input_file_path.stem}.xlsx"
        output_file_path = OUTPUT_DIR / output_filename
        try:
            final_processed_df.to_excel(output_file_path, index=False)
            print(f"Success! Processed data saved to: {output_file_path}")
        except Exception as e:
            print(f"Error: Could not save processed data to {output_file_path}: {e}")
            print(f"Ensure 'openpyxl' is installed (`pip install openpyxl`).")
    else:
        print(f"Warning: No processed data from {input_file_path}. Output file not created.")

    print(f"--- Football Data Processing Script (match_loading.py) Finished ---")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()