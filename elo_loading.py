import requests
from datetime import date
import csv
from io import StringIO
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any

# --- Configuration ---
OUTPUT_DIR = Path("output_data")
TARGET_COUNTRY_CODE = "ENG"  # Only interested in English teams for this public example
SPECIFIC_DATES_TO_FETCH = [
    date(2025, 1, 1),
    date(2025, 1, 15),
    date(2025, 2, 1),
    date(2025, 2, 15),
]


def get_elo_data_for_date(api_date: date) -> Optional[List[Dict[str, Any]]]:
    """Fetches Elo data from the API for a given date and filters for the target country."""
    url = f"http://api.clubelo.com/{api_date.strftime('%Y-%m-%d')}"
    print(f"Fetching Elo data for {api_date.strftime('%Y-%m-%d')}...")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

    csv_data = StringIO(response.text)
    reader = csv.DictReader(csv_data)

    filtered_elo_data = []
    for row in reader:
        if row.get('Country') == TARGET_COUNTRY_CODE:
            try:
                processed_row = {
                    'date': api_date.strftime('%Y-%m-%d'),
                    'club': row['Club'],
                    'country': row['Country'],
                    'elo': round(float(row['Elo']), 2)
                }
                filtered_elo_data.append(processed_row)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to data issue (date: {api_date}, row: {row}): {e}")
                continue

    return filtered_elo_data


def main():
    """Fetches Elo ratings for specified dates and English teams, then saves to Excel."""
    print(f"--- Elo Data Loading Script Started ---")

    all_elo_ratings = []

    for target_date in SPECIFIC_DATES_TO_FETCH:
        elo_data_for_day = get_elo_data_for_date(target_date)
        if elo_data_for_day:
            all_elo_ratings.extend(elo_data_for_day)
            print(f"Successfully fetched and processed {len(elo_data_for_day)} English teams for {target_date}.")
        else:
            print(f"No data processed for {target_date}.")

    if not all_elo_ratings:
        print("No Elo data was collected. Exiting.")
        print(f"--- Elo Data Loading Script Finished ---")
        return

    df_elo = pd.DataFrame(all_elo_ratings)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}: {e}")
        print(f"--- Elo Data Loading Script Finished ---")
        return

    output_file_path = OUTPUT_DIR / "processed_elo.xlsx"
    try:
        df_elo.to_excel(output_file_path, index=False)
        print(f"Success! Elo data saved to: {output_file_path}")
    except Exception as e:
        print(f"Error: Could not save Elo data to {output_file_path}: {e}")
        print(f"Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    print(f"--- Elo Data Loading Script Finished ---")


if __name__ == "__main__":
    main()