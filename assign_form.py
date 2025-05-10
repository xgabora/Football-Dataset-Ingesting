import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

# --- Configuration ---
INPUT_MATCHES_FILE = Path("output_data/matches_with_elo.xlsx")  # Output from assign_elo.py
OUTPUT_DIR = Path("output_data")
OUTPUT_MATCHES_WITH_FORM_FILE = OUTPUT_DIR / "matches_with_form.xlsx"


def calculate_team_form(team_name: str,
                        current_match_date: datetime.date,
                        all_matches_df: pd.DataFrame,
                        num_matches_for_form: int) -> Optional[float]:
    """
    Calculates the form (points earned) for a team in its last 'num_matches_for_form'
    matches played strictly before the 'current_match_date'.
    Returns NaN if the team has not played at least 'num_matches_for_form' prior games,
    or if any of the required past N games have missing scores.

    Args:
        team_name: The name of the team.
        current_match_date: The date of the match for which to calculate form.
        all_matches_df: DataFrame containing all historical matches with results.
                        Expected columns: 'MatchDate', 'HomeTeam', 'AwayTeam',
                                          'FTHome' (Full Time Home Goals),
                                          'FTAway' (Full Time Away Goals).
        num_matches_for_form: The number of past matches to consider for form.

    Returns:
        The form (total points) or np.nan if insufficient/invalid past matches.
    """
    team_past_matches = all_matches_df[
        ((all_matches_df['HomeTeam'] == team_name) | (all_matches_df['AwayTeam'] == team_name)) &
        (all_matches_df['MatchDate'] < current_match_date)
        ].copy()

    team_past_matches.sort_values(by='MatchDate', ascending=False, inplace=True)

    if len(team_past_matches) < num_matches_for_form:
        return np.nan

    last_n_matches = team_past_matches.head(num_matches_for_form)

    if len(last_n_matches) < num_matches_for_form:
        return np.nan

    total_points = 0
    for _, match_row in last_n_matches.iterrows():
        home_goals = match_row.get('FTHome')
        away_goals = match_row.get('FTAway')

        if pd.isna(home_goals) or pd.isna(away_goals):
            return np.nan

        if match_row['HomeTeam'] == team_name:
            if home_goals > away_goals:
                total_points += 3
            elif home_goals == away_goals:
                total_points += 1
        elif match_row['AwayTeam'] == team_name:  # Team played away
            if away_goals > home_goals:
                total_points += 3
            elif away_goals == home_goals:
                total_points += 1

    return float(total_points)


def main():
    """
    Loads match data, calculates Form3 and Form5 for home and away teams,
    and saves the updated match data.
    """
    print(f"--- Calculating Team Form Script Started ---")

    # --- 1. Load Data ---
    if not INPUT_MATCHES_FILE.exists():
        print(f"Error: Matches file not found at {INPUT_MATCHES_FILE}")
        return

    try:
        df_matches = pd.read_excel(INPUT_MATCHES_FILE)
        print(f"Successfully loaded {len(df_matches)} matches.")
    except Exception as e:
        print(f"Error loading data from Excel file: {e}")
        return

    # --- 2. Pre-process Data ---
    df_matches['MatchDate'] = pd.to_datetime(df_matches['MatchDate']).dt.date

    score_cols = ['FTHome', 'FTAway']
    for col in score_cols:
        if col in df_matches.columns:
            df_matches[col] = pd.to_numeric(df_matches[col], errors='coerce')
        else:
            print(
                f"Warning: Score column '{col}' not found in the input file '{INPUT_MATCHES_FILE.name}'. Form calculation might be incorrect or result in NaNs.")
            df_matches[col] = np.nan

    df_matches.sort_values(by='MatchDate', ascending=True, inplace=True)
    df_matches.reset_index(drop=True, inplace=True)

    # --- 3. Calculate Form for Each Match ---
    form3_home_list = []
    form5_home_list = []
    form3_away_list = []
    form5_away_list = []

    print("Calculating form for matches...")
    for index, current_match_row in df_matches.iterrows():
        current_match_date = current_match_row['MatchDate']
        home_team = current_match_row['HomeTeam']
        away_team = current_match_row['AwayTeam']

        form3_home = calculate_team_form(home_team, current_match_date, df_matches, 3)
        form5_home = calculate_team_form(home_team, current_match_date, df_matches, 5)
        form3_away = calculate_team_form(away_team, current_match_date, df_matches, 3)
        form5_away = calculate_team_form(away_team, current_match_date, df_matches, 5)

        form3_home_list.append(form3_home)
        form5_home_list.append(form5_home)
        form3_away_list.append(form3_away)
        form5_away_list.append(form5_away)

        if (index + 1) % 10 == 0:
            print(f"Processed form for {index + 1}/{len(df_matches)} matches...")

    df_matches['Form3Home'] = form3_home_list
    df_matches['Form5Home'] = form5_home_list
    df_matches['Form3Away'] = form3_away_list
    df_matches['Form5Away'] = form5_away_list
    print("Form calculation complete.")

    # --- 4. Save Output ---
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}: {e}")
        return

    try:
        df_matches.to_excel(OUTPUT_MATCHES_WITH_FORM_FILE, index=False)
        print(f"Success! Updated match data with form saved to: {OUTPUT_MATCHES_WITH_FORM_FILE}")
    except Exception as e:
        print(f"Error: Could not save updated match data to {OUTPUT_MATCHES_WITH_FORM_FILE}: {e}")
        print(f"Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    print(f"--- Calculating Team Form Script Finished ---")


if __name__ == "__main__":
    main()