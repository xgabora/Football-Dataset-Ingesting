import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Configuration ---
INPUT_MATCHES_FILE = Path("output_data/processed_E0.xlsx")
INPUT_ELO_FILE = Path("output_data/processed_elo.xlsx")
OUTPUT_DIR = Path("output_data")
OUTPUT_MATCHES_WITH_ELO_FILE = OUTPUT_DIR / "matches_with_elo.xlsx"

# How far back from the match date to consider Elo ratings
ELO_LOOKBACK_DAYS = 365  # Or None to consider all past ratings


def get_closest_elo(match_date: datetime.date,
                    team_name: str,
                    team_elo_ratings: pd.DataFrame,
                    lookback_days: Optional[int] = ELO_LOOKBACK_DAYS) -> Optional[float]:
    """
    Finds the most recent Elo rating for a team on or before the match date.

    Args:
        match_date: The date of the match.
        team_name: The name of the team.
        team_elo_ratings: A DataFrame of Elo ratings for a specific team,
                          sorted by date (descending recommended).
                          Expected columns: 'date' (datetime.date), 'elo' (float).
        lookback_days: Optional. How many days before the match_date to consider ratings.
                       If None, all past ratings are considered.

    Returns:
        The Elo rating (float) or None if no suitable record is found.
    """
    if team_elo_ratings.empty:
        return None

    # Filter ratings on or before the match date
    relevant_ratings = team_elo_ratings[
        team_elo_ratings['date'] <= match_date].copy()

    if lookback_days is not None:
        min_elo_date = match_date - timedelta(days=lookback_days)
        relevant_ratings = relevant_ratings[relevant_ratings['date'] >= min_elo_date]

    if relevant_ratings.empty:
        return None

    # The DataFrame should be pre-sorted by date descending, so the first row is the most recent
    closest_elo_record = relevant_ratings.iloc[0]
    return float(closest_elo_record['elo'])


def main():
    """
    Loads match data and Elo ratings, assigns Elo to matches,
    and saves the updated match data.
    """
    print(f"--- Assigning Elo Ratings Script Started ---")

    # --- 1. Load Data ---
    if not INPUT_MATCHES_FILE.exists():
        print(f"Error: Matches file not found at {INPUT_MATCHES_FILE}")
        return
    if not INPUT_ELO_FILE.exists():
        print(f"Error: Elo ratings file not found at {INPUT_ELO_FILE}")
        return

    try:
        df_matches = pd.read_excel(INPUT_MATCHES_FILE)
        df_elo = pd.read_excel(INPUT_ELO_FILE)
        print(f"Successfully loaded {len(df_matches)} matches and {len(df_elo)} Elo records.")
    except Exception as e:
        print(f"Error loading data from Excel files: {e}")
        return

    # --- 2. Pre-process Data ---
    df_matches['MatchDate'] = pd.to_datetime(df_matches['MatchDate']).dt.date
    df_elo['date'] = pd.to_datetime(df_elo['date']).dt.date

    # Sort Elo ratings by club and then by date descending for efficient lookup
    df_elo.sort_values(by=['club', 'date'], ascending=[True, False], inplace=True)

    # Group Elo ratings by team for faster access
    elo_ratings_by_team: Dict[str, pd.DataFrame] = {
        team: group[['date', 'elo']] for team, group in df_elo.groupby('club')
    }

    # --- 3. Assign Elo Ratings to Matches ---
    home_elos = []
    away_elos = []

    print("Assigning Elo ratings to matches...")
    for index, match_row in df_matches.iterrows():
        match_date = match_row['MatchDate']
        home_team_name = match_row['HomeTeam']
        away_team_name = match_row['AwayTeam']

        home_team_elo_history = elo_ratings_by_team.get(home_team_name, pd.DataFrame())
        away_team_elo_history = elo_ratings_by_team.get(away_team_name, pd.DataFrame())

        home_elo = get_closest_elo(match_date, home_team_name, home_team_elo_history)
        away_elo = get_closest_elo(match_date, away_team_name, away_team_elo_history)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

    df_matches['HomeElo'] = home_elos
    df_matches['AwayElo'] = away_elos
    print("Elo assignment complete.")

    # --- 4. Save Output ---
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}: {e}")
        return

    try:
        df_matches.to_excel(OUTPUT_MATCHES_WITH_ELO_FILE, index=False)
        print(f"Success! Updated match data saved to: {OUTPUT_MATCHES_WITH_ELO_FILE}")
    except Exception as e:
        print(f"Error: Could not save updated match data to {OUTPUT_MATCHES_WITH_ELO_FILE}: {e}")
        print(f"Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    print(f"--- Assigning Elo Ratings Script Finished ---")


if __name__ == "__main__":
    main()