import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_MATCHES_FILE = Path("output_data/matches_with_form.xlsx")  # Output from calculate_form.py
MODEL_DIR = Path("clusters/")
SCALER_FILE = MODEL_DIR / "robust_scaler.pkl"
KMEANS_MODEL_FILE = MODEL_DIR / "kmeans_model.pkl"
OUTPUT_DIR = Path("output_data")
OUTPUT_MATCHES_WITH_CLUSTERS_FILE = OUTPUT_DIR / "matches_with_clusters.xlsx"

# Define cluster names based on your original script
CLUSTER_NAMES = ['LTH', 'HTB', 'LTA', 'VAD', 'VHD', 'PHB']
# Features used for clustering
FEATURE_COLS = ['ShotEfficiencyDiff', 'PossessionDominance', 'GamePhysicality', 'GameTempo', 'EloDifferential']


# --- Model Loading ---
def load_models():
    """Loads the RobustScaler and KMeans model from .pkl files."""
    scaler = None
    kmeans = None
    try:
        if SCALER_FILE.exists():
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
        if KMEANS_MODEL_FILE.exists():
            with open(KMEANS_MODEL_FILE, 'rb') as f:
                kmeans = pickle.load(f)

        if scaler and kmeans:
            print("Models loaded successfully from .pkl files.")
            return scaler, kmeans
        else:
            print("Warning: One or both .pkl model files not found. Attempting fallback.")
            raise FileNotFoundError  # Trigger fallback

    except Exception as e:
        print(
            f"Could not load saved models from .pkl files (Error: {e}). Attempting to create with predefined centers.")
        scaler = RobustScaler()
        # Sample data to fit the scaler if pkl is missing.
        sample_data_for_scaler = pd.DataFrame({
            'ShotEfficiencyDiff': [-0.5, 0, 0.5, -0.2, 0.2],
            'PossessionDominance': [0.3, 0.5, 0.7, 0.4, 0.6],
            'GamePhysicality': [15, 25, 40, 20, 30],
            'GameTempo': [15, 25, 35, 20, 30],
            'EloDifferential': [-200, 0, 200, -100, 100]
        })
        scaler.fit(sample_data_for_scaler[FEATURE_COLS])

        cluster_centers_predefined = np.array([
            [0.2323, 0.5144, 24.9182, 20.0713, 7.0859],  # LTH
            [-0.0080, 0.5551, 24.4104, 31.3046, 0.5123],  # HTB
            [-0.2471, 0.6107, 25.4407, 20.3818, -0.2755],  # LTA
            [-0.0475, 0.3813, 24.4507, 24.3623, -198.1429],  # VAD
            [0.0641, 0.7044, 23.2613, 25.8237, 210.1748],  # VHD
            [0.0194, 0.5484, 37.9422, 22.9559, -4.5165]  # PHB
        ])
        # Rounding predefined centers for consistency if they were truncated in original display

        kmeans = KMeans(n_clusters=len(CLUSTER_NAMES), init=cluster_centers_predefined, n_init=1, random_state=42)
        dummy_data_for_kmeans = pd.DataFrame(cluster_centers_predefined, columns=FEATURE_COLS)
        kmeans.fit(dummy_data_for_kmeans)
        kmeans.cluster_centers_ = cluster_centers_predefined

        print("Created fallback models with predefined centers.")
        return scaler, kmeans


# --- Feature Engineering ---
def estimate_elo_from_odds(row, default_elo_val=1500.0):  # Accept default_elo_val
    """Estimates Elo ratings from betting odds if Elo is missing."""
    # default_elo = 1500.0 # Removed, use passed argument
    home_elo, away_elo = row.get('HomeElo'), row.get('AwayElo')

    if pd.isna(home_elo) or pd.isna(away_elo):
        odd_home, odd_away = row.get('OddHome'), row.get('OddAway')
        if pd.notna(odd_home) and pd.notna(odd_away) and odd_home > 1.0 and odd_away > 1.0:
            try:
                p_home = 1.0 / float(odd_home)
                p_away = 1.0 / float(odd_away)
                total_p = p_home + p_away
                if total_p > 0:
                    p_home_norm = p_home / total_p

                    if p_home_norm > 0 and p_home_norm < 1:
                        log_odds_ratio = np.log(p_home_norm / (1.0 - p_home_norm))
                        elo_diff = log_odds_ratio * (400 / np.log(10))

                        home_elo = default_elo_val + elo_diff / 2.0  # Use default_elo_val
                        away_elo = default_elo_val - elo_diff / 2.0  # Use default_elo_val
                    else:
                        home_elo, away_elo = None, None
                else:
                    home_elo, away_elo = None, None

            except Exception:
                home_elo, away_elo = None, None

    return pd.Series({'HomeElo': home_elo, 'AwayElo': away_elo})


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the features required for clustering from the match data."""
    df_processed = df.copy()
    default_elo = 1500.0  # Define default_elo here

    # Ensure necessary columns are numeric, coercing errors
    numeric_cols_stats = ['HomeShots', 'AwayShots', 'HomeTarget', 'AwayTarget',
                          'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners']
    numeric_cols_elo_odds = ['HomeElo', 'AwayElo', 'OddHome', 'OddAway']

    for col in numeric_cols_stats + numeric_cols_elo_odds:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in input data. Will be treated as NaN.")
            df_processed[col] = np.nan  # Create column if missing so subsequent ops don't fail

    # Estimate missing Elo (if any)
    if 'HomeElo' in df_processed.columns and 'AwayElo' in df_processed.columns:
        estimated_elos = df_processed.apply(lambda row: estimate_elo_from_odds(row, default_elo_val=default_elo),
                                            axis=1)
        df_processed['HomeElo'] = estimated_elos['HomeElo']
        df_processed['AwayElo'] = estimated_elos['AwayElo']

    # Create engineered features
    epsilon = 0.001
    df_processed['HomeShots_eff'] = df_processed['HomeShots'].fillna(0).clip(lower=epsilon)
    df_processed['AwayShots_eff'] = df_processed['AwayShots'].fillna(0).clip(lower=epsilon)

    df_processed['ShotEfficiencyDiff'] = (df_processed['HomeTarget'].fillna(0) / df_processed['HomeShots_eff']) - \
                                         (df_processed['AwayTarget'].fillna(0) / df_processed['AwayShots_eff'])

    home_possession_metric = df_processed['HomeShots'].fillna(0) + df_processed['HomeCorners'].fillna(0)
    away_possession_metric = df_processed['AwayShots'].fillna(0) + df_processed['AwayCorners'].fillna(0)
    total_possession_metric = (home_possession_metric + away_possession_metric).clip(lower=epsilon)
    df_processed['PossessionDominance'] = home_possession_metric / total_possession_metric

    df_processed['GamePhysicality'] = df_processed['HomeFouls'].fillna(0) + df_processed['AwayFouls'].fillna(0)
    df_processed['GameTempo'] = df_processed['HomeShots'].fillna(0) + df_processed['AwayShots'].fillna(0)

    # Use the defined default_elo for filling NaNs before calculating EloDifferential
    df_processed['EloDifferential'] = df_processed['HomeElo'].fillna(default_elo) - df_processed['AwayElo'].fillna(
        default_elo)

    df_features = df_processed[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        if df_features[col].isna().any():
            median_val = df_features[col].median()
            if pd.notna(median_val):  # Ensure median is not NaN
                df_features[col].fillna(median_val, inplace=True)
                print(f"Filled NaNs in feature '{col}' with median ({median_val:.2f}).")
            else:
                df_features[col].fillna(0, inplace=True)
                print(f"Filled NaNs in feature '{col}' with 0 (median was NaN).")

    return df_features, df_processed.index

# --- Clustering ---
def calculate_cluster_probabilities(features_df: pd.DataFrame, scaler: RobustScaler, kmeans: KMeans) -> pd.DataFrame:
    """Calculates cluster probabilities for the given features."""
    if features_df.empty:
        return pd.DataFrame(columns=[f'C_{name}' for name in CLUSTER_NAMES])

    features_scaled = scaler.transform(features_df)

    distances_to_centers = pairwise_distances(features_scaled, kmeans.cluster_centers_)

    # Softmax with temperature to convert distances to probabilities
    # Lower temperature = more confident (sharper) probabilities
    # Higher temperature = softer probabilities
    temperature = 0.35  # From your original script

    neg_dist_scaled = -distances_to_centers / temperature
    max_val_per_row = np.max(neg_dist_scaled, axis=1, keepdims=True)
    exp_values = np.exp(neg_dist_scaled - max_val_per_row)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    prob_df = pd.DataFrame(probabilities, index=features_df.index, columns=[f'C_{name}' for name in CLUSTER_NAMES])
    return prob_df.round(4)


# --- Main Execution ---
def main():
    print(f"--- Assigning Clusters Script Started ---")

    # 1. Load Models
    scaler, kmeans = load_models()
    if scaler is None or kmeans is None:
        print("Error: Critical models could not be loaded or created. Exiting.")
        return

    # 2. Load Match Data
    if not INPUT_MATCHES_FILE.exists():
        print(f"Error: Matches file not found at {INPUT_MATCHES_FILE}")
        return
    try:
        df_matches_full = pd.read_excel(INPUT_MATCHES_FILE)
        print(f"Successfully loaded {len(df_matches_full)} matches.")
    except Exception as e:
        print(f"Error loading data from Excel file {INPUT_MATCHES_FILE}: {e}")
        return

    if df_matches_full.empty:
        print("No matches loaded. Exiting.")
        return

    # 3. Prepare Features
    print("Preparing features for clustering...")
    df_features, original_index = prepare_features(df_matches_full.copy())  # Pass a copy

    if df_features.empty:
        print("No features could be prepared. Exiting.")
        return

    # 4. Calculate Cluster Probabilities
    print("Calculating cluster probabilities...")
    df_cluster_probs = calculate_cluster_probabilities(df_features, scaler, kmeans)

    # Ensure df_cluster_probs has the original index for correct joining
    df_cluster_probs.index = original_index

    # 5. Combine with original match data
    df_output = df_matches_full.join(df_cluster_probs)

    # 6. Save Output
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR}: {e}")
        return

    try:
        df_output.to_excel(OUTPUT_MATCHES_WITH_CLUSTERS_FILE, index=False)
        print(f"Success! Updated match data with clusters saved to: {OUTPUT_MATCHES_WITH_CLUSTERS_FILE}")
    except Exception as e:
        print(f"Error: Could not save updated match data to {OUTPUT_MATCHES_WITH_CLUSTERS_FILE}: {e}")
        print(f"Ensure 'openpyxl' is installed (`pip install openpyxl`).")

    print(f"--- Assigning Clusters Script Finished ---")


if __name__ == "__main__":
    main()