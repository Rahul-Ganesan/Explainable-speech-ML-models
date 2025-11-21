# ==========================================================
# prosodic_data_cleaning.py
# ==========================================================

import os
import time
import logging
from typing import Tuple
import pandas as pd

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================================
# Utility Functions
# ==========================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def split_interviews(
    prosodic_df: pd.DataFrame,
    score_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split prosodic and score data into Interview 1 (P) and Interview 2 (PP).
    Returns:
        prosodic_df_first, prosodic_df_second, score_df_first, score_df_second
    """
    prosodic_df_first = prosodic_df[prosodic_df['participant&question'].str.match(r'^P(?!P)', case=False)].copy()
    prosodic_df_second = prosodic_df[prosodic_df['participant&question'].str.match(r'^PP', case=False)].copy()
    score_df_first = score_df[score_df['Participant'].str.match(r'^P(?!P)', case=False)].copy()
    score_df_second = score_df[score_df['Participant'].str.match(r'^PP', case=False)].copy()
    return prosodic_df_first, prosodic_df_second, score_df_first, score_df_second


def convert_all_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric where possible."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, reset index, and drop NaNs."""
    df = df.sort_values(by=['Participant']).reset_index(drop=True)
    df = df.dropna()
    return df


# ==========================================================
# Core Cleaning Functions
# ==========================================================

def create_all_questions_prosodic_df(prosodic_df: pd.DataFrame, score_df: pd.DataFrame, interview_number: int) -> pd.DataFrame:
    prosodic_df = prosodic_df.copy()
    score_df = score_df.copy()

    prosodic_df['Participant'] = prosodic_df['participant&question'].str.split('Q').str[0].str.replace('^P+', '', regex=True, case=False)
    prosodic_df['Question'] = "Q" + prosodic_df['participant&question'].str.split('Q').str[1].str.replace('^Q+', '', regex=True, case=False)
    score_df['Participant'] = score_df['Participant'].str.replace('^P+', '', regex=True, case=False)

    exclude_cols = ['participant&question', 'Participant', 'Question']
    measurement_cols = [c for c in prosodic_df.columns if c not in exclude_cols]

    cleaned_df = pd.DataFrame()

    for participant in prosodic_df['Participant'].unique():
        chunk = prosodic_df[prosodic_df['Participant'] == participant]
        combined_data = {}
        for _, row in chunk.iterrows():
            question = row['Question']
            for col in measurement_cols:
                combined_data[f"{col}_{question}"] = row[col]
                combined_data = {k.strip(): v for k, v in combined_data.items()}

        combined_data['Participant'] = participant
        combined_data['Interview'] = interview_number
        temp_df = pd.DataFrame([combined_data])
        cleaned_df = pd.concat([cleaned_df, temp_df], ignore_index=True)

    cleaned_df = pd.merge(cleaned_df, score_df, on='Participant', how='left')
    return cleaned_df


def avg_questions_prosodic_df(prosodic_df: pd.DataFrame, score_df: pd.DataFrame, interview_number: int) -> pd.DataFrame:
    prosodic_df = prosodic_df.copy()
    score_df = score_df.copy()

    prosodic_df['Participant'] = prosodic_df['participant&question'].str.split('Q').str[0].str.replace('^P+', '', regex=True, case=False)
    prosodic_df['Question'] = "Q" + prosodic_df['participant&question'].str.split('Q').str[1].str.replace('^Q+', '', regex=True, case=False)
    score_df['Participant'] = score_df['Participant'].str.replace('^P+', '', regex=True, case=False)

    exclude_cols = ['participant&question', 'Participant', 'Question']
    measurement_cols = [c for c in prosodic_df.columns if c not in exclude_cols]

    for col in measurement_cols:
        prosodic_df[col] = pd.to_numeric(prosodic_df[col], errors='coerce')

    cleaned_df = pd.DataFrame()
    for participant in prosodic_df['Participant'].unique():
        chunk = prosodic_df[prosodic_df['Participant'] == participant]
        combined_data = {f"avg_{col}": chunk[col].mean() for col in measurement_cols}
        combined_data['Participant'] = participant
        combined_data['Interview'] = interview_number
        temp_df = pd.DataFrame([combined_data])
        cleaned_df = pd.concat([cleaned_df, temp_df], ignore_index=True)

    cleaned_df = pd.merge(cleaned_df, score_df, on='Participant', how='left')
    return cleaned_df


# ==========================================================
# Main Pipeline Function
# ==========================================================

def process_prosodic_data(prosodic_path: str, score_path: str, output_allq_path: str, output_avg_path: str):
    """
    Full cleaning pipeline for prosodic data.
    """
    start_total = time.time()
    os.makedirs(os.path.dirname(output_allq_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_avg_path), exist_ok=True)

    logger.info("Loading prosodic and score data...")
    prosodic_features_df = pd.read_csv(prosodic_path)
    score_df = pd.read_csv(score_path)

    prosodic_features_df = clean_column_names(prosodic_features_df)
    score_df = clean_column_names(score_df)

    prosodic_df_first, prosodic_df_second, score_df_first, score_df_second = split_interviews(prosodic_features_df, score_df)

    # ---------- All Questions ----------
    logger.info("Processing all-questions prosodic dataset...")
    start = time.time()
    cleaned_df_first = create_all_questions_prosodic_df(prosodic_df_first, score_df_first, 1)
    cleaned_df_second = create_all_questions_prosodic_df(prosodic_df_second, score_df_second, 2)
    all_questions_df = pd.concat([cleaned_df_first, cleaned_df_second], ignore_index=True)
    all_questions_df = convert_all_numeric(all_questions_df)
    all_questions_df = finalize_dataframe(all_questions_df)
    all_questions_df.to_csv(output_allq_path, index=False)
    logger.info(f"All-questions prosodic data saved to {output_allq_path} in {time.time()-start:.2f} seconds")

    # ---------- Averaged Questions ----------
    logger.info("Processing averaged-questions prosodic dataset...")
    start = time.time()
    avg_questions_df_first = avg_questions_prosodic_df(prosodic_df_first, score_df_first, 1)
    avg_questions_df_second = avg_questions_prosodic_df(prosodic_df_second, score_df_second, 2)
    avg_questions_df = pd.concat([avg_questions_df_first, avg_questions_df_second], ignore_index=True)
    avg_questions_df = convert_all_numeric(avg_questions_df)
    avg_questions_df = finalize_dataframe(avg_questions_df)
    avg_questions_df.to_csv(output_avg_path, index=False)
    logger.info(f"Averaged-questions prosodic data saved to {output_avg_path} in {time.time()-start:.2f} seconds")

    logger.info(f"Total prosodic data processing time: {time.time()-start_total:.2f} seconds")
    return all_questions_df, avg_questions_df

# ==========================================================
# Standalone Execution
# ==========================================================
if __name__ == "__main__":
    process_prosodic_data(
        prosodic_path="../data/prosodic_features.csv",
        score_path="../data/scores.csv",
        output_allq_path="../data/all_q_cleaned_prosodic_features.csv",
        output_avg_path="../data/avg_cleaned_prosodic_features.csv"
    )
