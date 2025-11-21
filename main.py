# ==========================================================
# main_pipeline_driver.py
# Full pipeline for Multimodal ML Project
# ==========================================================

import os
import sys
import time
import logging
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Create required directories
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# Adjust sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------


# ==========================================================
# Main Pipeline Execution
# ==========================================================
def main_pipeline():
    start_total = time.time()

    # -----------------------------
    # Clean Prosodic Data
    # -----------------------------
    from src.prosodic_data_cleaning import process_prosodic_data

    logger.info("Cleaning prosodic data...")
    start = time.time()
    cleaned_prosodic_allq, cleaned_prosodic_avg = process_prosodic_data(
        prosodic_path="data/prosodic_features.csv",
        score_path="data/scores.csv",
        output_allq_path="data/all_q_cleaned_prosodic_features.csv",
        output_avg_path="data/avg_cleaned_prosodic_features.csv"
    )
    logger.info(f"Prosodic data cleaned in {time.time() - start:.2f} seconds.")

    # -----------------------------
    # Clean Textual Data
    # -----------------------------
    from src.textual_data_cleaning import process_textual_data

    logger.info("Cleaning textual data...")
    start = time.time()
    cleaned_textual = process_textual_data(
        transcripts_path="data/transcripts.csv",
        score_path="data/scores.csv",
        output_path="data/text_cleaned_features.csv"
    )
    logger.info(f"Textual data cleaned in {time.time() - start:.2f} seconds.")

    # -----------------------------
    # Feature Selection
    # -----------------------------
    from src.feature_selection import run_feature_selection

    logger.info("Running feature selection...")
    start = time.time()
    prosodic_avg_features = run_feature_selection(cleaned_prosodic_avg, dataset_name="AVERAGED PROSODIC")
    prosodic_all_features = run_feature_selection(cleaned_prosodic_allq, dataset_name="ALL Q PROSODIC")
    text_features = run_feature_selection(cleaned_textual, dataset_name="TEXT FEATURES")
    logger.info(f"Feature selection completed in {time.time() - start:.2f} seconds.")

    # -----------------------------
    # Independent Modeling
    # -----------------------------
    from src.modeling import run_modeling

    logger.info("Modeling on prosodic all-question dataset...")
    start = time.time()
    all_q_prosodic_results, best_all_q_models = run_modeling(
        df=cleaned_prosodic_allq,
        feature_sets=prosodic_all_features,
        dataset_name="ALL Q PROSODIC",
        output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
        test_size=0.2,
        random_state=42,
        save_path="results/all_q_prosodic_modeling_results.csv"
    )
    logger.info(f"All-question prosodic modeling completed in {time.time() - start:.2f} seconds.")

    logger.info("Modeling on prosodic averaged dataset...")
    start = time.time()
    avg_prosodic_results, best_avg_prosodic_models = run_modeling(
        df=cleaned_prosodic_avg,
        feature_sets=prosodic_avg_features,
        dataset_name="AVERAGED PROSODIC",
        output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
        test_size=0.2,
        random_state=42,
        save_path="results/avg_prosodic_modeling_results.csv"
    )
    logger.info(f"Averaged prosodic modeling completed in {time.time() - start:.2f} seconds.")

    logger.info("Modeling on textual dataset...")
    start = time.time()
    textual_results, best_textual_models = run_modeling(
        df=cleaned_textual,
        feature_sets=text_features,
        dataset_name="TEXT FEATURES",
        output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
        test_size=0.2,
        random_state=42,
        save_path="results/textual_modeling_results.csv"
    )
    logger.info(f"Textual modeling completed in {time.time() - start:.2f} seconds.")

    # -----------------------------
    # Multimodal Modeling
    # -----------------------------
    from src.multimodal_modeling import run_multimodal_modeling

    logger.info("Running multimodal modeling (prosodic + textual)...")
    start = time.time()
    multimodal_results = run_multimodal_modeling(
        prosodic_df=cleaned_prosodic_avg,
        textual_df=cleaned_textual,
        prosodic_feature_sets=prosodic_avg_features,
        textual_feature_sets=text_features,
        best_prosodic_models=best_avg_prosodic_models,
        best_textual_models=best_textual_models,
        dataset_name="MULTIMODAL AVG PROSODIC + TEXT",
        output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
        test_size=0.2,
        random_state=42,
        save_path="results/multimodal_modeling_results.csv"
    )
    logger.info(f"Multimodal modeling completed in {time.time() - start:.2f} seconds.")

    logger.info(f"Total execution time: {time.time() - start_total:.2f} seconds")
    logger.info("All modeling finished successfully!")


# ==========================================================
# Execute Pipeline
# ==========================================================
if __name__ == "__main__":
    main_pipeline()
