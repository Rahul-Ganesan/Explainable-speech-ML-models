# ==========================================================
# main_pipeline_driver.py
# Full pipeline for Multimodal ML Project
# ==========================================================

import os
import sys
import time
import logging
from typing import Tuple, List, Dict
import yaml
import pandas as pd
import numpy as np

# -----------------------------
# Logging Setup
# -----------------------------
# Create results/logs folder if it doesn't exist
os.makedirs("results/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="results/logs/pipeline.log",  # Log file path
    filemode="w",                          # Overwrite each run; use "a" to append
    level=logging.INFO,                     # Minimum level to log
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

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
    if config.get("run_prosodic_data_cleaning", True):
        from src.prosodic_data_cleaning import process_prosodic_data

        logger.info("Cleaning prosodic data...")
        start = time.time()
        cleaned_prosodic_allq, cleaned_prosodic_avg = process_prosodic_data(
            prosodic_path="data/prosodic_features.csv",
            score_path="data/scores.csv",
            output_allq_path="results/all_q_cleaned_prosodic_features.csv",
            output_avg_path="results/avg_cleaned_prosodic_features.csv"
        )
        logger.info(f"Prosodic data cleaned in {time.time() - start:.2f} seconds.")
    else:
        # Load pre-cleaned prosodic data
        logger.info("Cleaned prosodic data...")
        cleaned_prosodic_allq = pd.read_csv("results/all_q_cleaned_prosodic_features.csv")
        cleaned_prosodic_avg = pd.read_csv("results/avg_cleaned_prosodic_features.csv")
        logger.info("Cleaned prosodic data loaded.")
    
    # -----------------------------
    # Clean Textual Data
    # -----------------------------
    if config.get("run_textual_data_cleaning", True):
        from src.textual_data_cleaning import process_textual_data

        logger.info("Cleaning textual data...")
        start = time.time()
        cleaned_textual = process_textual_data(
            transcripts_path="data/transcripts.csv",
            score_path="data/scores.csv",
            output_textual_path="results/text_cleaned_features.csv"
        )
        logger.info(f"Textual data cleaned in {time.time() - start:.2f} seconds.")
    else:
        # Load pre-cleaned textual data
        logger.info("Loading pre-cleaned textual data...")
        cleaned_textual = pd.read_csv("results/text_cleaned_features.csv")
        logger.info("Cleaned textual data loaded.")
        

    # -----------------------------
    # Feature Selection
    # -----------------------------
    logger.info("Starting feature selection...")
    from src.feature_selection import run_feature_selection

    # Prosodic Averaged Question Feature Selection
    if config.get("run_avg_q_prosodic_feature_selection", True):
        logger.info("Running averaged question prosodic feature selection...")
        start = time.time()
        prosodic_avg_features_df = run_feature_selection(cleaned_prosodic_avg, dataset_name="AVERAGED PROSODIC", output_path="results/selected_features_avg_q_prosodic.csv")
        logger.info(f"Averaged question Feature selection completed in {time.time() - start:.2f} seconds.")
    else:
        prosodic_avg_features_df = pd.read_csv("results/selected_features_avg_q_prosodic.csv")
    
    # Prosodic All Question Feature Selection
    if config.get("run_all_q_prosodic_feature_selection", True):        
        logger.info("Running all question prosodic feature selection...")
        start = time.time()
        prosodic_all_features_df = run_feature_selection(cleaned_prosodic_allq, dataset_name="ALL Q PROSODIC", output_path="results/selected_features_all_q_prosodic.csv")
        logger.info(f"All question Feature selection completed in {time.time() - start:.2f} seconds.")
    else:
        prosodic_all_features_df = pd.read_csv("results/selected_features_all_q_prosodic.csv")
    
    # Textual Feature Selection
    if config.get("run_textual_feature_selection", True):        
        logger.info("Running textual feature selection...")
        start = time.time()
        text_features_df = run_feature_selection(cleaned_textual, dataset_name="TEXT FEATURES", output_path="results/selected_features_textual.csv")
        logger.info(f"Textual Feature selection completed in {time.time() - start:.2f} seconds.")
    else:
        text_features_df = pd.read_csv("results/selected_features_textual.csv")

    logger.info("Feature selection completed.")
    
    # -----------------------------
    # Independent Modeling
    # -----------------------------
    logger.info("Starting independent modeling...")
    from src.modeling import run_modeling
        
    # Prosodic Averaged Question Modeling
    if config.get("run_avg_q_prosodic_model_training", True):
        logger.info("Modeling on prosodic averaged dataset...")
        start = time.time()
        avg_prosodic_results_df, best_avg_prosodic_models_df = run_modeling(
            df=cleaned_prosodic_avg,
            feature_sets=prosodic_avg_features_df,
            dataset_name="AVERAGED PROSODIC",
            output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
            test_size=0.2,
            random_state=42,
            results_save_path="results/avg_prosodic_modeling_results.csv",
            best_model_save_path="results/best_models_averaged_prosodic.csv"
        )
        logger.info(f"Averaged prosodic modeling completed in {time.time() - start:.2f} seconds.")
    else:
        logger.info("Loading pre-computed averaged prosodic modeling results...")
        avg_prosodic_results_df = pd.read_csv("results/avg_prosodic_modeling_results.csv")
        best_avg_prosodic_models_df = pd.read_csv("results/best_models_averaged_prosodic.csv")
        logger.info("Averaged prosodic modeling results loaded.")

    # Prosodic All Question Modeling
    if config.get("run_prosodic_model_training", True):
        logger.info("Modeling on prosodic all-question dataset...")
        start = time.time()
        all_q_prosodic_results_df, best_all_q_models_df = run_modeling(
            df=cleaned_prosodic_allq,
            feature_sets=prosodic_all_features_df,
            dataset_name="ALL Q PROSODIC",
            output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
            test_size=0.2,
            random_state=42,
            results_save_path="results/all_q_prosodic_modeling_results.csv",
            best_model_save_path="results/best_models_all_q_prosodic.csv"
        )
        logger.info(f"All-question prosodic modeling completed in {time.time() - start:.2f} seconds.")
    else:
        logger.info("Loading pre-computed all-question prosodic modeling results...")
        all_q_prosodic_results_df = pd.read_csv("results/all_q_prosodic_modeling_results.csv")
        best_all_q_models_df = pd.read_csv("results/best_models_all_q_prosodic.csv")
        logger.info("All-question prosodic modeling results loaded.")

    # Textual Modeling
    if config.get("run_textual_model_training", True):
        logger.info("Modeling on textual dataset...")
        start = time.time()
        textual_results_df, best_textual_models_df = run_modeling(
            df=cleaned_textual,
            feature_sets=text_features_df,
            dataset_name="TEXT FEATURES",
            output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
            test_size=0.2,
            random_state=42,
            results_save_path="results/textual_modeling_results.csv",
            best_model_save_path="results/best_models_textual.csv"
        )
        logger.info(f"Textual modeling completed in {time.time() - start:.2f} seconds.")
        
    else:
        logger.info("Loading pre-computed textual modeling results...")
        textual_results_df = pd.read_csv("results/textual_modeling_results.csv")
        best_textual_models_df = pd.read_csv("results/best_models_textual.csv")
        logger.info("Textual modeling results loaded.")
        
    logger.info("Independent modeling completed.")

    # -----------------------------
    # Multimodal Modeling
    # -----------------------------
    from src.multimodal_modeling import run_multimodal_modeling

    if config.get("run_multimodal_model_training", True):
        logger.info("Running multimodal modeling (prosodic + textual)...")
        start = time.time()
        multimodal_results = run_multimodal_modeling(
            prosodic_df=cleaned_prosodic_avg,
            textual_df=cleaned_textual,
            prosodic_feature_sets=prosodic_avg_features_df,
            textual_feature_sets=text_features_df,
            best_prosodic_models=best_avg_prosodic_models_df,
            best_textual_models=best_textual_models_df,
            dataset_name="MULTIMODAL AVG PROSODIC + TEXT",
            output_vars=['Overall', 'Excited', ['Overall', 'Excited']],
            test_size=0.2,
            random_state=42,
            save_path="results/multimodal_modeling_results.csv"
        )
        logger.info(f"Multimodal modeling completed in {time.time() - start:.2f} seconds.")
    else:
        logger.info("Loading pre-computed multimodal modeling results...")
        multimodal_results = pd.read_csv("results/multimodal_modeling_results.csv")
        logger.info("Multimodal modeling results loaded.")

    logger.info(f"Total execution time: {time.time() - start_total:.2f} seconds")
    logger.info("All modeling finished successfully!")


# ==========================================================
# Execute Pipeline
# ==========================================================
if __name__ == "__main__":
    main_pipeline()
