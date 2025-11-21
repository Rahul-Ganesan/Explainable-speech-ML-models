# ==========================================================
# feature_selection.py
# ==========================================================

import logging
import time
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================================
# Correlation Utilities
# ==========================================================

def get_top_corr_pairs(corr_matrix: pd.DataFrame, top_n: int = 5, most_positive: bool = True) -> pd.Series:
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_pairs = corr_matrix.where(mask).stack()
    sorted_corr = corr_pairs.sort_values(ascending=not most_positive)
    return sorted_corr.head(top_n)


def print_corr_with_target(corr_matrix: pd.DataFrame, target: str, top_n: int = 5):
    series = corr_matrix[target].drop(labels=[target])
    logger.info(f"\nTop {top_n} positively correlated with '{target}':")
    for feat, val in series.sort_values(ascending=False).head(top_n).items():
        logger.info(f"{feat}: {val:.4f}")
    logger.info(f"\nTop {top_n} negatively correlated with '{target}':")
    for feat, val in series.sort_values().head(top_n).items():
        logger.info(f"{feat}: {val:.4f}")


# ==========================================================
# Feature Selection Utilities
# ==========================================================

mi = partial(mutual_info_regression, random_state=42, discrete_features=False)


def get_selected_features(X_train: pd.DataFrame, y_train: pd.DataFrame, k: int, score_func) -> Tuple[pd.Index, pd.Index]:
    """Runs SelectKBest separately for each output."""
    pipe_inner = Pipeline([
        ('kbst', SelectKBest(score_func, k=k)),
        ('regr', Ridge())
    ])
    pipe_outer = Pipeline([
        ('multi', MultiOutputRegressor(pipe_inner))
    ])
    pipe_outer.fit(X_train, y_train)
    feats_output_1 = pipe_outer.named_steps['multi'].estimators_[0].named_steps['kbst'].get_support(indices=True)
    feats_output_2 = pipe_outer.named_steps['multi'].estimators_[1].named_steps['kbst'].get_support(indices=True)
    return X_train.columns[feats_output_1], X_train.columns[feats_output_2]


def get_selected_features_aggregate(X_train: pd.DataFrame, y_train: pd.DataFrame, k: int, score_func) -> List[str]:
    """Aggregates MI scores across both targets using mean."""
    pipe_inner = Pipeline([
        ('kbst', SelectKBest(score_func, k=k)),
        ('regr', Ridge())
    ])
    pipe_outer = Pipeline([
        ('multi', MultiOutputRegressor(pipe_inner))
    ])
    pipe_outer.fit(X_train, y_train)

    scores_0 = pipe_outer.named_steps['multi'].estimators_[0].named_steps['kbst'].scores_
    scores_1 = pipe_outer.named_steps['multi'].estimators_[1].named_steps['kbst'].scores_

    scores = (np.nan_to_num(scores_0) + np.nan_to_num(scores_1)) / 2
    combined_score = minmax_scale(scores)
    top_k_idx = np.argsort(combined_score)[-k:][::-1]
    return list(X_train.columns[top_k_idx])


def find_intersection(list1: List[str], list2: List[str]) -> List[str]:
    return list(set(list1).intersection(set(list2)))


def find_union(list1: List[str], list2: List[str]) -> List[str]:
    return list(set(list1).union(set(list2)))


# ==========================================================
# Main Feature Selection Engine
# ==========================================================

def run_feature_selection(
    df: pd.DataFrame,
    dataset_name: str,
    target_cols: List[str] = ['Overall', 'Excited'],
    k_values: List[int] = [5, 10, 15, 20, 25, 30],
    test_size: float = 0.20,
    random_state: int = 42,
    print_corr: bool = False
) -> List[Tuple[str, int, List[str]]]:
    """
    Runs full feature selection pipeline on any dataset.
    Returns a list of tuples: (method, k, feature_list)
    """
    feature_sets: List[Tuple[str, int, List[str]]] = []

    if print_corr:
        corr_matrix = df.corr(method='pearson')
        for target in target_cols:
            print_corr_with_target(corr_matrix, target)

    X = df.drop(columns=target_cols)
    y = df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    for k in k_values:
        logger.info(f"[{dataset_name}] === k = {k} ===")
        start = time.time()

        # Mutual Information
        mi_overall, mi_excited = get_selected_features(X_train, y_train, k, mi)
        mi_agg = get_selected_features_aggregate(X_train, y_train, k, mi)
        mi_inter = find_intersection(list(mi_overall), list(mi_excited))
        mi_union = find_union(list(mi_overall), list(mi_excited))

        # Random Forest
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_top_k = X_train.columns[np.argsort(rf_model.feature_importances_)[-k:]].tolist()

        # Store results
        feature_sets.extend([
            ("MI Overall", k, list(mi_overall)),
            ("MI Excited", k, list(mi_excited)),
            ("MI Intersection", k, mi_inter),
            ("MI Union", k, mi_union),
            ("MI Aggregate", k, mi_agg),
            ("Random Forest", k, rf_top_k)
        ])

        # Logging for tracking
        logger.info(f"MI Overall: {sorted(mi_overall)}")
        logger.info(f"MI Excited: {sorted(mi_excited)}")
        logger.info(f"MI Intersection: {sorted(mi_inter)}")
        logger.info(f"MI Union: {sorted(mi_union)}")
        logger.info(f"MI Aggregate: {sorted(mi_agg)}")
        logger.info(f"RF Selected: {sorted(rf_top_k)}")
        logger.info(f"Feature selection k={k} completed in {time.time()-start:.2f} seconds")

    return feature_sets


# ==========================================================
# Standalone Execution
# ==========================================================
if __name__ == "__main__":
    # Example usage
    df_example = pd.read_csv("../data/all_q_cleaned_prosodic_features.csv")
    features = run_feature_selection(df_example, dataset_name="ALL Q PROSODIC", print_corr=True)
    logger.info(f"Total feature sets generated: {len(features)}")
