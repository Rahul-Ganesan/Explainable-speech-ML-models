# ==========================================================
# modeling.py
# ==========================================================

import logging
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================================
# Utility Functions
# ==========================================================

def absolute_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Relative Error."""
    abs_rel_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)
    return np.mean(abs_rel_error)

def get_best_model(results_df: pd.DataFrame, output_var: Union[str, List[str]], model_type: str = "DT") -> Dict[str, Any]:
    """
    Returns the best model result row for a given output and model type
    based on highest R² and lowest Absolute Relative Error.
    """
    model_col = f"{model_type}_Test_R2"
    filtered = results_df[results_df["Output_Variable"] == str(output_var)]

    if filtered.empty:
        return None

    best_row = filtered.sort_values(
        by=[model_col, f"{model_type}_Absolute_Relative_Error"],
        ascending=[False, True]
    ).iloc[0]

    multi_output = isinstance(output_var, list) or ("Overall" in str(output_var) and "Excited" in str(output_var))

    return {
        "dataset": best_row["Dataset"],
        "features": best_row.get("Feature_List", None),
        "feature_selector": best_row["Feature_Selector"],
        "num_features": best_row["Num_Features"],
        "output": best_row["Output_Variable"],
        "model_type": model_type,
        "multi_output": multi_output,
        "cv_r2": best_row[f"{model_type}_CV_R2"],
        "test_r2": best_row[f"{model_type}_Test_R2"],
        "absolute_relative_error": best_row[f"{model_type}_Absolute_Relative_Error"],
        "params": best_row[f"{model_type}_Best_Params"]
    }

# ==========================================================
# Model Training Functions
# ==========================================================

def train_decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[GridSearchCV, float, float, float]:
    dt_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    }

    model = DecisionTreeRegressor(random_state=42)
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(model)

    grid = GridSearchCV(model, dt_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    cv_score = grid.best_score_
    test_score = grid.score(X_test, y_test)
    abs_rel_err = absolute_relative_error(y_test.values if hasattr(y_test, "values") else y_test, grid.predict(X_test))

    logger.info(f"DT -> CV R²: {cv_score:.4f}, Test R²: {test_score:.4f}, AbsRelError: {abs_rel_err:.4f}")
    logger.info(f"Best params: {grid.best_params_}")

    return grid, cv_score, test_score, abs_rel_err

def train_mlp(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[GridSearchCV, float, float, float]:
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MultiOutputRegressor(MLPRegressor(max_iter=1000, random_state=42)))
        ])
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(max_iter=1000, random_state=42))
        ])

    mlp_grid = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01]
    }

    grid = GridSearchCV(pipeline, mlp_grid, cv=5, n_jobs=-1, error_score="raise")
    grid.fit(X_train, y_train)

    cv_score = grid.best_score_
    test_score = grid.score(X_test, y_test)
    abs_rel_err = absolute_relative_error(y_test.values if hasattr(y_test, "values") else y_test, grid.predict(X_test))

    logger.info(f"MLP -> CV R²: {cv_score:.4f}, Test R²: {test_score:.4f}, AbsRelError: {abs_rel_err:.4f}")
    logger.info(f"Best params: {grid.best_params_}")

    return grid, cv_score, test_score, abs_rel_err

# ==========================================================
# Main Modeling Pipeline
# ==========================================================

def run_modeling(
    df: pd.DataFrame,
    feature_sets: List[Tuple[str, int, List[str]]],
    dataset_name: str,
    output_vars: List[Any] = ['Overall', 'Excited', ['Overall', 'Excited']],
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Runs Decision Tree and MLP models for all feature sets and targets.
    Returns full results and best models dictionary.
    """

    total_results_df = pd.DataFrame()

    for feature_selector, k, feature_names in feature_sets:
        if len(feature_names) == 0:
            continue

        for output_var in output_vars:
            logger.info(f"\n--- DATASET: {dataset_name} | Features: {feature_selector} | k={k} | Target: {output_var}")

            X = df[feature_names]
            y = df[output_var]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            dt_model, dt_cv_r2, dt_test_r2, dt_abs_rel_err = train_decision_tree(X_train, X_test, y_train, y_test)
            mlp_model, mlp_cv_r2, mlp_test_r2, mlp_abs_rel_err = train_mlp(X_train, X_test, y_train, y_test)

            row = {
                "Dataset": dataset_name,
                "Feature_Selector": feature_selector,
                "Feature_List": feature_names,
                "Num_Features": len(feature_names),
                "Output_Variable": str(output_var),

                "DT_CV_R2": dt_cv_r2,
                "DT_Test_R2": dt_test_r2,
                "DT_Absolute_Relative_Error": dt_abs_rel_err,
                "DT_Best_Params": str(dt_model.best_params_),

                "MLP_CV_R2": mlp_cv_r2,
                "MLP_Test_R2": mlp_test_r2,
                "MLP_Absolute_Relative_Error": mlp_abs_rel_err,
                "MLP_Best_Params": str(mlp_model.best_params_)
            }

            total_results_df = pd.concat([total_results_df, pd.DataFrame([row])], ignore_index=True)

    if save_path:
        total_results_df.to_csv(save_path, index=False)
        logger.info(f"Results saved to {save_path}")

    best_models = {"DT": {}, "MLP": {}}
    for model_type in ["DT", "MLP"]:
        best_models[model_type]["Overall"] = get_best_model(total_results_df, "Overall", model_type)
        best_models[model_type]["Excited"] = get_best_model(total_results_df, "Excited", model_type)
        best_models[model_type]["Multi-Output"] = get_best_model(total_results_df, ["Overall", "Excited"], model_type)

    return total_results_df, best_models

# ==========================================================
# Standalone Execution
# ==========================================================
if __name__ == "__main__":
    df_example = pd.read_csv("../data/avg_cleaned_prosodic_features.csv")
    from feature_selection import run_feature_selection

    features = run_feature_selection(df_example, dataset_name="AVERAGED PROSODIC")
    results, best_models = run_modeling(df_example, features, dataset_name="AVERAGED PROSODIC")
    logger.info(f"Modeling complete. Best models: {best_models}")
