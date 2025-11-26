# ==========================================================
# modeling.py
# ==========================================================

import logging
import time
from typing import Any, Dict, List, Tuple, Union
import re
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

# def get_best_model(results_df: pd.DataFrame, output_var: Union[str, List[str]], model_type: str = "DT") -> Dict[str, Any]:
#     """
#     Returns the best model result row for a given output and model type
#     based on highest R² and lowest Absolute Relative Error.
#     """
#     model_col = f"{model_type}_Test_R2"
#     filtered = results_df[results_df["Output_Variable"] == str(output_var)]

#     if filtered.empty:
#         return None

#     best_row = filtered.sort_values(
#         by=[model_col, f"{model_type}_Absolute_Relative_Error"],
#         ascending=[False, True]
#     ).iloc[0]

#     multi_output = isinstance(output_var, list) or ("Overall" in str(output_var) and "Excited" in str(output_var))

#     return {
#         "dataset": best_row["Dataset"],
#         "features": best_row.get("Feature_List", None),
#         "feature_selector": best_row["Feature_Selector"],
#         "num_features": best_row["Num_Features"],
#         "output": best_row["Output_Variable"],
#         "model_type": model_type,
#         "multi_output": multi_output,
#         "cv_r2": best_row[f"{model_type}_CV_R2"],
#         "test_r2": best_row[f"{model_type}_Test_R2"],
#         "absolute_relative_error": best_row[f"{model_type}_Absolute_Relative_Error"],
#         "params": best_row[f"{model_type}_Best_Params"]
#     }

from typing import Union, List, Dict, Any
import pandas as pd
import numpy as np

def get_best_model(results_df: pd.DataFrame, 
                   output_var: Union[str, List[str]], 
                   model_type: str = "DT") -> Dict[str, Any]:
    """
    Returns the best model result row for a given output and model type
    based on highest R² and lowest Absolute Relative Error.

    This version auto-cleans all fields so outputs are CSV/table friendly.
    """

    model_col = f"{model_type}_Test_R2"
    filtered = results_df[results_df["Output_Variable"] == str(output_var)]

    if filtered.empty:
        return None

    best_row = filtered.sort_values(
        by=[model_col, f"{model_type}_Absolute_Relative_Error"],
        ascending=[False, True]
    ).iloc[0]

    # ----- Determine multi-output -----
    multi_output = isinstance(output_var, list) or (
        "Overall" in str(output_var) and "Excited" in str(output_var)
    )

    # ----- Helpers to clean values -----
    def clean_value(val):
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        if isinstance(val, list):
            return "; ".join(map(str, val))
        if isinstance(val, dict):
            return "; ".join(f"{k}={v}" for k, v in val.items())
        return val

    # ----- Construct clean output dictionary -----
    return {
        "dataset": best_row["Dataset"],
        "features": clean_value(best_row.get("Feature_List", None)),
        "feature_selector": clean_value(best_row["Feature_Selector"]),
        "num_features": int(best_row["Num_Features"]),
        "output": str(best_row["Output_Variable"]),
        "model_type": model_type,
        "multi_output": multi_output,
        "cv_r2": clean_value(best_row[f"{model_type}_CV_R2"]),
        "test_r2": clean_value(best_row[f"{model_type}_Test_R2"]),
        "absolute_relative_error": clean_value(best_row[f"{model_type}_Absolute_Relative_Error"]),
        "params": clean_value(best_row[f"{model_type}_Best_Params"])
    }


# ==========================================================
# Model Training Functions
# ==========================================================

def train_decision_tree(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[GridSearchCV, float, float, float]:

    # Base Decision Tree
    dt_base = DecisionTreeRegressor(random_state=42)

    # Wrap in MultiOutputRegressor only if multi-output
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(dt_base)
        param_grid = {
            "estimator__max_depth": [3, 5, 10, None],
            "estimator__min_samples_split": [2, 5, 10]
        }
    else:
        model = dt_base
        param_grid = {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        }

    # Wrap in a pipeline (needed if using GridSearchCV with MultiOutput)
    pipeline = Pipeline([
        ("model", model)
    ])

    # Update param grid to use pipeline step name
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        grid_param = {f"model__{k}": v for k, v in param_grid.items()}
    else:
        grid_param = {f"model__{k}": v for k, v in param_grid.items()}

    # GridSearchCV
    grid = GridSearchCV(
        pipeline,
        grid_param,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    cv_score = grid.best_score_
    test_score = grid.score(X_test, y_test)
    abs_rel_err = absolute_relative_error(y_test.values if hasattr(y_test, "values") else y_test,
                                          grid.predict(X_test))

    logger.info(f"DT -> CV R^2: {cv_score:.4f}, Test R^2: {test_score:.4f}, AbsRelError: {abs_rel_err:.4f}")
    logger.info(f"Best params: {grid.best_params_}")

    return grid, cv_score, test_score, abs_rel_err


def train_mlp(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[GridSearchCV, float, float, float]:
    
    # Base model
    mlp_base = MLPRegressor(random_state=42, max_iter=1000)

    # Wrap base in MultiOutputRegressor if multi-output
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        model = MultiOutputRegressor(mlp_base)
        param_grid = {
            "estimator__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "estimator__activation": ["relu", "tanh"],
            "estimator__alpha": [0.0001, 0.001, 0.01]
        }
        # Wrap in pipeline for scaling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", model)
        ])
        grid_param = {f"mlp__{k}": v for k, v in param_grid.items()}
    else:
        # Single-output
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp_base)
        ])
        param_grid = {
            "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [0.0001, 0.001, 0.01]
        }
        grid_param = param_grid

    # GridSearchCV
    grid = GridSearchCV(
        pipeline,
        grid_param,
        cv=5,
        n_jobs=-1,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    cv_score = grid.best_score_
    test_score = grid.score(X_test, y_test)
    abs_rel_err = absolute_relative_error(
        y_test.values if hasattr(y_test, "values") else y_test,
        grid.predict(X_test)
    )

    logger.info(f"MLP -> CV R^2: {cv_score:.4f}, Test R^2: {test_score:.4f}, AbsRelError: {abs_rel_err:.4f}")
    logger.info(f"Best params: {grid.best_params_}")

    return grid, cv_score, test_score, abs_rel_err


# ==========================================================
# Main Modeling Pipeline
# ==========================================================

def run_modeling(
    df: pd.DataFrame,
    feature_sets: pd.DataFrame,
    dataset_name: str,
    output_vars: List[Any] = ['Overall', 'Excited', ['Overall', 'Excited']],
    test_size: float = 0.2,
    random_state: int = 42,
    results_save_path: str = None,
    best_model_save_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Runs Decision Tree and MLP models for all feature sets and targets.
    Returns full results and best models dictionary.
    """

    total_results_df = pd.DataFrame()
    for _, row in feature_sets.iterrows():
        feature_selector = row['Method']
        k = row['K']
        raw_features  = row['Features']
        if isinstance(raw_features, str):
            feature_names = re.findall(r"'([^']+)'", raw_features)
        else:
            feature_names = raw_features

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

    if results_save_path:
        total_results_df.to_csv(results_save_path, index=False)
        logger.info(f"Results saved to {results_save_path}")

    best_models = []

    for model_type in ["DT", "MLP"]:
        for output in ["Overall", "Excited", ["Overall", "Excited"]]:

            best_model = get_best_model(total_results_df, output, model_type)

            if best_model is not None:
                best_models.append(best_model)

    # Convert to DataFrame
    best_models_df = pd.DataFrame(best_models)

    # Ensure clean column order
    column_order = [
        "dataset", "model_type", "output", "multi_output",
        "features", "feature_selector", "num_features",
        "cv_r2", "test_r2", "absolute_relative_error", "params"
    ]
    best_models_df = best_models_df[column_order]

    # Save
    if best_model_save_path:
        best_models_df.to_csv(best_model_save_path, index=False)
        logger.info(f"Best models saved to {best_model_save_path}")

    return total_results_df, best_models_df



# ==========================================================
# Standalone Execution
# ==========================================================
if __name__ == "__main__":
    df_example = pd.read_csv("../data/avg_cleaned_prosodic_features.csv")
    from feature_selection import run_feature_selection

    features = run_feature_selection(df_example, dataset_name="AVERAGED PROSODIC")
    results, best_models = run_modeling(df_example, features, dataset_name="AVERAGED PROSODIC")
    logger.info(f"Modeling complete. Best models: {best_models}")
