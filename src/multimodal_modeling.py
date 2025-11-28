# ==========================================================
# multimodal_modeling.py
# ==========================================================

import logging
import re
from typing import Any, Dict, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.modeling import train_decision_tree, train_mlp

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================================
# Helper: Rebuild Model from Parameters
# ==========================================================
def build_model_from_params(model_info: Dict[str, Any]) -> Any:
    """
    Reconstructs sklearn model using stored best parameters.
    """
    model_type = model_info["model_type"]
    params = model_info["params"]
    is_multi = model_info["multi_output"]

    if model_type == "DT":
        base = DecisionTreeRegressor(**params, random_state=42)
        if is_multi:
            return MultiOutputRegressor(base)
        return base

    elif model_type == "MLP":
        base = MLPRegressor(**params, max_iter=1000, random_state=42)
        if is_multi:
            return Pipeline([("scaler", StandardScaler()), ("mlp", MultiOutputRegressor(base))])
        return Pipeline([("scaler", StandardScaler()), ("mlp", base)])

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ==========================================================
# Main Multimodal Modeling Function
# ==========================================================
def run_multimodal_modeling(
    prosodic_df: pd.DataFrame,
    textual_df: pd.DataFrame,
    prosodic_feature_sets: List[tuple],
    textual_feature_sets: List[tuple],
    best_prosodic_models: Dict[str, Dict[str, Any]],
    best_textual_models: Dict[str, Dict[str, Any]],
    dataset_name: str,
    output_vars: List[Union[str, List[str]]] = ['Overall', 'Excited', ['Overall', 'Excited']],
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Runs multimodal modeling with:
    1) Feature Fusion
    2) Model Fusion (Ensemble Averaging)

    Returns a DataFrame of results.
    """
    all_results = []

    # ================================
    # FEATURE FUSION
    # ================================
    for _, row in prosodic_feature_sets.iterrows():
        p_selector = row['Method']
        p_k = row['K']
        p_raw_features  = row['Features']
        if isinstance(p_raw_features, str):
            p_features = re.findall(r"'([^']+)'", p_raw_features)
        else:
            p_features = p_raw_features
            
        for _, row in textual_feature_sets.iterrows():
            t_selector = row['Method']
            t_k = row['K']
            t_raw_features  = row['Features']
            if isinstance(t_raw_features, str):
                t_features = re.findall(r"'([^']+)'", t_raw_features)
            else:
                t_features = t_raw_features
            
            X_fused = pd.concat([prosodic_df[p_features], textual_df[t_features]], axis=1)

            for output_var in output_vars:
                logger.info(f"\n=== MULTIMODAL FEATURE FUSION ===")
                logger.info(f"Prosodic: {p_selector} ({p_k}) | Textual: {t_selector} ({t_k}) | Target: {output_var}")

                y = prosodic_df[output_var]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_fused, y, test_size=test_size, random_state=random_state
                )

                # Train Decision Tree
                dt_model, dt_cv, dt_test, dt_abs_rel = train_decision_tree(X_train, X_test, y_train, y_test)
                # Train MLP
                mlp_model, mlp_cv, mlp_test, mlp_abs_rel = train_mlp(X_train, X_test, y_train, y_test)

                all_results.append({
                    "Dataset": dataset_name,
                    "Fusion_Type": "Feature",
                    "Prosodic_Selector": p_selector,
                    "Text_Selector": t_selector,
                    "Prosodic_k": p_k,
                    "Text_k": t_k,
                    "Output": str(output_var),
                    "DT_CV_R2": dt_cv,
                    "DT_Test_R2": dt_test,
                    "DT_Best_Params": dt_model.best_params_,
                    "MLP_CV_R2": mlp_cv,
                    "MLP_Test_R2": mlp_test,
                    "MLP_Best_Params": mlp_model.best_params_
                })

    # ================================
    # MODEL FUSION (ENSEMBLE AVERAGING)
    # ================================
    for output_var in output_vars:
        logger.info(f"\n=== MULTIMODAL MODEL FUSION === | Target: {output_var}")

        pros_info = best_prosodic_models[str(output_var)]
        text_info = best_textual_models[str(output_var)]

        Xp_train, Xp_test, y_train, y_test = train_test_split(
            prosodic_df[pros_info["features"]],
            prosodic_df[output_var],
            test_size=test_size,
            random_state=random_state
        )
        Xt_train, Xt_test, _, _ = train_test_split(
            textual_df[text_info["features"]],
            prosodic_df[output_var],
            test_size=test_size,
            random_state=random_state
        )

        pros_model = build_model_from_params(pros_info)
        text_model = build_model_from_params(text_info)

        pros_model.fit(Xp_train, y_train)
        text_model.fit(Xt_train, y_train)

        y_pred = (pros_model.predict(Xp_test) + text_model.predict(Xt_test)) / 2
        ensemble_r2 = r2_score(y_test, y_pred)

        all_results.append({
            "Dataset": dataset_name,
            "Fusion_Type": "Model",
            "Prosodic_Model_Type": pros_info["model_type"],
            "Textual_Model_Type": text_info["model_type"],
            "Output": str(output_var),
            "Ensemble_Strategy": "Averaging",
            "Test_R2": ensemble_r2,
            "Prosodic_Params": pros_info["params"],
            "Textual_Params": text_info["params"]
        })

    results_df = pd.DataFrame(all_results)

    if save_path:
        results_df.to_csv(save_path, index=False)
        logger.info(f"\n Multimodal results saved to: {save_path}")

    return results_df


# ==========================================================
# Standalone Execution
# ==========================================================
if __name__ == "__main__":
    from feature_selection import run_feature_selection

    pros_df = pd.read_csv("../data/avg_cleaned_prosodic_features.csv")
    text_df = pd.read_csv("../data/text_cleaned_features.csv")

    pros_features = run_feature_selection(pros_df, dataset_name="AVERAGED PROSODIC")
    text_features = run_feature_selection(text_df, dataset_name="TEXT FEATURES")

    # Mock best model dicts for demo
    best_pros = { "Overall": pros_features[0], "Excited": pros_features[0], "['Overall', 'Excited']": pros_features[0] }
    best_text = { "Overall": text_features[0], "Excited": text_features[0], "['Overall', 'Excited']": text_features[0] }

    results = run_multimodal_modeling(
        prosodic_df=pros_df,
        textual_df=text_df,
        prosodic_feature_sets=pros_features,
        textual_feature_sets=text_features,
        best_prosodic_models=best_pros,
        best_textual_models=best_text,
        dataset_name="MULTIMODAL AVG PROSODIC + TEXT"
    )
    logger.info("Multimodal modeling complete.")
