"""Model definitions and utilities."""

from typing import Union

from omegaconf import DictConfig
from pulearn import (
    BaggingPuClassifier,
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(
    cfg: DictConfig, verbose: bool = True
) -> Union[BaggingPuClassifier, ElkanotoPuClassifier, WeightedElkanotoPuClassifier]:
    """Create and return a PU classifier model.

    Creates a PU classifier based on the specified type in config, with
    either RandomForestClassifier or XGBClassifier as the base estimator.

    Args:
        cfg: Configuration dict containing model parameters under 'model' key.
            Required keys:
            - type: Model type ("bagging", "elkanoto", "weighted_elkanoto")
            - base_estimator_type: Base estimator type ("random_forest" or "xgboost")
            - base_estimator: Dict with estimator-specific parameters
            - <model_type>: Dict with model-specific parameters
                - bagging: pu_n_estimators
                - elkanoto: hold_out_ratio
                - weighted_elkanoto: hold_out_ratio
        verbose: Whether to print model configuration details.

    Returns:
        PU classifier instance (BaggingPuClassifier, ElkanotoPuClassifier, etc.)
        configured with the specified parameters.

    Raises:
        ValueError: If model type is not supported.
        KeyError: If required parameters are missing from config.
    """
    model_cfg = cfg.model
    model_type = model_cfg.type.lower()

    # Get base estimator type (default to random_forest for backward compatibility)
    base_estimator_type = getattr(
        model_cfg.base_estimator, "type", "random_forest"
    ).lower()

    if verbose:
        print(f"  -> Creating {model_type} PU classifier")

    # Get base estimator parameters
    base_estimator_cfg = model_cfg.base_estimator
    random_state = base_estimator_cfg.random_state

    # Create base estimator based on type
    if base_estimator_type == "random_forest":
        min_samples_split = getattr(base_estimator_cfg, "min_samples_split", 2)
        min_samples_leaf = getattr(base_estimator_cfg, "min_samples_leaf", 1)
        max_features = getattr(base_estimator_cfg, "max_features", "sqrt")
        criterion = getattr(base_estimator_cfg, "criterion", "gini")
        class_weight = getattr(base_estimator_cfg, "class_weight", None)

        if verbose:
            print(f"    - Base estimator (RandomForest):")
            print(f"      * n_estimators: {base_estimator_cfg.n_estimators}")
            print(f"      * max_depth: {base_estimator_cfg.max_depth}")
            print(f"      * min_samples_split: {min_samples_split}")
            print(f"      * min_samples_leaf: {min_samples_leaf}")
            print(f"      * max_features: {max_features}")
            print(f"      * criterion: {criterion}")
            if class_weight is not None:
                print(f"      * class_weight: {class_weight}")
            print(f"      * random_state: {random_state}")

        base_estimator = RandomForestClassifier(
            n_estimators=base_estimator_cfg.n_estimators,
            max_depth=base_estimator_cfg.max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )

    elif base_estimator_type == "xgboost":
        # XGBoost parameters
        learning_rate = getattr(base_estimator_cfg, "learning_rate", 0.1)
        subsample = getattr(base_estimator_cfg, "subsample", 1.0)
        colsample_bytree = getattr(base_estimator_cfg, "colsample_bytree", 1.0)
        reg_alpha = getattr(base_estimator_cfg, "reg_alpha", 0.0)
        reg_lambda = getattr(base_estimator_cfg, "reg_lambda", 1.0)
        min_child_weight = getattr(base_estimator_cfg, "min_child_weight", 1)
        gamma = getattr(base_estimator_cfg, "gamma", 0.0)

        if verbose:
            print(f"    - Base estimator (XGBoost):")
            print(f"      * n_estimators: {base_estimator_cfg.n_estimators}")
            print(f"      * max_depth: {base_estimator_cfg.max_depth}")
            print(f"      * learning_rate: {learning_rate}")
            print(f"      * subsample: {subsample}")
            print(f"      * colsample_bytree: {colsample_bytree}")
            print(f"      * min_child_weight: {min_child_weight}")
            print(f"      * reg_alpha: {reg_alpha}")
            print(f"      * reg_lambda: {reg_lambda}")
            print(f"      * gamma: {gamma}")
            print(f"      * random_state: {random_state}")

        base_estimator = XGBClassifier(
            n_estimators=base_estimator_cfg.n_estimators,
            max_depth=base_estimator_cfg.max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

    else:
        raise ValueError(
            f"Unsupported base estimator type: {base_estimator_type}. "
            f"Supported types: 'random_forest', 'xgboost'"
        )

    # Get model-specific parameters from the corresponding section
    model_specific_cfg = model_cfg[model_type]

    # Create PU classifier based on type
    if model_type == "bagging":
        if verbose:
            print(f"    - PU classifier parameters:")
            print(f"      * pu_n_estimators: {model_specific_cfg.pu_n_estimators}")
        pu_clf = BaggingPuClassifier(
            estimator=base_estimator,
            n_estimators=model_specific_cfg.pu_n_estimators,
            random_state=random_state,
        )
    elif model_type == "elkanoto":
        if verbose:
            print(f"    - PU classifier parameters:")
            print(f"      * hold_out_ratio: {model_specific_cfg.hold_out_ratio}")
        pu_clf = ElkanotoPuClassifier(
            estimator=base_estimator,
            hold_out_ratio=model_specific_cfg.hold_out_ratio,
            random_state=random_state,
        )
    elif model_type == "weighted_elkanoto":
        if verbose:
            print(f"    - PU classifier parameters:")
            print(f"      * hold_out_ratio: {model_specific_cfg.hold_out_ratio}")
        pu_clf = WeightedElkanotoPuClassifier(
            estimator=base_estimator,
            hold_out_ratio=model_specific_cfg.hold_out_ratio,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: 'bagging', 'elkanoto', 'weighted_elkanoto'"
        )

    if verbose:
        print(f"  -> Model created successfully")

    return pu_clf
