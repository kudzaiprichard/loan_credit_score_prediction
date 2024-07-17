"""Model zoo + hyperparameter search spaces.

Every classifier is configured to respect the ~85/15 class imbalance via
class weights (or scale_pos_weight for XGBoost). Search spaces are deliberately
compact so tuning finishes in minutes; widen them for a production sweep.

Parameter keys are prefixed `model__` because they are searched over the FULL
Pipeline (clean -> engineer -> preprocess -> model).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from . import config

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # noqa: F401
    # LightGBM 4.5 calls sklearn's check_X_y(force_all_finite=...), a kwarg removed
    # in scikit-learn >=1.6 -> it crashes at fit time. Disable until versions align
    # (upgrade lightgbm, or pin scikit-learn<1.6). Training stays resilient either way.
    import sklearn as _sk
    _HAS_LGBM = tuple(int(x) for x in _sk.__version__.split(".")[:2]) < (1, 6)
except Exception:  # pragma: no cover
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:  # pragma: no cover
    _HAS_CAT = False

RS = config.RANDOM_STATE


def scale_pos_weight(y) -> float:
    """neg/pos ratio used by gradient boosters to up-weight the minority class."""
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg) / float(pos) if pos else 1.0


def get_model_zoo(y_train) -> dict:
    """Return {name: {'estimator': est, 'params': search_space}}.

    Each estimator is a *bare* classifier; `make_pipeline` wraps it with the
    shared preprocessing.
    """
    spw = scale_pos_weight(y_train)
    zoo: dict = {}

    zoo["logistic_regression"] = {
        "estimator": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RS, n_jobs=-1
        ),
        "params": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
        },
    }

    zoo["knn"] = {
        "estimator": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "model__n_neighbors": [15, 25, 51, 75],
            "model__weights": ["uniform", "distance"],
        },
    }

    zoo["random_forest"] = {
        "estimator": RandomForestClassifier(
            class_weight="balanced_subsample", random_state=RS, n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 5, 20],
            "model__max_features": ["sqrt", 0.5],
        },
    }

    zoo["hist_gradient_boosting"] = {
        "estimator": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RS
        ),
        "params": {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [None, 6, 10],
            "model__max_leaf_nodes": [31, 63],
            "model__l2_regularization": [0.0, 1.0],
        },
    }

    if _HAS_XGB:
        zoo["xgboost"] = {
            "estimator": XGBClassifier(
                n_estimators=400,
                tree_method="hist",
                eval_metric="logloss",
                scale_pos_weight=spw,
                random_state=RS,
                n_jobs=-1,
            ),
            "params": {
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [1.0, 5.0],
            },
        }

    if _HAS_LGBM:
        zoo["lightgbm"] = {
            "estimator": LGBMClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RS,
                n_jobs=-1,
                verbose=-1,
            ),
            "params": {
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        }

    if _HAS_CAT:
        zoo["catboost"] = {
            "estimator": CatBoostClassifier(
                iterations=500,
                auto_class_weights="Balanced",
                random_state=RS,
                verbose=0,
                allow_writing_files=False,
            ),
            "params": {
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.1],
                "model__l2_leaf_reg": [3.0, 7.0],
            },
        }

    return zoo


def build_ensembles(fitted_bare_estimators: dict) -> dict:
    """Build soft-voting and stacking ensembles from already-tuned bare models.

    `fitted_bare_estimators` maps name -> estimator (cloned, unfitted is fine).
    Returns {name: bare_ensemble_estimator} ready to wrap with make_pipeline.
    """
    from sklearn.base import clone

    members = [
        (name, clone(est))
        for name, est in fitted_bare_estimators.items()
        if name not in {"knn"}  # KNN is a weak/slow ensemble member here
    ]
    if len(members) < 2:
        return {}

    ensembles = {
        "voting_soft": VotingClassifier(estimators=members, voting="soft", n_jobs=-1),
        "stacking": StackingClassifier(
            estimators=members,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=3,
            n_jobs=-1,
            passthrough=False,
        ),
    }
    return ensembles
