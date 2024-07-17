"""End-to-end training: tune a model zoo, build ensembles, pick a champion,
audit fairness, and persist a deployable artifact.

Run:
    python -m src.train            # full run
    python -m src.train --quick    # fast smoke run on a subsample

Leakage safety: clean/impute/encode all live inside each Pipeline and are fit
only on training folds. The decision threshold is tuned on cross-validated
out-of-fold predictions (never on the test set).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from . import config, evaluate, explain, fairness
from .cleaning import clean_features, prepare_target
from .data_loader import load_raw
from .models import build_ensembles, get_model_zoo
from .preprocessing import make_pipeline

SELECTION_METRIC = "roc_auc"   # threshold-free, stable ranking metric


def _grid_size(params: dict) -> int:
    n = 1
    for v in params.values():
        n *= len(v)
    return n


def _search(name, spec, X_train, y_train, cv, n_iter):
    pipe = make_pipeline(spec["estimator"])
    grid = _grid_size(spec["params"])
    rs = RandomizedSearchCV(
        pipe,
        spec["params"],
        n_iter=min(n_iter, grid),
        scoring=SELECTION_METRIC,
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=1,        # estimators already use all cores
        refit=True,
        error_score="raise",
    )
    print(f"  -> tuning {name} ({min(n_iter, grid)} candidates x {cv.get_n_splits()} folds)")
    rs.fit(X_train, y_train)
    return rs


def run(quick: bool = False):
    t0 = dt.datetime.now()
    print(f"[train] loading raw data ... (quick={quick})")
    raw = load_raw()
    raw = prepare_target(raw)               # builds binary target, drops bad rows
    y = raw[config.TARGET].to_numpy()
    X = raw.drop(columns=[config.TARGET])   # raw feature schema (pipeline cleans it)

    if quick:
        X, _, y, _ = train_test_split(
            X, y, train_size=20000, stratify=y, random_state=config.RANDOM_STATE
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"[train] train={len(X_train)} test={len(X_test)} "
          f"pos_rate_train={y_train.mean():.3f}")

    cv = StratifiedKFold(n_splits=3 if quick else 3, shuffle=True, random_state=config.RANDOM_STATE)
    n_iter = 4 if quick else 8

    # ---- tune each base model -------------------------------------------- #
    zoo = get_model_zoo(y_train)
    tuned, bare_best, comparison = {}, {}, []
    for name, spec in zoo.items():
        try:
            rs = _search(name, spec, X_train, y_train, cv, n_iter)
        except Exception as e:  # resilient: a broken/incompatible model is skipped
            print(f"  !! skipping {name}: {type(e).__name__}: {e}")
            continue
        tuned[name] = rs.best_estimator_
        bare_best[name] = clone(rs.best_estimator_.named_steps["model"])
        proba = rs.best_estimator_.predict_proba(X_test)[:, 1]
        m = evaluate.compute_metrics(y_test, proba, threshold=0.5)
        m["model"] = name
        m["cv_roc_auc"] = float(rs.best_score_)
        comparison.append(m)
        print(f"     {name}: test ROC-AUC={m['roc_auc']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} F1@0.5={m['f1']:.4f}")

    if not tuned:
        raise RuntimeError("No models trained successfully — check the environment.")

    # ---- ensembles -------------------------------------------------------- #
    for ens_name, ens in build_ensembles(bare_best).items():
        try:
            pipe = make_pipeline(ens)
            print(f"  -> fitting ensemble {ens_name}")
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(f"  !! skipping ensemble {ens_name}: {type(e).__name__}: {e}")
            continue
        tuned[ens_name] = pipe
        proba = pipe.predict_proba(X_test)[:, 1]
        m = evaluate.compute_metrics(y_test, proba, threshold=0.5)
        m["model"] = ens_name
        m["cv_roc_auc"] = np.nan
        comparison.append(m)
        print(f"     {ens_name}: test ROC-AUC={m['roc_auc']:.4f} "
              f"PR-AUC={m['pr_auc']:.4f} F1@0.5={m['f1']:.4f}")

    comp_df = pd.DataFrame(comparison).set_index("model").sort_values(
        SELECTION_METRIC, ascending=False
    )
    comp_df.to_csv(config.METRICS_DIR / "model_comparison.csv")
    print("\n[train] leaderboard:\n", comp_df[["roc_auc", "pr_auc", "f1", "cv_roc_auc"]].round(4))

    # ---- champion + threshold tuning (on CV out-of-fold preds) ----------- #
    champ_name = comp_df.index[0]
    champ = tuned[champ_name]
    print(f"\n[train] champion = {champ_name}")

    oof = cross_val_predict(champ, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold = evaluate.find_best_threshold(y_train, oof, objective="f1")
    print(f"[train] tuned decision threshold (F1-max, oof) = {threshold:.3f}")

    # ---- final evaluation at tuned threshold ----------------------------- #
    proba_test = champ.predict_proba(X_test)[:, 1]
    proba_train = champ.predict_proba(X_train)[:, 1]
    test_metrics = evaluate.compute_metrics(y_test, proba_test, threshold)
    train_metrics = evaluate.compute_metrics(y_train, proba_train, threshold)
    test_metrics["overfit_gap_roc_auc"] = evaluate.overfit_gap(train_metrics, test_metrics, "roc_auc")
    print(f"[train] champion test: {json.dumps({k: round(v,4) for k,v in test_metrics.items()})}")

    y_pred_test = (proba_test >= threshold).astype(int)

    # ---- plots ------------------------------------------------------------ #
    top = comp_df.head(6).index.tolist()
    roc_inputs = {n: (y_test, tuned[n].predict_proba(X_test)[:, 1]) for n in top}
    evaluate.plot_roc_curves(roc_inputs)
    evaluate.plot_pr_curves(roc_inputs)
    evaluate.plot_confusion(y_test, y_pred_test, title=f"{champ_name} @ thr={threshold:.2f}")
    evaluate.plot_calibration(y_test, proba_test, name=champ_name)
    evaluate.plot_model_comparison(comp_df.reset_index(), metric="f1")
    evaluate.plot_model_comparison(comp_df.reset_index(), metric="roc_auc")

    # ---- fairness audit --------------------------------------------------- #
    protected = clean_features(X_test)[config.PROTECTED_ATTRIBUTES]
    fair = fairness.audit(y_test, y_pred_test, protected)
    evaluate.save_json(fair, config.REPORTS_DIR / "fairness_audit.json")
    print("[train] fairness gaps:", {a: fair[a]["gaps"] for a in fair})

    # ---- explainability artifacts ---------------------------------------- #
    reference = explain.compute_reference_profile(X_train, y_train)
    try:
        gimp = explain.global_importance(champ, X_test, y_test)
        gimp.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
        print("[train] top features:\n", gimp.head(8).to_string(index=False))
    except Exception as e:  # keep training robust
        print(f"[train] permutation importance skipped: {e}")

    # ---- persist artifact + metadata ------------------------------------- #
    joblib.dump(champ, config.MODEL_ARTIFACT)
    metadata = {
        "model_name": champ_name,
        "selection_metric": SELECTION_METRIC,
        "decision_threshold": threshold,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "quick_mode": quick,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance_train": {"default": float(y_train.mean()),
                                "no_default": float(1 - y_train.mean())},
        "features": {
            "numeric": config.NUMERIC_FEATURES,
            "low_card_categorical": config.LOW_CARD_CATEGORICAL,
            "high_card_categorical": config.HIGH_CARD_CATEGORICAL,
        },
        "protected_attributes": config.PROTECTED_ATTRIBUTES,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "reference_profile": reference,
    }
    with open(config.METADATA_ARTIFACT, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    evaluate.save_json(comp_df.reset_index().to_dict(orient="records"),
                       config.METRICS_DIR / "model_comparison.json")

    print(f"\n[train] saved champion -> {config.MODEL_ARTIFACT}")
    print(f"[train] done in {(dt.datetime.now()-t0).total_seconds():.0f}s")
    return champ, metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run on a subsample")
    args = ap.parse_args()
    run(quick=args.quick)
