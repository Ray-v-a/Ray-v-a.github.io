#!/usr/bin/env python3

import argparse
import json
import joblib
import sys
from importlib import resources
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any


BASE_PACKAGE = "outbreak_probabilities.machine_learning"

def pkg_root():

    try:
        return resources.files(BASE_PACKAGE)
    except Exception as e:
        raise RuntimeError(f"Cannot access package {BASE_PACKAGE}: {e}")

def model_folders() -> List:
    """Return list of model folders in package."""
    root = pkg_root()
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("models_")]

def discover() -> List[Dict[str, Any]]:
    """Discover packaged models and their metadata."""
    out = []
    for folder in model_folders():
        for f in folder.iterdir():
            if f.is_file() and f.name.endswith(".pkl") and not f.name.endswith("_scaler.pkl") and not f.name.endswith("_meta.pkl"):
                stem = f.stem
                meta = None
                j = folder.joinpath(f"{stem}.json")
                if j.exists():
                    try:
                        with j.open("r") as fh:
                            meta = json.load(fh)
                    except Exception:
                        meta = None
                if meta is None:  # fallback inference
                    parts = stem.split("_")
                    model_type = parts[-1] if len(parts) >= 3 else stem
                    n_weeks = None
                    feat = []
                    if len(parts) >= 3:
                        tok = parts[1]
                        digits = "".join([c for c in tok if c.isdigit()])
                        if digits:
                            n_weeks = int(digits)
                            feat = [f"week_{i+1}" for i in range(n_weeks)]
                    meta = {"model_name": model_type, "n_weeks": n_weeks, "feature_names": feat}
                out.append({"stem": stem, "folder": folder, "meta": meta})
    return sorted(out, key=lambda x: (x["folder"].name, x["stem"]))

def load_from_pkg(folder, stem):
    """Load model and scaler from package folder."""
    model_res = folder.joinpath(f"{stem}.pkl")
    scaler_res = folder.joinpath(f"{stem}_scaler.pkl")
    if not model_res.exists() or not scaler_res.exists():
        raise FileNotFoundError("Missing model or scaler in package folder.")
    with resources.as_file(model_res) as mp, resources.as_file(scaler_res) as sp:
        model = joblib.load(mp)
        scaler = joblib.load(sp)
    return model, scaler

def predict_proba(model, scaler, X: np.ndarray) -> np.ndarray:
    """Predict PMO probabilities using model and scaler."""
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xs)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xs)
        if s.ndim > 1:
            s = s[:, 0]
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Model lacks predict_proba/decision_function")

def main(argv=None):
    """CLI entry point for PMO prediction."""
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--weeks", type=int)
    p.add_argument("--model", type=str, default="RF")
    p.add_argument("--week", action="append", type=float, help="repeat for each week (ordered)")
    p.add_argument("--batch", type=str, help="CSV path")
    p.add_argument("--out", type=str, help="CSV out path")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    models = discover()
    if args.list:
        if not models:
            print("No packaged models found.")
            return
        for m in models:
            meta = m["meta"]
            print(f"{meta.get('model_name')} | {meta.get('n_weeks')} weeks | {meta.get('feature_names')} | {m['folder'].name} | {m['stem']}")
        return

    if not models:
        print("No models available; ensure model folders are packaged.", file=sys.stderr); sys.exit(2)

    cand = [m for m in models if (m["meta"].get("model_name","").lower() == args.model.lower()) and (m["meta"].get("n_weeks") == args.weeks)]
    if not cand:
        print("No matching model for --model and --weeks. Use --list.", file=sys.stderr); sys.exit(2)
    chosen = sorted(cand, key=lambda c: c["folder"].joinpath(f"{c['stem']}.pkl").stat().st_mtime, reverse=True)[0]

    feature_names = chosen["meta"].get("feature_names") or []
    model, scaler = load_from_pkg(chosen["folder"], chosen["stem"])

    if args.week and not args.batch:
        if len(args.week) != len(feature_names):
            print(f"Expected {len(feature_names)} values: {feature_names}", file=sys.stderr); sys.exit(2)
        X = np.array([args.week], dtype=float)
        proba = float(predict_proba(model, scaler, X)[0])
        pred = int(proba >= args.threshold)
        print({"model": chosen["stem"], "features": dict(zip(feature_names, args.week)), "probability": proba, "PMO": pred, "label": "major" if pred else "minor"})
        return

    if args.batch:
        df = pd.read_csv(args.batch)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            print(f"Batch CSV missing columns: {missing}", file=sys.stderr); sys.exit(2)
        X = df[feature_names].astype(float).to_numpy()
        proba = predict_proba(model, scaler, X)
        df_out = df.copy()
        df_out["pmo_probability"] = proba
        df_out["pmo_predicted_class"] = (proba >= args.threshold).astype(int)
        df_out["pmo_predicted_label"] = df_out["pmo_predicted_class"].map({1: "major", 0: "minor"})
        if args.out:
            df_out.to_csv(args.out, index=False); print("Saved to", args.out)
        else:
            print(df_out.head().to_string(index=False))
        return

    print("No action requested. Use --list, or provide --weeks and --week(s) or --batch.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
    