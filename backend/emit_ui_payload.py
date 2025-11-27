# backend/emit_ui_payload.py
"""
Builds a JSON payload for the UI given a ticker.

This module can be used in two ways:
1. As a CLI:
   python -m backend.emit_ui_payload --ticker AAPL --start 2000-01-01 --models_dir app/models --model xgb

2. As a library:
   from backend.emit_ui_payload import build_payload
   payload = build_payload("AAPL", "2000-01-01", "app/models", model="xgb")
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import joblib

# -------- Timezone helpers (America/New_York) --------
try:
    from zoneinfo import ZoneInfo

    _NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - Python <3.9 fallback
    _NY_TZ = None


def now_ny() -> datetime:
    """Current time in America/New_York (naive if zoneinfo not available)."""
    if _NY_TZ is not None:
        return datetime.now(_NY_TZ)
    return datetime.now()


# -------- Small helpers --------
def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


# -------- Feature engineering --------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic technical features on a daily OHLCV dataframe.
    df index: DatetimeIndex
    columns: Open, High, Low, Close, Volume
    """
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # 1-day return
    feats["ret1"] = close.pct_change()

    # Moving averages
    feats["ma5"] = close.rolling(window=5, min_periods=5).mean()
    feats["ma10"] = close.rolling(window=10, min_periods=10).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    feats["rsi14"] = 100 - (100 / (1 + rs))

    # Volume z-score over 20 days
    vol_mean20 = vol.rolling(window=20, min_periods=20).mean()
    vol_std20 = vol.rolling(window=20, min_periods=20).std()
    feats["vol_z"] = (vol - vol_mean20) / (vol_std20 + 1e-9)

    return feats


def labels_next_day_up(close: pd.Series) -> pd.Series:
    """
    Label 1 if next day's close is greater than today's close, else 0.
    """
    next_close = close.shift(-1)
    y = (next_close > close).astype(int)
    # last day has no label; drop
    y = y.dropna()
    return y


# -------- Training helpers --------
def train_eval(X: np.ndarray, y: np.ndarray, kind: str) -> Tuple[object, Dict[str, float]]:
    """
    Train either Logistic Regression or XGBoost and return the model + metrics.
    Metrics are evaluated on the last 20% of data as a simple holdout.
    """
    n = len(X)
    if n < 50:
        raise ValueError("Not enough samples to train (%d)" % n)

    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if kind == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif kind == "xgb":
        # quiet xgboost warnings for small samples
        warnings.filterwarnings("ignore", category=UserWarning)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model kind '{kind}'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


# -------- Serialization helpers --------
def to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert OHLCV dataframe to a list of dicts for JSON.
    """
    rows: List[Dict[str, Any]] = []
    for ts, r in df.iterrows():
        rows.append(
            dict(
                date=ts.strftime("%Y-%m-%d"),
                open=_as_float(r["Open"]),
                high=_as_float(r["High"]),
                low=_as_float(r["Low"]),
                close=_as_float(r["Close"]),
                volume=_as_int(r["Volume"]),
            )
        )
    return rows


def next_weekdays(start_dt: pd.Timestamp, n: int = 5) -> List[pd.Timestamp]:
    """
    Generate the next n weekdays (Mon-Fri) after start_dt.
    """
    out: List[pd.Timestamp] = []
    d = start_dt + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# -------- Core payload builder --------
def build_payload(
    ticker: str,
    start: str,
    models_dir: str,
    model: str = "xgb",
) -> Dict[str, Any]:
    """
    Build the full UI payload for a given ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g., "AAPL").
    start : str
        Start date for history (YYYY-MM-DD).
    models_dir : str
        Directory where model files and optional JSON will be stored.
    model : str
        "xgb" or "logreg" – which model to use for prediction/metrics in payload.
    """
    ticker = ticker.upper()
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # ----- Download data up through yesterday (NY time) -----
    yesterday = now_ny().date() - timedelta(days=1)
    yf_end = (yesterday + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    df = yf.download(
        ticker,
        start=start,
        end=yf_end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty or len(df) < 120:
        raise SystemExit(f"Not enough data for {ticker} through {yesterday}")

    # safety: drop anything after yesterday if present
    df = df[df.index.date <= yesterday]

    # ----- Features / labels -----
    feats = make_features(df)
    y = labels_next_day_up(df["Close"])

    # Align indices
    idx = feats.index.intersection(y.index)
    feats = feats.loc[idx]
    y = y.loc[idx]

    # Drop rows with NaNs in features
    feats = feats.dropna()
    y = y.loc[feats.index]

    feature_cols = ["ret1", "ma5", "ma10", "rsi14", "vol_z"]
    X = feats[feature_cols].values
    y_arr = y.values

    if len(X) < 50:
        raise SystemExit(f"Not enough usable samples after feature engineering for {ticker}")

    # ----- Train both models -----
    mdl_lr, met_lr = train_eval(X, y_arr, "logreg")
    mdl_xg, met_xg = train_eval(X, y_arr, "xgb")

    if model == "logreg":
        use_model = mdl_lr
        use_name = "Logistic Regression"
        metrics = met_lr
    else:
        use_model = mdl_xg
        use_name = "XGBoost"
        metrics = met_xg

    # Save models for later reuse (keeps previous behavior)
    joblib.dump(mdl_lr, models_path / f"model_{ticker}_logreg.joblib")
    joblib.dump(mdl_xg, models_path / f"model_{ticker}_xgb.joblib")

    # ----- History rows (full adjusted OHLC) -----
    history_rows = to_rows(df)

    # ===== Next-5-day projections (recursive) =====
    WINDOW_SEED = 40  # enough for MA10/RSI14/vol_z(20)
    if len(df) < WINDOW_SEED:
        seed = df.copy()
    else:
        seed = df.iloc[-WINDOW_SEED:].copy()

    def compute_last_feats(block: pd.DataFrame) -> pd.Series:
        """
        Compute the same features as make_features(), but only for the last row,
        and also compute ATR14 for volatility-scaled steps.
        """
        close = block["Close"].astype(float)
        vol = block["Volume"].astype(float)

        ret1 = close.pct_change().iloc[-1]

        ma5 = close.rolling(5, min_periods=5).mean().iloc[-1]
        ma10 = close.rolling(10, min_periods=10).mean().iloc[-1]

        d = close.diff()
        up = d.clip(lower=0).rolling(14, min_periods=14).mean().iloc[-1]
        dn = (-d.clip(upper=0)).rolling(14, min_periods=14).mean().iloc[-1]
        rsi14 = 100 - (100 / (1 + (_as_float(up) / (_as_float(dn) + 1e-9))))

        vol_mean20 = vol.rolling(20, min_periods=20).mean().iloc[-1]
        vol_std20 = vol.rolling(20, min_periods=20).std().iloc[-1]
        last_vol = vol.iloc[-1]

        if not np.isfinite(vol_std20) or vol_std20 == 0.0:
            vol_z = 0.0
        else:
            vol_z = (last_vol - vol_mean20) / (vol_std20 + 1e-9)

        # ATR14
        tr = pd.concat(
            [
                (block["High"] - block["Low"]),
                (block["High"] - block["Close"].shift()).abs(),
                (block["Low"] - block["Close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=14).mean().iloc[-1]

        return pd.Series(
            {
                "ret1": _as_float(ret1),
                "ma5": _as_float(ma5),
                "ma10": _as_float(ma10),
                "rsi14": _as_float(rsi14),
                "vol_z": _as_float(vol_z),
                "atr14": _as_float(atr14),
            }
        )

    preds: List[Dict[str, Any]] = []
    last_ts = seed.index[-1]
    future_days = next_weekdays(last_ts, 5)

    # start from last close as synthetic open
    o = _as_float(seed["Close"].iloc[-1])
    vol_last = _as_int(seed["Volume"].iloc[-1]) if "Volume" in seed.columns else 0

    for d in future_days:
        f = compute_last_feats(seed).fillna(0.0)
        X_next = f[["ret1", "ma5", "ma10", "rsi14", "vol_z"]].values.reshape(1, -1)

        try:
            p_proba = use_model.predict_proba(X_next)[0]
            p_up = float(p_proba[1])
        except Exception:
            p_up = 0.5

        atr_now = f.get("atr14", 0.0)
        atr_now = _as_float(atr_now)
        if not np.isfinite(atr_now) or atr_now <= 0:
            atr_now = max(1e-3, 0.01 * o)

        # confidence [-1, +1]
        conf = (p_up - 0.5) * 2.0
        step = conf * 0.9 * atr_now

        c = max(0.01, o + step)
        wick_span = 0.4 * atr_now
        h = max(o, c) + 0.5 * wick_span
        l = min(o, c) - 0.5 * wick_span

        preds.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=0,
            )
        )

        # append new synthetic bar and continue recursion
        new_row = pd.DataFrame(
            [[o, h, l, c, vol_last]],
            index=pd.DatetimeIndex([d]),
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        seed = pd.concat([seed, new_row])
        o = c  # next day opens at prior close

    # ----- Build final payload -----
    payload: Dict[str, Any] = dict(
        ticker=ticker,
        model=use_name,
        metrics=metrics,
        history=history_rows,
        predictions_next5=preds,
        backtest={
            "trades": 0,
            "win_rate": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
            "sharpe": 0.0,
        },
        data_until=str(yesterday),
    )

    return payload


# -------- CLI wrapper (keeps old behavior) --------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build UI payload for a given ticker")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        required=False,
        help="End date (ignored; we always use yesterday in NY time)",
    )
    parser.add_argument("--models_dir", required=True, help="Directory for models/payload")
    parser.add_argument(
        "--model",
        choices=["xgb", "logreg"],
        default="xgb",
        help="Which model to use for metrics/predictions",
    )
    args = parser.parse_args()

    payload = build_payload(
        ticker=args.ticker,
        start=args.start,
        models_dir=args.models_dir,
        model=args.model,
    )

    out_path = Path(args.models_dir) / f"ui_payload_{args.ticker.upper()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
