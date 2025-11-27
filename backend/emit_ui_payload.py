import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import warnings

# ============================================================
# Config
# ============================================================

# how many years of history to use for training
TRAIN_YEARS = 7

# ============================================================
# Timezone helpers (America/New_York)
# ============================================================
try:
    from zoneinfo import ZoneInfo

    TZ_NY = ZoneInfo("America/New_York")
except Exception:
    TZ_NY = None


def now_ny() -> datetime:
    return datetime.now(TZ_NY) if TZ_NY else datetime.now()


# ============================================================
# Small helpers
# ============================================================
def _as_float(x) -> float:
    """Coerce scalar/array/Series to float (last element), NaN on failure."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")
        return float(arr.ravel()[-1])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


def _as_int(x) -> int:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return 0
        return int(arr.ravel()[-1])
    except Exception:
        try:
            return int(x)
        except Exception:
            return 0


# ============================================================
# Feature engineering
# ============================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rich technical features for each bar.
    """
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # Simple returns on close
    feats["ret1"] = close.pct_change()
    feats["ret5"] = close.pct_change(5)

    # Moving averages
    ma20 = close.rolling(window=20, min_periods=20).mean()
    ma50 = close.rolling(window=50, min_periods=50).mean()
    feats["ma20"] = ma20
    feats["ma50"] = ma50

    # EMAs + MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    feats["ema12"] = ema12
    feats["ema26"] = ema26
    feats["macd"] = macd
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd - macd_signal

    # Bollinger Bands (20, 2)
    std20 = close.rolling(window=20, min_periods=20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    feats["bb_upper"] = bb_upper
    feats["bb_lower"] = bb_lower
    feats["bb_width"] = (bb_upper - bb_lower) / (ma20 + 1e-9)

    # ATR(14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feats["atr14"] = tr.rolling(window=14, min_periods=14).mean()

    # Volume features
    vol_mean20 = vol.rolling(window=20, min_periods=20).mean()
    vol_std20 = vol.rolling(window=20, min_periods=20).std()
    feats["vol_z20"] = (vol - vol_mean20) / (vol_std20 + 1e-9)
    feats["vol_change"] = vol.pct_change()

    # Open/High/Low vs Close spreads
    feats["open_close_spread"] = (open_ - close) / (close + 1e-9)
    feats["high_close_spread"] = (high - close) / (close + 1e-9)
    feats["low_close_spread"] = (low - close) / (close + 1e-9)

    feats = feats.ffill().bfill().fillna(0.0)
    return feats


FEATURE_COLS: List[str] = [
    "ret1",
    "ret5",
    "ma20",
    "ma50",
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr14",
    "vol_z20",
    "vol_change",
    "open_close_spread",
    "high_close_spread",
    "low_close_spread",
]


def build_supervised_sequences(
    df: pd.DataFrame,
    window: int = 30,
    feature_cols: List[str] = None,
    target_cols: List[str] = None,  # unused, kept for compat
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Sliding-window dataset for sequence models.

    X_seq: [n_samples, window, n_features]
    y:     [n_samples, 4] for next-day log-returns of [Open, High, Low, Close].
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    # Force columns to 1D Series
    open_vals = np.asarray(df["Open"].astype(float)).ravel()
    high_vals = np.asarray(df["High"].astype(float)).ravel()
    low_vals = np.asarray(df["Low"].astype(float)).ravel()
    close_vals = np.asarray(df["Close"].astype(float)).ravel()
    vol_vals = np.asarray(df["Volume"].astype(float)).ravel()

    open_ = pd.Series(open_vals, index=df.index)
    high = pd.Series(high_vals, index=df.index)
    low = pd.Series(low_vals, index=df.index)
    close = pd.Series(close_vals, index=df.index)
    vol = pd.Series(vol_vals, index=df.index)

    # Next-day values
    nxt_open = open_.shift(-1)
    nxt_high = high.shift(-1)
    nxt_low = low.shift(-1)
    nxt_close = close.shift(-1)

    # Log-returns for OHLC
    ret_open_arr = np.log(nxt_open / open_.replace(0, np.nan))
    ret_high_arr = np.log(nxt_high / high.replace(0, np.nan))
    ret_low_arr = np.log(nxt_low / low.replace(0, np.nan))
    ret_close_arr = np.log(nxt_close / close.replace(0, np.nan))

    ret_open = pd.Series(np.asarray(ret_open_arr).ravel(), index=df.index)
    ret_high = pd.Series(np.asarray(ret_high_arr).ravel(), index=df.index)
    ret_low = pd.Series(np.asarray(ret_low_arr).ravel(), index=df.index)
    ret_close = pd.Series(np.asarray(ret_close_arr).ravel(), index=df.index)

    targets_df = pd.DataFrame(
        {
            "next_ret_open": ret_open,
            "next_ret_high": ret_high,
            "next_ret_low": ret_low,
            "next_ret_close": ret_close,
        },
        index=df.index,
    )

    feats = make_features(df)
    data = pd.concat([feats, targets_df], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) <= window + 30:
        raise SystemExit(
            f"Not enough usable samples after feature engineering ({len(data)})"
        )

    X_all = data[feature_cols].values
    y_all = data[
        ["next_ret_open", "next_ret_high", "next_ret_low", "next_ret_close"]
    ].values
    idx_all = data.index.to_list()

    X_seq: List[np.ndarray] = []
    y_seq: List[np.ndarray] = []
    ts_seq: List[pd.Timestamp] = []

    for i in range(window, len(data)):
        X_seq.append(X_all[i - window : i])
        y_seq.append(y_all[i])
        ts_seq.append(idx_all[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32), ts_seq


# ============================================================
# Regression metrics (focus on Close)
# ============================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    close_index: int = 3,
) -> Dict[str, float]:
    """
    Metrics on Close (target index 3).
    """
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "rmse_close": float("nan"),
            "mae_close": float("nan"),
            "r2_close": float("nan"),
        }

    if y_true.ndim == 1:
        y_true_main = y_true
        y_pred_main = y_pred
    else:
        if close_index >= y_true.shape[1]:
            close_index = 0
        y_true_main = y_true[:, close_index]
        y_pred_main = y_pred[:, close_index]

    mse = mean_squared_error(y_true_main, y_pred_main)
    mae = mean_absolute_error(y_true_main, y_pred_main)
    r2 = r2_score(y_true_main, y_pred_main)

    return {
        "rmse_close": float(np.sqrt(mse)),
        "mae_close": float(mae),
        "r2_close": float(r2),
    }


# ============================================================
# Tabular regressors (XGB / RF)
# ============================================================

def train_xgb_regressor(
    X_seq: np.ndarray,
    y: np.ndarray,
    split: int,
) -> Tuple[List[XGBRegressor], Dict[str, float]]:
    """
    Train one XGBRegressor per target dimension (4 dims: O,H,L,C).
    """
    n_samples, window, n_feats = X_seq.shape
    X_flat = X_seq.reshape(n_samples, window * n_feats)

    X_train, X_test = X_flat[:split], X_flat[split:]
    y_train, y_test = y[:split], y[split:]

    models: List[XGBRegressor] = []
    preds_test = []

    warnings.filterwarnings("ignore", category=UserWarning)

    n_targets = y.shape[1]
    for j in range(n_targets):
        mdl = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=1.0,
            objective="reg:squarederror",
            n_jobs=4,
        )
        mdl.fit(X_train, y_train[:, j])
        models.append(mdl)
        preds_test.append(mdl.predict(X_test))

    y_pred = np.column_stack(preds_test)
    metrics = regression_metrics(y_test, y_pred, close_index=3)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    return models, metrics


def train_rf_regressor(
    X_seq: np.ndarray,
    y: np.ndarray,
    split: int,
) -> Tuple[MultiOutputRegressor, Dict[str, float]]:
    """
    RandomForest multi-output regressor over flattened windows.
    """
    n_samples, window, n_feats = X_seq.shape
    X_flat = X_seq.reshape(n_samples, window * n_feats)

    X_train, X_test = X_flat[:split], X_flat[split:]
    y_train, y_test = y[:split], y[split:]

    base = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        n_jobs=4,
        random_state=42,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, close_index=3)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    return model, metrics


# ============================================================
# Serialization helpers
# ============================================================

def to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
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
    out: List[pd.Timestamp] = []
    d = start_dt + timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ============================================================
# Next-5-day prediction using trained model (direct OHLC)
# ============================================================

def predict_next5_ohlc(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    model_kind: str,
    model_obj,
) -> List[Dict[str, Any]]:
    """
    Predict next 5 trading days.

    Models predict next-day log returns for [Open, High, Low, Close].
    We decode directly to prices. Volume is held at last value.
    """
    seed = df.copy()
    last_ts = seed.index[-1]
    future_days = next_weekdays(last_ts, 5)

    preds: List[Dict[str, Any]] = []

    for d in future_days:
        feats = make_features(seed)[feature_cols]
        if len(feats) < window:
            needed = window - len(feats)
            last_row = feats.iloc[[-1]]
            pad_df = pd.concat([last_row] * needed, axis=0)
            feats = pd.concat([pad_df, feats], axis=0)

        window_slice = feats.values[-window:]

        # model inference: returns [ret_open, ret_high, ret_low, ret_close]
        if model_kind == "xgb":
            X_flat = window_slice.reshape(1, -1)
            models: List[XGBRegressor] = model_obj  # type: ignore
            y_hat_cols = [m.predict(X_flat)[0] for m in models]
            y_hat = np.array(y_hat_cols, dtype=float)
        elif model_kind == "rf":
            X_flat = window_slice.reshape(1, -1)
            y_hat = np.asarray(model_obj.predict(X_flat)[0], dtype=float)  # type: ignore
        else:
            raise SystemExit(f"Unsupported model kind for prediction: {model_kind}")

        y_hat = np.asarray(y_hat, dtype=float)

        last = seed.iloc[-1]
        last_open = float(last["Open"])
        last_high = float(last["High"])
        last_low = float(last["Low"])
        last_close = float(last["Close"])
        last_vol = float(last["Volume"])

        # decode log-returns -> prices, with basic sanity checks
        def safe_price(prev: float, ret_idx: int) -> float:
            if ret_idx >= y_hat.size or not np.isfinite(y_hat[ret_idx]):
                return prev
            val = prev * float(np.exp(float(y_hat[ret_idx])))
            return float(max(val, 0.01))

        o_raw = safe_price(last_open, 0)
        h_raw = safe_price(last_high, 1)
        l_raw = safe_price(last_low, 2)
        c_raw = safe_price(last_close, 3)

        # enforce OHLC consistency: high >= max(o,c), low <= min(o,c)
        o = o_raw
        c = c_raw
        h = max(h_raw, o, c)
        l = min(l_raw, o, c)
        l = max(l, 0.01)

        v = last_vol  # for now, hold volume constant for accuracy on price

        preds.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )
        )

        new_row = pd.DataFrame(
            [[o, h, l, c, v]],
            index=pd.DatetimeIndex([d]),
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        seed = pd.concat([seed, new_row])

    return preds


# ============================================================
# Model caching: load or train per ticker/model
# ============================================================

def _model_paths(models_dir: Path, ticker: str, model_kind: str) -> Tuple[Path, Path]:
    if model_kind == "xgb":
        model_path = models_dir / f"{ticker}_{model_kind}_models.joblib"
    elif model_kind == "rf":
        model_path = models_dir / f"{ticker}_{model_kind}.joblib"
    else:
        raise SystemExit(f"Unsupported model kind for paths: {model_kind}")
    metrics_path = models_dir / f"{ticker}_{model_kind}_metrics.json"
    return model_path, metrics_path


def get_or_train_model(
    ticker: str,
    df: pd.DataFrame,
    models_dir: Path,
    model_kind: str,
    feature_cols: List[str],
    window: int,
) -> Tuple[Any, Dict[str, float]]:
    """
    Load model+metrics for (ticker, model_kind) if available, otherwise train and save.
    Supported model_kind: 'xgb', 'rf'.
    """
    model_kind = model_kind.lower()
    model_path, metrics_path = _model_paths(models_dir, ticker, model_kind)

    # Try loading cached model
    if model_path.exists() and metrics_path.exists():
        try:
            model = joblib.load(model_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return model, metrics
        except Exception:
            pass

    # Train from scratch
    X_seq, y_seq, _ = build_supervised_sequences(
        df,
        window=window,
        feature_cols=feature_cols,
        target_cols=["Open", "High", "Low", "Close"],
    )
    n_samples = len(X_seq)
    split = int(n_samples * 0.8)
    if split <= 0 or split >= n_samples:
        raise SystemExit("Not enough rows to split train/test in sequence dataset.")

    if model_kind == "xgb":
        model, metrics = train_xgb_regressor(X_seq, y_seq, split)
    elif model_kind == "rf":
        model, metrics = train_rf_regressor(X_seq, y_seq, split)
    else:
        raise SystemExit(f"Unknown model kind: {model_kind}")

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception:
        pass

    return model, metrics


# ============================================================
# Core payload builder
# ============================================================

def build_payload(
    ticker: str,
    start: str,          # kept for API signature; training uses TRAIN_YEARS
    models_dir: str,
    model: str = "xgb",
) -> Dict[str, Any]:
    """
    Build UI payload for a given ticker and chosen model.
    Models:
      - xgb   : XGBoost Regressor
      - rf    : Random Forest Regressor
    """
    ticker = ticker.upper()
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_lower = model.lower()
    if model_lower not in {"xgb", "rf"}:
        model_lower = "xgb"

    # --- training data: last TRAIN_YEARS years up to yesterday ---
    yesterday = now_ny().date() - timedelta(days=1)
    yf_end = (yesterday + timedelta(days=1)).strftime("%Y-%m-%d")

    train_start_date = (yesterday - timedelta(days=365 * TRAIN_YEARS)).strftime(
        "%Y-%m-%d"
    )

    df = yf.download(
        ticker,
        start=train_start_date,
        end=yf_end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty or len(df) < 260:
        raise SystemExit(f"Not enough data for {ticker} through {yesterday}")

    df = df[df.index.date <= yesterday]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    feature_cols = FEATURE_COLS
    window = 30

    _ = make_features(df)

    use_model, metrics = get_or_train_model(
        ticker=ticker,
        df=df,
        models_dir=models_path,
        model_kind=model_lower,
        feature_cols=feature_cols,
        window=window,
    )

    model_name_map = {
        "xgb": "XGBoost Regressor",
        "rf": "Random Forest Regressor",
    }
    use_name = model_name_map[model_lower]

    history_rows = to_rows(df)

    preds = predict_next5_ohlc(
        df=df,
        feature_cols=feature_cols,
        window=window,
        model_kind=model_lower,
        model_obj=use_model,
    )

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


# ============================================================
# CLI wrapper
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build UI payload for a given ticker")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)  # unused for training horizon
    parser.add_argument("--end", required=False)   # ignored; we always use yesterday
    parser.add_argument("--models_dir", required=True)
    parser.add_argument(
        "--model",
        choices=["xgb", "rf"],
        default="xgb",
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
