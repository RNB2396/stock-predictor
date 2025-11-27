import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier
import joblib
import warnings

# ============================================================
# Config
# ============================================================

TRAIN_YEARS = 7
WINDOW = 30  # lookback window for sequence features

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


# Simple sector ETF mapping for some common tickers
SECTOR_ETF_MAP: Dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMZN": "XLY",
    "META": "XLC",
    "GOOGL": "XLC",
    "JPM": "XLF",
    "BAC": "XLF",
    "XOM": "XLE",
    "CVX": "XLE",
}

# ============================================================
# Feature engineering
# ============================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def make_features(df: pd.DataFrame, ctx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Technical features for the target ticker + optional market context.
    """
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # Price-based features
    feats["ret1"] = close.pct_change()
    feats["ret5"] = close.pct_change(5)

    ma20 = close.rolling(window=20, min_periods=20).mean()
    ma50 = close.rolling(window=50, min_periods=50).mean()
    feats["ma20"] = ma20
    feats["ma50"] = ma50

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    feats["ema12"] = ema12
    feats["ema26"] = ema26
    feats["macd"] = macd
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd - macd_signal

    std20 = close.rolling(window=20, min_periods=20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    feats["bb_upper"] = bb_upper
    feats["bb_lower"] = bb_lower
    feats["bb_width"] = (bb_upper - bb_lower) / (ma20 + 1e-9)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feats["atr14"] = tr.rolling(window=14, min_periods=14).mean()

    vol_mean20 = vol.rolling(window=20, min_periods=20).mean()
    vol_std20 = vol.rolling(window=20, min_periods=20).std()
    feats["vol_z20"] = (vol - vol_mean20) / (vol_std20 + 1e-9)
    feats["vol_change"] = vol.pct_change()

    feats["open_close_spread"] = (open_ - close) / (close + 1e-9)
    feats["high_close_spread"] = (high - close) / (close + 1e-9)
    feats["low_close_spread"] = (low - close) / (close + 1e-9)

    # Market context features if provided and present
    if ctx is not None and not ctx.empty:
        if "SPY_Close" in ctx.columns:
            spy = ctx["SPY_Close"].astype(float)
            feats["spy_ret1"] = spy.pct_change()
            feats["spy_ret5"] = spy.pct_change(5)

        if "QQQ_Close" in ctx.columns:
            qqq = ctx["QQQ_Close"].astype(float)
            feats["qqq_ret1"] = qqq.pct_change()
            feats["qqq_ret5"] = qqq.pct_change(5)

        if "VIX_Close" in ctx.columns:
            vix = ctx["VIX_Close"].astype(float)
            feats["vix_chg1"] = vix.pct_change()
            feats["vix_chg5"] = vix.pct_change(5)
            feats["vix_level"] = vix

        if "SECTOR_Close" in ctx.columns:
            sec = ctx["SECTOR_Close"].astype(float)
            feats["sector_ret1"] = sec.pct_change()
            feats["sector_ret5"] = sec.pct_change(5)

    feats = feats.ffill().bfill().fillna(0.0)
    return feats


# Base feature list; some may be absent if context could not be loaded for a ticker
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
    "spy_ret1",
    "spy_ret5",
    "qqq_ret1",
    "qqq_ret5",
    "vix_chg1",
    "vix_chg5",
    "vix_level",
    "sector_ret1",
    "sector_ret5",
]


def build_supervised_sequences(
    df: pd.DataFrame,
    ctx: Optional[pd.DataFrame],
    window: int = WINDOW,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Build sequence dataset.

    X_seq: [n_samples, window, n_features]
    y:     [n_samples, 4] = volatility-scaled log-returns for [Open, High, Low, Close].
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

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

    # Log returns
    ret_open = np.log(nxt_open / open_.replace(0, np.nan))
    ret_high = np.log(nxt_high / high.replace(0, np.nan))
    ret_low = np.log(nxt_low / low.replace(0, np.nan))
    ret_close = np.log(nxt_close / close.replace(0, np.nan))

    # Volatility for scaling: std of close pct-change
    close_ret = close.pct_change()
    vol20 = close_ret.rolling(window=20, min_periods=20).std()
    scale = vol20.replace(0, np.nan)

    targets_df = pd.DataFrame(
        {
            "ret_open_scaled": ret_open / scale,
            "ret_high_scaled": ret_high / scale,
            "ret_low_scaled": ret_low / scale,
            "ret_close_scaled": ret_close / scale,
        },
        index=df.index,
    )

    feats = make_features(df, ctx)

    # Keep only feature columns that actually exist
    available_cols = [c for c in feature_cols if c in feats.columns]
    feats = feats[available_cols]

    data = pd.concat([feats, targets_df], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) <= window + 30:
        raise SystemExit(
            f"Not enough usable samples after feature engineering ({len(data)})"
        )

    X_all = data[available_cols].values
    y_all = data[
        ["ret_open_scaled", "ret_high_scaled", "ret_low_scaled", "ret_close_scaled"]
    ].values
    idx_all = data.index.to_list()

    X_seq_list: List[np.ndarray] = []
    y_seq_list: List[np.ndarray] = []
    ts_seq: List[pd.Timestamp] = []

    for i in range(window, len(data)):
        X_seq_list.append(X_all[i - window : i])
        y_seq_list.append(y_all[i])
        ts_seq.append(idx_all[i])

    X_seq = np.asarray(X_seq_list, dtype=np.float32)
    y_seq = np.asarray(y_seq_list, dtype=np.float32)
    return X_seq, y_seq, ts_seq


# ============================================================
# Regression metrics (focus on scaled Close)
# ============================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    close_index: int = 3,
) -> Dict[str, float]:
    """
    Metrics on scaled Close (target index 3).
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
# Regressors (XGB / RF)
# ============================================================

def train_xgb_regressor(
    X_seq: np.ndarray,
    y: np.ndarray,
    split: int,
) -> Tuple[List[XGBRegressor], Dict[str, float]]:
    """
    Train one XGBRegressor per scaled target dimension (4 dims).
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
# Context data (SPY, QQQ, VIX, sector ETF)
# ============================================================

def _build_context(df_index: pd.Index, ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download SPY, QQQ, VIX, and a sector ETF (if mapped) and align to df_index.
    """
    ctx = pd.DataFrame(index=df_index)

    def add_ctx(symbol: str, col_prefix: str, auto_adjust: bool = True):
        try:
            data = yf.download(
                symbol, start=start, end=end, auto_adjust=auto_adjust, progress=False
            )
            if data.empty:
                return
            close = data["Close"].reindex(df_index).ffill()
            ctx[f"{col_prefix}_Close"] = close
        except Exception:
            return

    add_ctx("SPY", "SPY", auto_adjust=True)
    add_ctx("QQQ", "QQQ", auto_adjust=True)
    add_ctx("^VIX", "VIX", auto_adjust=False)

    sector_symbol = SECTOR_ETF_MAP.get(ticker.upper())
    if sector_symbol:
        add_ctx(sector_symbol, "SECTOR", auto_adjust=True)

    return ctx


# ============================================================
# Model paths (regressors) – versioned to avoid old-cache shape mismatch
# ============================================================

def _model_paths_reg(models_dir: Path, ticker: str, model_kind: str) -> Tuple[Path, Path]:
    """
    Versioned model filenames so that changes in feature set don't re-use old models.
    """
    version = "v2"  # bump this whenever feature layout changes
    if model_kind == "xgb":
        model_path = models_dir / f"{ticker}_{model_kind}_{version}_models.joblib"
    elif model_kind == "rf":
        model_path = models_dir / f"{ticker}_{model_kind}_{version}.joblib"
    else:
        raise SystemExit(f"Unsupported model kind for paths: {model_kind}")
    metrics_path = models_dir / f"{ticker}_{model_kind}_{version}_metrics.json"
    return model_path, metrics_path


def get_or_train_regressor(
    ticker: str,
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    models_dir: Path,
    model_kind: str,
    feature_cols: List[str],
    window: int,
) -> Tuple[Any, Dict[str, float]]:
    """
    Load or train the regression model (XGB/RF) and metrics (single 80/20 split).
    """
    model_kind = model_kind.lower()
    model_path, metrics_path = _model_paths_reg(models_dir, ticker, model_kind)

    # Load cached
    if model_path.exists() and metrics_path.exists():
        try:
            model = joblib.load(model_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return model, metrics
        except Exception:
            pass

    # Train from scratch
    X_seq, y_seq, _ = build_supervised_sequences(
        df, ctx, window=window, feature_cols=feature_cols
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
# Decode scaled returns to OHLC
# ============================================================

def _decode_scaled_returns_to_ohlc(
    last_row: pd.Series,
    y_hat_scaled: np.ndarray,
    seed: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Convert scaled returns back to OHLC prices using current volatility.
    """
    last_open = float(last_row["Open"])
    last_high = float(last_row["High"])
    last_low = float(last_row["Low"])
    last_close = float(last_row["Close"])

    close_series = seed["Close"].astype(float)
    close_ret = close_series.pct_change()
    vol20 = close_ret.rolling(window=20, min_periods=5).std()

    # Robust, explicit handling of volatility:
    vol_valid = vol20.dropna()
    if len(vol_valid) > 0:
        vol_last = float(vol_valid.iloc[-1])
        if not np.isfinite(vol_last) or vol_last <= 1e-6:
            vol_last = float(vol_valid.tail(60).mean())
    else:
        vol_last = 0.02

    if not np.isfinite(vol_last) or vol_last <= 1e-6:
        vol_last = 0.02

    scaled = np.asarray(y_hat_scaled, dtype=float)
    while scaled.size < 4:
        scaled = np.append(scaled, 0.0)

    ret_open = scaled[0] * vol_last
    ret_high = scaled[1] * vol_last
    ret_low = scaled[2] * vol_last
    ret_close = scaled[3] * vol_last

    o_raw = last_open * float(np.exp(ret_open))
    h_raw = last_high * float(np.exp(ret_high))
    l_raw = last_low * float(np.exp(ret_low))
    c_raw = last_close * float(np.exp(ret_close))

    o = max(o_raw, 0.01)
    c = max(c_raw, 0.01)
    h = max(h_raw, o, c)
    l = max(min(l_raw, o, c), 0.01)

    return float(o), float(h), float(l), float(c)


# ============================================================
# Next-5-day prediction using regression model
# ============================================================

def predict_next5_ohlc(
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    model_kind: str,
    model_obj: Any,
) -> List[Dict[str, Any]]:
    """
    Predict next 5 trading days using regression model on scaled returns.
    """
    seed = df.copy()
    ctx_seed = ctx.copy()
    last_ts = seed.index[-1]
    future_days = next_weekdays(last_ts, 5)

    preds: List[Dict[str, Any]] = []

    for d in future_days:
        feats = make_features(seed, ctx_seed)
        available_cols = [c for c in feature_cols if c in feats.columns]
        feats = feats[available_cols]

        if len(feats) < window:
            needed = window - len(feats)
            last_feats = feats.iloc[[-1]]
            pad_df = pd.concat([last_feats] * needed, axis=0)
            feats = pd.concat([pad_df, feats], axis=0)

        window_slice = feats.values[-window:]

        if model_kind == "xgb":
            X_flat = window_slice.reshape(1, -1)
            models: List[XGBRegressor] = model_obj  # type: ignore
            y_hat_cols = [m.predict(X_flat)[0] for m in models]
            y_hat_scaled = np.array(y_hat_cols, dtype=float)
        elif model_kind == "rf":
            X_flat = window_slice.reshape(1, -1)
            y_hat_scaled = np.asarray(
                model_obj.predict(X_flat)[0], dtype=float
            )  # type: ignore
        else:
            raise SystemExit(f"Unsupported model kind for prediction: {model_kind}")

        last_row = seed.iloc[-1]
        o, h, l, c = _decode_scaled_returns_to_ohlc(last_row, y_hat_scaled, seed)
        v = float(last_row["Volume"])

        preds.append(
            dict(
                date=d.strftime("%Y-%m-%d"),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
        )

        new_row = pd.DataFrame(
            [[o, h, l, c, v]],
            index=pd.DatetimeIndex([d]),
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
        seed = pd.concat([seed, new_row])

        if not ctx_seed.empty:
            last_ctx = ctx_seed.iloc[[-1]].copy()
            last_ctx.index = new_row.index
            ctx_seed = pd.concat([ctx_seed, last_ctx])

    return preds


# ============================================================
# Rolling walk-forward classifier + signals
# ============================================================

def _fit_xgb_classifier(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)
    return clf


def rolling_walkforward_classifier_signals(
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    feature_cols: List[str],
    window: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Rolling walk-forward evaluation for next-day up/down classifier.
    """
    X_seq, y_seq, ts_seq = build_supervised_sequences(
        df, ctx, window=window, feature_cols=feature_cols
    )
    n_samples, win, n_feats = X_seq.shape
    if n_samples < 200:
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": 0,
            "cls_n_test": 0,
        }, []

    X_flat = X_seq.reshape(n_samples, win * n_feats)
    labels = (y_seq[:, 3] > 0).astype(int)  # up/down based on scaled close

    # Rolling setup: first 50% as initial train, remaining as test blocks
    initial_train = max(int(n_samples * 0.5), window * 2)
    if initial_train >= n_samples - 5:
        initial_train = n_samples // 2

    remaining = n_samples - initial_train
    test_block = max(20, remaining // 4)  # ~4 blocks over remaining

    all_true: List[int] = []
    all_pred: List[int] = []
    signals: List[Dict[str, Any]] = []
    total_train = 0
    total_test = 0

    idx = initial_train
    while idx < n_samples:
        train_end = idx
        test_start = idx
        test_end = min(idx + test_block, n_samples)

        if train_end <= window:
            break

        X_train = X_flat[:train_end]
        y_train = labels[:train_end]
        X_test = X_flat[test_start:test_end]
        y_test = labels[test_start:test_end]

        clf = _fit_xgb_classifier(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

        total_train += len(X_train)
        total_test += len(X_test)

        # signals for these test days
        for i_local, i_global in enumerate(range(test_start, test_end)):
            ts = ts_seq[i_global]
            if ts not in df.index:
                continue
            price = float(df.loc[ts, "Close"])
            side = "buy" if pred[i_local] == 1 else "sell"
            signals.append(
                dict(
                    date=ts.strftime("%Y-%m-%d"),
                    side=side,
                    price=price,
                )
            )

        idx = test_end

    if total_test == 0 or not all_true:
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": total_train,
            "cls_n_test": total_test,
        }, signals

    y_true_arr = np.asarray(all_true)
    y_pred_arr = np.asarray(all_pred)

    cls_metrics = {
        "cls_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "cls_precision": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "cls_recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "cls_n_train": int(total_train),
        "cls_n_test": int(total_test),
    }
    return cls_metrics, signals


# ============================================================
# Core payload builder
# ============================================================

def build_payload(
    ticker: str,
    start: str,
    models_dir: str,
    model: str = "xgb",
) -> Dict[str, Any]:
    """
    Build UI payload for a given ticker and chosen model.
    """
    ticker = ticker.upper()
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_lower = model.lower()
    if model_lower not in {"xgb", "rf"}:
        model_lower = "xgb"

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

    ctx = _build_context(df.index, ticker, train_start_date, yf_end)

    feature_cols = FEATURE_COLS
    window = WINDOW

    # Regressor (cached)
    reg_model, reg_metrics = get_or_train_regressor(
        ticker=ticker,
        df=df,
        ctx=ctx,
        models_dir=models_path,
        model_kind=model_lower,
        feature_cols=feature_cols,
        window=window,
    )

    # Rolling walk-forward classifier + signals
    cls_metrics, signals = rolling_walkforward_classifier_signals(
        df=df,
        ctx=ctx,
        feature_cols=feature_cols,
        window=window,
    )

    metrics: Dict[str, Any] = {}
    metrics.update(reg_metrics)
    metrics.update(cls_metrics)

    history_rows = to_rows(df)

    preds_next5 = predict_next5_ohlc(
        df=df,
        ctx=ctx,
        feature_cols=feature_cols,
        window=window,
        model_kind=model_lower,
        model_obj=reg_model,
    )

    model_name_map = {
        "xgb": "XGBoost Regressor",
        "rf": "Random Forest Regressor",
    }
    use_name = model_name_map[model_lower]

    payload: Dict[str, Any] = dict(
        ticker=ticker,
        model=use_name,
        metrics=metrics,
        history=history_rows,
        predictions_next5=preds_next5,
        signals=signals,
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
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=False)
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
