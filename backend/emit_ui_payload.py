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
WINDOW = 30  #lookback window

#Move Filtering
CLS_RET_THRESHOLD = 0.30

# Confidence threshold
CLS_CONFIDENCE_BAND = 0.05

# Base training
BASE_UNIVERSE_TICKERS: List[str] = [
    # Large-cap tech / growth
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AMD", "AVGO",
    "ADBE", "CRM", "INTC", "CSCO", "ORCL", "IBM", "NFLX", "SHOP", "PYPL",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
    # Energy
    "XOM", "CVX", "SLB", "COP", "EOG", "PSX",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY",
    # Consumer
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "HD", "LOW", "NKE", "SBUX",
    # Industrials
    "BA", "CAT", "GE", "HON", "LMT", "DE",
    # Communications / media
    "T", "VZ", "DIS", "CMCSA", "TMUS",
    # Materials / utilities / REITs
    "LIN", "NEM", "FCX", "NEE", "DUK", "SO", "PLD", "O",
    # Broad ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "XLB", "XLC", "XLRE",
]

GLOBAL_CLS_VERSION = "v2"

# Timezone helpers (America/New_York)

try:
    from zoneinfo import ZoneInfo

    TZ_NY = ZoneInfo("America/New_York")
except Exception:
    TZ_NY = None


def now_ny() -> datetime:
    return datetime.now(TZ_NY) if TZ_NY else datetime.now()

# Small helpers

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

# Sector ETF mapping

SECTOR_ETF_MAP: Dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMD": "XLK",
    "AVGO": "XLK",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "META": "XLC",
    "GOOGL": "XLC",
    "GOOG": "XLC",
    "JPM": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "XOM": "XLE",
    "CVX": "XLE",
    "UNH": "XLV",
    "JNJ": "XLV",
    "WMT": "XLP",
    "COST": "XLP",
}

# Feature engineering

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def make_features(df: pd.DataFrame, ctx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Rich technical + context features. ctx columns are optional.
    """
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # Price and returns
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

    # Context, when available
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

        if "UUP_Close" in ctx.columns:
            uup = ctx["UUP_Close"].astype(float)
            feats["uup_ret1"] = uup.pct_change()
            feats["uup_ret5"] = uup.pct_change(5)

        if "GLD_Close" in ctx.columns:
            gld = ctx["GLD_Close"].astype(float)
            feats["gld_ret1"] = gld.pct_change()
            feats["gld_ret5"] = gld.pct_change(5)

        if "TLT_Close" in ctx.columns:
            tlt = ctx["TLT_Close"].astype(float)
            feats["tlt_ret1"] = tlt.pct_change()
            feats["tlt_ret5"] = tlt.pct_change(5)

        if "HYG_Close" in ctx.columns:
            hyg = ctx["HYG_Close"].astype(float)
            feats["hyg_ret1"] = hyg.pct_change()
            feats["hyg_ret5"] = hyg.pct_change(5)

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
    "spy_ret1",
    "spy_ret5",
    "qqq_ret1",
    "qqq_ret5",
    "vix_chg1",
    "vix_chg5",
    "vix_level",
    "sector_ret1",
    "sector_ret5",
    "uup_ret1",
    "uup_ret5",
    "gld_ret1",
    "gld_ret5",
    "tlt_ret1",
    "tlt_ret5",
    "hyg_ret1",
    "hyg_ret5",
]


def build_supervised_sequences(
    df: pd.DataFrame,
    ctx: Optional[pd.DataFrame],
    window: int = WINDOW,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    X_seq: [n_samples, window, n_features]
    y:     [n_samples, 4] = scaled log-returns for [Open, High, Low, Close].
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

    #Log returns
    ret_open = np.log(nxt_open / open_.replace(0, np.nan))
    ret_high = np.log(nxt_high / high.replace(0, np.nan))
    ret_low = np.log(nxt_low / low.replace(0, np.nan))
    ret_close = np.log(nxt_close / close.replace(0, np.nan))

    ret_open = pd.Series(np.asarray(ret_open).ravel(), index=df.index)
    ret_high = pd.Series(np.asarray(ret_high).ravel(), index=df.index)
    ret_low = pd.Series(np.asarray(ret_low).ravel(), index=df.index)
    ret_close = pd.Series(np.asarray(ret_close).ravel(), index=df.index)

    # Volatility scale
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

    # Ensure all feature_cols exist 
    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0.0
    feats = feats[feature_cols]

    data = pd.concat([feats, targets_df], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) <= window + 30:
        raise SystemExit(
            f"Not enough usable samples after feature engineering ({len(data)})"
        )

    X_all = data[feature_cols].values
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

# Regression metrics

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    close_index: int = 3,
) -> Dict[str, float]:
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

# Regressors (per ticker)

def train_xgb_regressor(
    X_seq: np.ndarray,
    y: np.ndarray,
    split: int,
) -> Tuple[List[XGBRegressor], Dict[str, float]]:
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
            subsample=0.9,
            colsample_bytree=0.9,
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

# Context data

def _build_context(df_index: pd.Index, ticker: str, start: str, end: str) -> pd.DataFrame:
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

    add_ctx("UUP", "UUP", auto_adjust=True)
    add_ctx("GLD", "GLD", auto_adjust=True)
    add_ctx("TLT", "TLT", auto_adjust=True)
    add_ctx("HYG", "HYG", auto_adjust=True)

    return ctx

# Regressor model paths

def _model_paths_reg(models_dir: Path, ticker: str, model_kind: str) -> Tuple[Path, Path]:
    version = "v3"
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
    model_kind = model_kind.lower()
    model_path, metrics_path = _model_paths_reg(models_dir, ticker, model_kind)

    if model_path.exists() and metrics_path.exists():
        try:
            model = joblib.load(model_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return model, metrics
        except Exception:
            pass

    X_seq, y_seq, _ = build_supervised_sequences(
        df, ctx, window=window, feature_cols=feature_cols
    )
    n_samples = len(X_seq)
    split = int(n_samples * 0.8)
    if split <= 0 or split >= n_samples:
        raise SystemExit("Not enough rows to split train/test for regressor.")

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

# Decode scaled returns to OHLC

def _decode_scaled_returns_to_ohlc(
    last_row: pd.Series,
    y_hat_scaled: np.ndarray,
    seed: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    last_open = _as_float(last_row["Open"])
    last_high = _as_float(last_row["High"])
    last_low = _as_float(last_row["Low"])
    last_close = _as_float(last_row["Close"])

    close_series = seed["Close"].astype(float)
    close_ret = close_series.pct_change()
    vol20 = close_ret.rolling(window=20, min_periods=5).std()

    vol_valid = vol20.dropna()
    if len(vol_valid) > 0:
        vol_last = _as_float(vol_valid.iloc[-1])
        if not np.isfinite(vol_last) or vol_last <= 1e-6:
            vol_last = _as_float(vol_valid.tail(60).mean())
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


def predict_next5_ohlc(
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    model_kind: str,
    model_obj: Any,
) -> List[Dict[str, Any]]:
    seed = df.copy()
    ctx_seed = ctx.copy()
    last_ts = seed.index[-1]
    future_days = next_weekdays(last_ts, 5)

    preds: List[Dict[str, Any]] = []

    for d in future_days:
        feats = make_features(seed, ctx_seed)
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[feature_cols]

        if len(feats) < window:
            needed = window - len(feats)
            last_feats = feats.iloc[[-1]]
            pad_df = pd.concat([last_feats] * needed, axis=0)
            feats = pd.concat([pad_df, feats], axis=0)

        window_slice = feats.values[-window:]

        if model_kind == "xgb":
            X_flat = window_slice.reshape(1, -1)
            models: List[XGBRegressor] = model_obj 
            y_hat_cols = [m.predict(X_flat)[0] for m in models]
            y_hat_scaled = np.array(y_hat_cols, dtype=float)
        elif model_kind == "rf":
            X_flat = window_slice.reshape(1, -1)
            y_hat_scaled = np.asarray(
                model_obj.predict(X_flat)[0], dtype=float
            ) 
        else:
            raise SystemExit(f"Unsupported model kind for prediction: {model_kind}")

        last_row = seed.iloc[-1]
        o, h, l, c = _decode_scaled_returns_to_ohlc(last_row, y_hat_scaled, seed)
        v = _as_float(last_row["Volume"])

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

# Global multi-ticker classifier

def _fit_xgb_classifier(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos == 0 or neg == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = neg / pos

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=4,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_train)
    return clf


def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    if probs.size == 0:
        return 0.5
    thresholds = np.linspace(0.3, 0.7, 41)
    best_thr = 0.5
    best_acc = 0.0
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return float(best_thr)


def _global_cls_paths(models_dir: Path) -> Tuple[Path, Path]:
    model_path = models_dir / f"global_cls_{GLOBAL_CLS_VERSION}.joblib"
    meta_path = models_dir / f"global_cls_{GLOBAL_CLS_VERSION}_meta.json"
    return model_path, meta_path


def _build_cls_dataset_for_ticker(
    ticker: str,
    start: str,
    end: str,
    feature_cols: List[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Build classifier dataset for one ticker. Returns (X_flat, y_labels, ts).
    On download or feature failure, returns empty arrays.
    """
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return np.empty((0, window * len(feature_cols))), np.empty((0,)), []

    if df.empty or len(df) < 260:
        return np.empty((0, window * len(feature_cols))), np.empty((0,)), []

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    ctx = _build_context(df.index, ticker, start, end)

    X_seq, y_seq, ts_seq = build_supervised_sequences(
        df, ctx, window=window, feature_cols=feature_cols
    )
    if X_seq.size == 0:
        return np.empty((0, window * len(feature_cols))), np.empty((0,)), []

    scaled_close_ret = y_seq[:, 3]
    mask = np.abs(scaled_close_ret) > CLS_RET_THRESHOLD
    if not mask.any():
        return np.empty((0, window * len(feature_cols))), np.empty((0,)), []

    X_seq_f = X_seq[mask]
    y_f = (scaled_close_ret[mask] > 0.0).astype(int)
    ts_f = [ts for ts, m in zip(ts_seq, mask) if m]

    n_samples, win, n_feats = X_seq_f.shape
    X_flat = X_seq_f.reshape(n_samples, win * n_feats)
    return X_flat, y_f, ts_f


def _train_global_classifier(
    models_dir: Path,
    universe: List[str],
    start: str,
    end: str,
    feature_cols: List[str],
    window: int,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train multi-ticker classifier on (strong-move) days across a universe.
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    uniq_universe = sorted(set(universe))

    for t in uniq_universe:
        X_flat, y_labels, _ = _build_cls_dataset_for_ticker(
            t, start=start, end=end, feature_cols=feature_cols, window=window
        )
        if X_flat.size == 0:
            continue
        all_X.append(X_flat)
        all_y.append(y_labels)

    if not all_X:
        raise SystemExit("Global classifier: no training data assembled.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    clf = _fit_xgb_classifier(X_all, y_all)

    meta = {
        "universe": uniq_universe,
        "start": start,
        "end": end,
        "n_samples": int(len(X_all)),
        "feature_dim": int(X_all.shape[1]),
    }

    model_path, meta_path = _global_cls_paths(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return clf, meta


def ensure_global_classifier(
    models_dir: Path,
    base_universe: List[str],
    extra_ticker: Optional[str],
    start: str,
    end: str,
    feature_cols: List[str],
    window: int,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Ensure a global classifier exists that covers at least base_universe + extra_ticker.
    Retrains if required.
    """
    model_path, meta_path = _global_cls_paths(models_dir)

    desired_universe = list(sorted(set(base_universe + ([extra_ticker] if extra_ticker else []))))

    if model_path.exists() and meta_path.exists():
        try:
            clf = joblib.load(model_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            trained_universe = meta.get("universe", [])
            if set(desired_universe).issubset(set(trained_universe)):
                return clf, meta
        except Exception:
            pass

    clf, meta = _train_global_classifier(
        models_dir=models_dir,
        universe=desired_universe,
        start=start,
        end=end,
        feature_cols=feature_cols,
        window=window,
    )
    return clf, meta


def evaluate_classifier_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    ctx: pd.DataFrame,
    clf: XGBClassifier,
    feature_cols: List[str],
    window: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate global classifier on one ticker using strong-move days + confidence filter.
    """
    X_seq, y_seq, ts_seq = build_supervised_sequences(
        df, ctx, window=window, feature_cols=feature_cols
    )
    if X_seq.size == 0:
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": 0,
            "cls_n_test": 0,
        }, []

    scaled_close_ret = y_seq[:, 3]
    mask = np.abs(scaled_close_ret) > CLS_RET_THRESHOLD
    if not mask.any():
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": 0,
            "cls_n_test": 0,
        }, []

    X_seq_f = X_seq[mask]
    y_f = (scaled_close_ret[mask] > 0.0).astype(int)
    ts_f = [ts for ts, m in zip(ts_seq, mask) if m]

    n_samples, win, n_feats = X_seq_f.shape
    X_flat = X_seq_f.reshape(n_samples, win * n_feats)

    split = int(0.8 * n_samples)
    if split <= 0 or split >= n_samples:
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": 0,
            "cls_n_test": 0,
        }, []

    X_train = X_flat[:split]
    y_train = y_f[:split]
    X_test = X_flat[split:]
    y_test = y_f[split:]
    ts_test = ts_f[split:]

    prob_train = clf.predict_proba(X_train)[:, 1]
    thr = _best_threshold(prob_train, y_train)

    prob_test = clf.predict_proba(X_test)[:, 1]
    pred_test_raw = (prob_test >= thr).astype(int)

    band = CLS_CONFIDENCE_BAND
    confident_mask = (prob_test >= thr + band) | (prob_test <= thr - band)
    if not confident_mask.any():
        return {
            "cls_accuracy": float("nan"),
            "cls_precision": float("nan"),
            "cls_recall": float("nan"),
            "cls_n_train": int(len(X_train)),
            "cls_n_test": 0,
        }, []

    y_true_c = y_test[confident_mask]
    y_pred_c = pred_test_raw[confident_mask]
    ts_c = [ts for ts, m in zip(ts_test, confident_mask) if m]
    prob_c = prob_test[confident_mask]

    acc = accuracy_score(y_true_c, y_pred_c)
    prec = precision_score(y_true_c, y_pred_c, zero_division=0)
    rec = recall_score(y_true_c, y_pred_c, zero_division=0)

    metrics = {
        "cls_accuracy": float(acc),
        "cls_precision": float(prec),
        "cls_recall": float(rec),
        "cls_n_train": int(len(X_train)),
        "cls_n_test": int(len(y_true_c)),
    }

    signals: List[Dict[str, Any]] = []
    for label, p, ts in zip(y_pred_c, prob_c, ts_c):
        if ts not in df.index:
            continue
        price = _as_float(df.loc[ts, "Close"])
        side = "buy" if label == 1 else "sell"
        signals.append(
            dict(
                date=ts.strftime("%Y-%m-%d"),
                side=side,
                price=price,
            )
        )

    return metrics, signals

# Payload builder

def build_payload(
    ticker: str,
    start: str,
    models_dir: str,
    model: str = "xgb",
) -> Dict[str, Any]:
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

    reg_model, reg_metrics = get_or_train_regressor(
        ticker=ticker,
        df=df,
        ctx=ctx,
        models_dir=models_path,
        model_kind=model_lower,
        feature_cols=feature_cols,
        window=window,
    )

    clf, cls_meta = ensure_global_classifier(
        models_dir=models_path,
        base_universe=BASE_UNIVERSE_TICKERS,
        extra_ticker=ticker,
        start=train_start_date,
        end=yf_end,
        feature_cols=feature_cols,
        window=window,
    )

    cls_metrics, signals = evaluate_classifier_for_ticker(
        ticker=ticker,
        df=df,
        ctx=ctx,
        clf=clf,
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
            "trades": int(cls_metrics.get("cls_n_test", 0)),
            "win_rate": float(cls_metrics.get("cls_accuracy", float("nan"))),
            "avg_gain": 0.0,
            "avg_loss": 0.0,
            "sharpe": 0.0,
        },
        data_until=str(yesterday),
    )
    return payload

# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Build UI payload or train global classifier")
    parser.add_argument("--ticker", required=False)
    parser.add_argument("--start", required=False, default=None)
    parser.add_argument("--end", required=False, default=None)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument(
        "--model",
        choices=["xgb", "rf"],
        default="xgb",
    )
    parser.add_argument(
        "--train_global_cls",
        action="store_true",
        help="Train the global multi-ticker classifier and exit.",
    )
    args = parser.parse_args()

    models_path = Path(args.models_dir)
    yesterday = now_ny().date() - timedelta(days=1)
    yf_end = (yesterday + timedelta(days=1)).strftime("%Y-%m-%d")
    train_start_date = (yesterday - timedelta(days=365 * TRAIN_YEARS)).strftime(
        "%Y-%m-%d"
    )

    if args.train_global_cls:
        ensure_global_classifier(
            models_dir=models_path,
            base_universe=BASE_UNIVERSE_TICKERS,
            extra_ticker=None,
            start=train_start_date,
            end=yf_end,
            feature_cols=FEATURE_COLS,
            window=WINDOW,
        )
        print("Global classifier trained.")
        return

    if not args.ticker:
        raise SystemExit("--ticker is required unless --train_global_cls is set.")

    payload = build_payload(
        ticker=args.ticker,
        start=train_start_date,
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
