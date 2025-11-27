import json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QRadioButton, QGroupBox, QFileDialog, QMessageBox
)
import pyqtgraph as pg

# ---- Robust DateAxis import (covers different pyqtgraph versions) ----
try:
    from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem
except Exception:
    DateAxisItem = pg.graphicsItems.DateAxisItem.DateAxisItem

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APP_ROOT / "ui_config.json"

def load_config():
    if not CONFIG_PATH.exists():
        QMessageBox.critical(None, "Missing config", f"Cannot find {CONFIG_PATH}")
        raise SystemExit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

class StockApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Predictor (MVP)")
        self.resize(1200, 750)

        self.cfg = load_config()
        self.models_dir = Path(self.cfg["models_dir"]).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Hover lookup cache
        self.all_rows = []            # list of dicts (history + predicted)
        self.all_xs = np.array([], dtype=float)   # timestamps
        self.all_lows = np.array([], dtype=float)
        self.all_highs = np.array([], dtype=float)
        self.all_pred = np.array([], dtype=bool)
        self.candle_half_width = (24*60*60 * 0.50) / 2.0  # MATCH body_frac below (0.50)

        root = QVBoxLayout(self)

        # --- Top Controls ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Ticker:"))
        self.ticker_in = QLineEdit()
        self.ticker_in.setPlaceholderText("AAPL")
        top.addWidget(self.ticker_in)

        self.btn_fetch = QPushButton("Fetch & Predict")
        self.btn_fetch.clicked.connect(self.on_fetch)
        top.addWidget(self.btn_fetch)

        # Model choice
        model_box = QGroupBox("Model")
        hb = QHBoxLayout()
        self.rb_xgb = QRadioButton("XGBoost")
        self.rb_lr  = QRadioButton("Logistic Regression")
        self.rb_xgb.setChecked(True)
        hb.addWidget(self.rb_xgb)
        hb.addWidget(self.rb_lr)
        model_box.setLayout(hb)
        top.addWidget(model_box)

        # Config browse
        self.btn_models_dir = QPushButton("Choose models dir…")
        self.btn_models_dir.clicked.connect(self.on_choose_dir)
        top.addWidget(self.btn_models_dir)

        top.addStretch(1)
        root.addLayout(top)

        # --- Chart with Date axis (keeps your original layout) ---
        date_axis = DateAxisItem(orientation='bottom')
        self.plot = pg.PlotWidget(axisItems={'bottom': date_axis})
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Price (adjusted)")
        self.plot.setLabel("bottom", "Date")
        self.plot.getPlotItem().getViewBox().setDefaultPadding(0.05)  # ~5% edge padding
        root.addWidget(self.plot, stretch=2)

        # --- Metrics + JSON preview (unchanged) ---
        bottom = QHBoxLayout()
        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setPlaceholderText("Metrics will appear here…")
        bottom.addWidget(self.metrics_view, stretch=1)

        self.json_view = QTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setPlaceholderText("Payload preview will appear here…")
        bottom.addWidget(self.json_view, stretch=1)

        root.addLayout(bottom, stretch=1)

        # --- Persistent hover label (in-chart, not OS tooltip) ---
        self.hover_label = pg.TextItem("", anchor=(0,1))  # bottom-left of text sits at given pos
        self.hover_label.setZValue(200)
        self.hover_label.setVisible(False)
        self.plot.addItem(self.hover_label)

        # Mouse move connection
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # ---------- helpers ----------
    def on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select models dir", str(self.models_dir))
        if d:
            self.models_dir = Path(d)

    def backend_module_cmd(self, ticker: str):
        backend_root = Path(self.cfg["backend_root"]).resolve()
        start = self.cfg["start_date"]
        end = self.cfg.get("end_date", "")
        model = "xgb" if self.rb_xgb.isChecked() else "logreg"
        return [
            sys.executable, "-m", "backend.emit_ui_payload",
            "--ticker", ticker, "--start", start, "--end", end,
            "--models_dir", str(self.models_dir),
            "--model", model
        ], backend_root

    @staticmethod
    def _parse_date_to_ts(date_str: str) -> float:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def _update_hover_cache(self, hist, preds):
        rows, xs, lows, highs, preds_flag = [], [], [], [], []
        for c in hist:
            ts = self._parse_date_to_ts(c["date"])
            r = dict(c); r["_predicted"] = False; r["_ts"] = ts
            rows.append(r); xs.append(ts); lows.append(float(c["low"])); highs.append(float(c["high"])); preds_flag.append(False)
        for c in preds:
            ts = self._parse_date_to_ts(c["date"])
            r = dict(c); r["_predicted"] = True; r["_ts"] = ts
            rows.append(r); xs.append(ts); lows.append(float(c["low"])); highs.append(float(c["high"])); preds_flag.append(True)
        self.all_rows  = rows
        self.all_xs    = np.array(xs, dtype=float)  if xs   else np.array([], dtype=float)
        self.all_lows  = np.array(lows, dtype=float) if lows else np.array([], dtype=float)
        self.all_highs = np.array(highs, dtype=float) if highs else np.array([], dtype=float)
        self.all_pred  = np.array(preds_flag, dtype=bool) if preds_flag else np.array([], dtype=bool)

    def _format_row_html(self, row):
        # compact, readable monospace
        flag = " <b>(predicted)</b>" if row.get("_predicted") else ""
        vol = int(row.get("volume", 0))
        return (
            f"<span style='font-family:monospace'>"
            f"Date: <b>{row['date']}</b>{flag}<br>"
            f"O:{row['open']:.2f}  H:{row['high']:.2f}  "
            f"L:{row['low']:.2f}  C:<b>{row['close']:.2f}</b><br>"
            f"Vol: {vol:n}"
            f"</span>"
        )

    def _on_mouse_moved(self, pos):
        """Show label while cursor is within a candle hit-box; hide otherwise."""
        if self.all_xs.size == 0:
            self.hover_label.setVisible(False)
            return

        if not self.plot.sceneBoundingRect().contains(pos):
            self.hover_label.setVisible(False)
            return

        vb = self.plot.getViewBox()
        mp = vb.mapSceneToView(pos)
        x, y = mp.x(), mp.y()

        # Nearest timestamp
        idx = int(np.argmin(np.abs(self.all_xs - x)))
        ts  = self.all_xs[idx]

        # Hit-test in X: within candle body width (same width you draw with: body_frac=0.50)
        if abs(x - ts) > self.candle_half_width:
            self.hover_label.setVisible(False)
            return

        # Hit-test in Y: inside candle's high/low (with a small tolerance, 0.2% of price)
        low, high = self.all_lows[idx], self.all_highs[idx]
        tol = max(0.002 * max(abs(high), 1.0), 0.02)  # 0.2% or 0.02 absolute min
        if not (low - tol <= y <= high + tol):
            self.hover_label.setVisible(False)
            return

        # Passed hit-test: show label with this row
        row = self.all_rows[idx]
        self.hover_label.setHtml(self._format_row_html(row))

        # Position label slightly above/right of cursor (in data coords)
        # Offset ~1/6 candle width horizontally, ~1.5% of price range vertically
        x_off = self.candle_half_width * 0.33
        yr = vb.viewRange()[1]
        y_off = 0.015 * (yr[1] - yr[0])
        self.hover_label.setPos(x + x_off, y + y_off)
        self.hover_label.setVisible(True)

    # ---------- Candle plotting ----------
    def _plot_candles(self, ohlc, *, body_frac: float = 0.50,
                      up_color=(0, 170, 0), down_color=(200, 0, 0),
                      dashed_outline=False, z=10):
        """
        Render candlesticks at POSIX timestamps using:
          - ErrorBarItem for wicks
          - BarGraphItem for bodies
        body_frac: fraction of one day used for candle width (0..1)
        """
        if not ohlc:
            return

        opens  = np.array([float(c["open"])  for c in ohlc])
        highs  = np.array([float(c["high"])  for c in ohlc])
        lows   = np.array([float(c["low"])   for c in ohlc])
        closes = np.array([float(c["close"]) for c in ohlc])
        xs     = np.array([self._parse_date_to_ts(c["date"]) for c in ohlc], dtype=float)

        DAY = 24 * 60 * 60
        body_width = DAY * max(0.0, min(1.0, body_frac))

        # Keep hover hit-test in sync with what we draw
        self.candle_half_width = body_width / 2.0

        up_mask   = closes >= opens
        down_mask = ~up_mask

        # Wicks
        mids = (highs + lows) / 2.0
        wick = pg.ErrorBarItem(
            x=xs, y=mids, top=highs - mids, bottom=mids - lows, beam=0,
            pen=pg.mkPen((120, 120, 120), width=1)
        )
        wick.setZValue(z)
        self.plot.addItem(wick)

        # Bodies
        def add_bodies(mask, color, alpha=180, dash=False):
            if not np.any(mask):
                return
            x = xs[mask]
            o = opens[mask]
            c = closes[mask]
            y0 = np.minimum(o, c)
            h  = np.abs(c - o)
            brush = pg.mkBrush(color[0], color[1], color[2], 0 if dash else alpha)
            pen = pg.mkPen(color, width=1.5,
                           style=Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
            bodies = pg.BarGraphItem(x=x, height=h, width=body_width, y0=y0, pen=pen, brush=brush)
            bodies.setZValue(z+1)
            self.plot.addItem(bodies)

        if not dashed_outline:
            add_bodies(up_mask,   up_color, alpha=180, dash=False)
            add_bodies(down_mask, down_color, alpha=180, dash=False)
        else:
            # Predicted candles: dashed outline, transparent fill
            neutral = (30, 120, 200)
            add_bodies(np.ones_like(up_mask, dtype=bool), neutral, alpha=0, dash=True)

    # ---------- main action ----------
    def on_fetch(self):
        ticker = (self.ticker_in.text() or "AAPL").upper().strip()
        cmd, cwd = self.backend_module_cmd(ticker)

        try:
            proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                QMessageBox.warning(
                    self, "Backend error",
                    f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{proc.stderr or '(none)'}"
                )
                return
        except Exception as e:
            QMessageBox.critical(self, "Execution error", str(e))
            return

        payload_path = self.models_dir / f"ui_payload_{ticker}.json"
        if not payload_path.exists():
            QMessageBox.warning(self, "Missing payload", f"Expected {payload_path} but it was not created.")
            return

        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Metrics panel
        met = payload.get("metrics", {})
        met_lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in met.items()]
        self.metrics_view.setPlainText(
            f"Ticker: {payload.get('ticker','')}\nModel: {payload.get('model','')}\n(Adjusted OHLC)\n\n" +
            ("\n".join(met_lines) if met_lines else "No metrics")
        )

        # JSON preview (unchanged)
        self.json_view.setPlainText(json.dumps(payload, indent=2))

        # Candles
        hist = payload.get("history", [])
        preds = payload.get("predictions_next5", [])

        self.plot.clear()
        if hist:
            self._plot_candles(hist, body_frac=0.50, up_color=(0,160,0), down_color=(200,0,0), dashed_outline=False)
        if preds:
            self._plot_candles(preds, body_frac=0.50, dashed_outline=True)

        # Update hover cache for persistent label
        self._update_hover_cache(hist, preds)
        # Ensure hover label is brought back on top after clearing plot
        self.plot.addItem(self.hover_label)
        self.hover_label.setVisible(False)

def main():
    app = QApplication(sys.argv)
    w = StockApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()