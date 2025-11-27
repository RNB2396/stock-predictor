// Backend base URL
const API_BASE = "http://localhost:8000"; // dev

const tickerInput = document.getElementById("ticker");
const runBtn = document.getElementById("run-btn");
const metricsPre = document.getElementById("metrics");
const payloadPre = document.getElementById("payload");

runBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  const ticker = (tickerInput.value || "AAPL").trim().toUpperCase();
  if (!ticker) return;

  const modelRadio = document.querySelector('input[name="model"]:checked');
  const model = modelRadio ? modelRadio.value : "xgb";

  metricsPre.textContent = "Loading...";
  payloadPre.textContent = "";
  Plotly.purge("chart");

  try {
    const url = `${API_BASE}/api/predict?ticker=${encodeURIComponent(
      ticker
    )}&model=${encodeURIComponent(model)}`;

    const res = await fetch(url);
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`Backend error: ${res.status} ${msg}`);
    }

    const payload = await res.json();
    renderPayload(payload);
  } catch (err) {
    console.error(err);
    metricsPre.textContent = String(err);
  }
}

function renderPayload(payload) {
  // Metrics
  const met = payload.metrics || {};
  const lines = [
    `Ticker: ${payload.ticker || ""}`,
    `Model: ${payload.model || ""}`,
    "",
    ...Object.entries(met).map(([k, v]) =>
      typeof v === "number" ? `${k}: ${v.toFixed(4)}` : `${k}: ${v}`
    ),
  ];
  metricsPre.textContent = lines.join("\n");

  // Raw payload JSON
  payloadPre.textContent = JSON.stringify(payload, null, 2);

  // Chart
  const hist = payload.history || [];
  const preds = payload.predictions_next5 || payload.predictions || [];
  renderCandles(hist, preds);
}

function renderCandles(history, preds) {
  const histArr = Array.isArray(history) ? history : [];
  const predArr = Array.isArray(preds) ? preds : [];

  console.log("history len:", histArr.length);
  console.log("predictions_next5 len:", predArr.length);
  console.log("pred samples:", predArr.slice(0, 5));

  if (!histArr.length && !predArr.length) {
    Plotly.newPlot("chart", [], { title: "No data" });
    return;
  }

  // History arrays
  const histDates = histArr.map((c) => c.date || c.datetime || c.time);
  const histOpen = histArr.map((c) => c.open);
  const histHigh = histArr.map((c) => c.high);
  const histLow = histArr.map((c) => c.low);
  const histClose = histArr.map((c) => c.close);

  // Prediction arrays
  const predDates = predArr.map((c) => c.date || c.datetime || c.time);
  const predOpen = predArr.map((c) => c.open);
  const predHigh = predArr.map((c) => c.high);
  const predLow = predArr.map((c) => c.low);
  const predClose = predArr.map((c) => c.close);

  const traces = [];

  if (histDates.length) {
    traces.push({
      x: histDates,
      open: histOpen,
      high: histHigh,
      low: histLow,
      close: histClose,
      type: "candlestick",
      name: "History",
    });
  }

  if (predDates.length) {
    traces.push({
      x: predDates,
      open: predOpen,
      high: predHigh,
      low: predLow,
      close: predClose,
      type: "candlestick",
      name: "Predicted (next 5)",
      increasing: { line: { width: 1.5 } },
      decreasing: { line: { width: 1.5 } },
    });
  }

  const layout = {
    dragmode: "zoom",
    showlegend: true,
    legend: { x: 0, y: 1.1, orientation: "h" },
    margin: { t: 40, r: 10, b: 40, l: 50 },
    xaxis: { rangeslider: { visible: false } },
  };

  Plotly.newPlot("chart", traces, layout, { responsive: true });
}
