// Set this to your deployed backend URL later
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
  const preds = payload.predictions_next5 || [];
  renderCandles(hist, preds);
}

function renderCandles(history, preds) {
  if (!history.length && !preds.length) {
    Plotly.newPlot("chart", [], { title: "No data" });
    return;
  }

  const all = history.concat(preds);

  const dates = all.map((c) => c.date);
  const open = all.map((c) => c.open);
  const high = all.map((c) => c.high);
  const low = all.map((c) => c.low);
  const close = all.map((c) => c.close);

  const trace = {
    x: dates,
    open,
    high,
    low,
    close,
    type: "candlestick",
    name: "Price",
  };

  const layout = {
    dragmode: "zoom",
    showlegend: false,
    margin: { t: 30, r: 10, b: 40, l: 50 },
    xaxis: { rangeslider: { visible: false } },
  };

  Plotly.newPlot("chart", [trace], layout);
}
