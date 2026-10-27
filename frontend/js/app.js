/**
 * RampSense Dashboard Frontend Client
 * Communicates with the Python HTTP Backend and renders interactive SCADA views,
 * charts, metrics, live inference tests, and ablation analytics.
 */

const API_BASE = (window.location.protocol === "http:" || window.location.protocol === "https:") 
  ? window.location.origin 
  : "http://127.0.0.1:8000";

// Application State
const state = {
  currentTab: "tab-live",
  isStreaming: true,
  streamInterval: null,
  pvThreshold: 12.25,
  windThreshold: 205.0,
  pvTimeseries: [],
  windTimeseries: [],
  pvSweepData: null,
  pvModels: [],
  windModels: [],
  windAblation: [],
  windDirection: [],
  charts: {}
};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initThresholdSliders();
  initPresets();
  initPlayground();
  initPVSliceSelector();
  
  // Load initial backend datasets
  loadSystemStatus();
  loadOverview();
  loadPVData();
  loadWindData();
  loadBenchmarkData();

  // Run initial preset so playground has live chart & prediction immediately
  setTimeout(() => {
    runPresetInference("cloud_drop");
  }, 400);

  // Start live telemetry polling
  startTelemetryStream();

  // SCADA Clock
  setInterval(updateClock, 1000);
  updateClock();
});

function initPVSliceSelector() {
  const btns = document.querySelectorAll(".pv-slice-btn");
  btns.forEach(btn => {
    btn.addEventListener("click", async () => {
      btns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.dataset.mode;
      await switchPVSlice(mode);
    });
  });
}

async function switchPVSlice(mode) {
  try {
    await fetch(`${API_BASE}/api/pv/slice?mode=${mode}`);
    // Reload PV timeseries
    const tsRes = await fetch(`${API_BASE}/api/pv/timeseries?limit=250`);
    const tsData = await tsRes.json();
    state.pvTimeseries = tsData.data || [];
    renderPVTimeseriesChart();
    updatePVThresholdEvaluation(state.pvThreshold);
  } catch (err) {
    console.error("Error switching PV slice:", err);
  }
}

function updateClock() {
  const el = document.getElementById("scada-clock");
  if (el) {
    const now = new Date();
    el.innerText = now.toUTCString().replace("GMT", "UTC");
  }
}

// ============================================================
// TABS NAVIGATION
// ============================================================
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.dataset.target;
      state.currentTab = targetId;

      document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.remove("active");
      });

      const activeContent = document.getElementById(targetId);
      if (activeContent) {
        activeContent.classList.add("active");
        renderCurrentTabCharts(targetId);
      }
    });
  });
}

function renderCurrentTabCharts(tabId) {
  if (tabId === "tab-pv" && !state.charts.pvTimeseries) {
    renderPVTimeseriesChart();
    renderPVSweepChart();
  } else if (tabId === "tab-wind" && !state.charts.windTimeseries) {
    renderWindTimeseriesChart();
    renderWindDirectionChart();
  } else if (tabId === "tab-benchmarks" && !state.charts.weightSweep) {
    renderWeightSweepChart();
  }
}

// ============================================================
// BACKEND API CLIENT CALLS
// ============================================================
async function loadSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error("Status check failed");
    const data = await res.json();
    
    const badge = document.getElementById("backend-status-badge");
    if (badge) {
      badge.innerHTML = `<span class="status-dot"></span> BACKEND CONNECTED (${data.device.toUpperCase()})`;
      badge.className = "status-pill";
    }

    const sysInfo = document.getElementById("system-info-summary");
    if (sysInfo) {
      sysInfo.innerHTML = `Model Device: <strong>${data.device}</strong> | PV Model: <span class="badge badge-green">Loaded</span> | Records: <strong>${data.dataset_records.pv_sample_points + data.dataset_records.wind_sample_points}</strong>`;
    }
  } catch (err) {
    console.error("Backend offline:", err);
    const badge = document.getElementById("backend-status-badge");
    if (badge) {
      badge.innerHTML = `<span class="status-dot" style="background:#f43f5e;"></span> BACKEND OFFLINE`;
      badge.className = "status-pill";
      badge.style.color = "#f43f5e";
      badge.style.borderColor = "rgba(244,63,94,0.3)";
    }
  }
}

async function loadOverview() {
  try {
    const res = await fetch(`${API_BASE}/api/overview`);
    const data = await res.json();
    
    // Update overview metric labels
    document.getElementById("stat-pv-rating").innerText = data.pv_system.nominal_rating;
    document.getElementById("stat-pv-best-f1").innerText = data.pv_system.best_test_f1;
    document.getElementById("stat-wind-rating").innerText = data.wind_system.nominal_rating;
    document.getElementById("stat-wind-best-f1").innerText = `${data.wind_system.best_test_f1}%`;
    document.getElementById("stat-wind-dir-acc").innerText = data.wind_system.direction_accuracy;
  } catch (err) {
    console.error("Error loading overview:", err);
  }
}

async function loadPVData() {
  try {
    // 1. Load Timeseries sample
    const tsRes = await fetch(`${API_BASE}/api/pv/timeseries?limit=200`);
    const tsData = await tsRes.json();
    state.pvTimeseries = tsData.data || [];

    // 2. Load Sweep data
    const swRes = await fetch(`${API_BASE}/api/pv/sweep`);
    state.pvSweepData = await swRes.json();

    // 3. Load Models
    const mRes = await fetch(`${API_BASE}/api/pv/models`);
    const mData = await mRes.json();
    state.pvModels = mData.models || [];

    renderPVTable(state.pvModels);
    renderPVTimeseriesChart();
    renderPVSweepChart();
    updatePVThresholdEvaluation(state.pvThreshold);
  } catch (err) {
    console.error("Error loading PV data:", err);
  }
}

async function loadWindData() {
  try {
    // 1. Wind timeseries
    const tsRes = await fetch(`${API_BASE}/api/wind/timeseries?limit=200`);
    const tsData = await tsRes.json();
    state.windTimeseries = tsData.data || [];

    // 2. Wind Models & Direction
    const mRes = await fetch(`${API_BASE}/api/wind/models`);
    const mData = await mRes.json();
    state.windModels = mData.models || [];
    state.windAblation = mData.ablation || [];
    state.windDirection = mData.direction || [];

    renderWindTable(state.windModels);
    renderWindDirectionTable(state.windDirection);
    renderWindTimeseriesChart();
    renderWindDirectionChart();
  } catch (err) {
    console.error("Error loading Wind data:", err);
  }
}

async function loadBenchmarkData() {
  try {
    const res = await fetch(`${API_BASE}/api/wind/ablation`);
    const data = await res.json();
    renderAblationTable(data.summary || []);
    renderWeightSweepChart(data.sweep || []);
  } catch (err) {
    console.error("Error loading benchmarks:", err);
  }
}

// ============================================================
// LIVE SCADA TELEMETRY STREAM
// ============================================================
function startTelemetryStream() {
  if (state.streamInterval) clearInterval(state.streamInterval);
  state.streamInterval = setInterval(fetchTelemetryTick, 2500);
}

async function fetchTelemetryTick() {
  if (!state.isStreaming) return;
  try {
    const res = await fetch(`${API_BASE}/api/telemetry/live`);
    if (!res.ok) return;
    const tick = await res.json();

    // Update Live Gauges
    const pvPwr = document.getElementById("gauge-pv-power");
    const pvRamp = document.getElementById("gauge-pv-ramp");
    const windPwr = document.getElementById("gauge-wind-power");
    const windSpeed = document.getElementById("gauge-wind-speed");

    if (pvPwr) pvPwr.innerText = `${tick.pv.power_w.toFixed(1)} W`;
    if (pvRamp) {
      pvRamp.innerText = `${tick.pv.ramp_rate_w_min.toFixed(1)} W/min`;
      pvRamp.style.color = tick.pv.is_ramp ? "var(--accent-rose)" : "var(--accent-emerald)";
    }
    if (windPwr) windPwr.innerText = `${tick.wind.power_kw.toFixed(1)} kW`;
    if (windSpeed) windSpeed.innerText = `${tick.wind.wind_speed_ms.toFixed(1)} m/s`;

    // Alert Banner Updates
    const banner = document.getElementById("global-alert-banner");
    const alertText = document.getElementById("alert-banner-text");
    if (tick.pv.alert === "CRITICAL" || tick.wind.alert === "CRITICAL") {
      banner.style.display = "flex";
      banner.style.background = "linear-gradient(90deg, rgba(244,63,94,0.4), rgba(245,158,11,0.3))";
      alertText.innerHTML = `<strong>⚠️ CRITICAL RAMP EVENT DETECTED:</strong> ${tick.wind.is_ramp ? `Penmanshiel WT15 ΔP: ${tick.wind.delta_power_kw} kW (${tick.wind.direction})` : `PV Power Ramp: ${tick.pv.ramp_rate_w_min} W/min`}`;
    } else if (tick.pv.is_ramp || tick.wind.is_ramp) {
      banner.style.display = "flex";
      banner.style.background = "linear-gradient(90deg, rgba(245,158,11,0.25), rgba(6,182,212,0.2))";
      alertText.innerHTML = `<strong>⚡ RAMP ALERT:</strong> Ramp event predicted at ${tick.timestamp.split(" ")[1] || tick.timestamp}`;
    } else {
      banner.style.background = "linear-gradient(90deg, rgba(16,185,129,0.15), rgba(6,182,212,0.15))";
      alertText.innerHTML = `<strong>🟢 GRID GENERATION STABLE:</strong> No anomalous ramp conditions detected across SolarTech & Penmanshiel SCADA.`;
    }

    // Append to SCADA event feed
    appendSCADAEvent(tick);
  } catch (err) {
    // console.warn("Live stream tick failed");
  }
}

function appendSCADAEvent(tick) {
  const feed = document.getElementById("scada-event-feed");
  if (!feed) return;

  const row = document.createElement("div");
  row.className = "event-row";
  row.style.display = "flex";
  row.style.alignItems = "center";
  row.style.justifyContent = "space-between";
  row.style.padding = "0.45rem 0.6rem";
  row.style.borderBottom = "1px solid rgba(45,62,100,0.3)";
  row.style.fontSize = "0.78rem";

  const timeStr = tick.timestamp ? tick.timestamp.split(" ")[1] || tick.timestamp : new Date().toLocaleTimeString();
  const isRamp = tick.pv.is_ramp || tick.wind.is_ramp;
  const badgeClass = isRamp ? "badge-red" : "badge-green";
  const badgeText = isRamp ? (tick.wind.is_ramp ? `WIND ${tick.wind.direction}` : "PV RAMP") : "STABLE";

  row.innerHTML = `
    <span style="color:var(--text-muted); font-family:monospace;">${timeStr}</span>
    <span>PV: <strong>${tick.pv.power_w.toFixed(0)}W</strong> | Wind: <strong>${tick.wind.power_kw.toFixed(0)}kW</strong></span>
    <span class="badge ${badgeClass}">${badgeText}</span>
  `;

  feed.prepend(row);
  if (feed.children.length > 8) {
    feed.removeChild(feed.lastChild);
  }
}

// Stream toggle button
function toggleStream() {
  state.isStreaming = !state.isStreaming;
  const btn = document.getElementById("btn-toggle-stream");
  if (btn) {
    btn.innerHTML = state.isStreaming ? `⏸ Pause Telemetry` : `▶ Resume Telemetry`;
    btn.className = state.isStreaming ? "btn btn-secondary" : "btn";
  }
}

// ============================================================
// THRESHOLD SLIDERS & DYNAMIC INTERACTION
// ============================================================
function initThresholdSliders() {
  const pvSlider = document.getElementById("pv-threshold-slider");
  const pvVal = document.getElementById("pv-threshold-val");
  if (pvSlider && pvVal) {
    pvSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      state.pvThreshold = val;
      pvVal.innerText = `${val.toFixed(2)} W/min`;
      updatePVThresholdEvaluation(val);
    });
  }

  const windSlider = document.getElementById("wind-threshold-slider");
  const windVal = document.getElementById("wind-threshold-val");
  if (windSlider && windVal) {
    windSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      state.windThreshold = val;
      windVal.innerText = `${val.toFixed(1)} kW`;
      updateWindThresholdEvaluation(val);
    });
  }
}

function updatePVThresholdEvaluation(threshold) {
  if (!state.pvTimeseries.length) return;
  let oreCount = 0;
  let preCount = 0;
  let tp = 0;

  state.pvTimeseries.forEach(pt => {
    const isOre = pt.actual_max_ramp >= 12.25; // ground truth fixed
    const isPre = pt.pred_max_ramp >= threshold;
    if (isOre) oreCount++;
    if (isPre) preCount++;
    if (isOre && isPre) tp++;
  });

  const precision = preCount > 0 ? (tp / preCount) : 0;
  const recall = oreCount > 0 ? (tp / oreCount) : 0;
  const f1 = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0;

  const f1Display = document.getElementById("dynamic-pv-f1");
  const preDisplay = document.getElementById("dynamic-pv-pre-count");
  if (f1Display) f1Display.innerText = f1.toFixed(3);
  if (preDisplay) preDisplay.innerText = `${preCount} / ${state.pvTimeseries.length} windows`;

  // Re-draw chart threshold line
  renderPVTimeseriesChart();
}

function updateWindThresholdEvaluation(threshold) {
  if (!state.windTimeseries.length) return;
  let rampCount = 0;
  state.windTimeseries.forEach(pt => {
    if (Math.abs(pt.delta_power) >= threshold) rampCount++;
  });

  const rampCountDisplay = document.getElementById("dynamic-wind-ramp-count");
  if (rampCountDisplay) rampCountDisplay.innerText = `${rampCount} ramp events (${(rampCount / state.windTimeseries.length * 100).toFixed(1)}%)`;
  renderWindTimeseriesChart();
}

// ============================================================
// PRESET SCENARIOS & PLAYGROUND INFERENCE
// ============================================================
function initPresets() {
  const presetBtns = document.querySelectorAll(".preset-btn");
  presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      presetBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const presetId = btn.dataset.preset;
      runPresetInference(presetId);
    });
  });
}

async function runPresetInference(presetId) {
  try {
    const res = await fetch(`${API_BASE}/api/pv/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset: presetId, threshold: state.pvThreshold })
    });
    const result = await res.json();
    renderInferenceResult(result);
  } catch (err) {
    console.error("Preset inference error:", err);
  }
}

function initPlayground() {
  const btnRunCustomPV = document.getElementById("btn-run-custom-pv");
  if (btnRunCustomPV) {
    btnRunCustomPV.addEventListener("click", () => {
      const activePreset = document.querySelector(".preset-btn.active");
      const presetId = activePreset ? activePreset.dataset.preset : "cloud_drop";
      runPresetInference(presetId);
    });
  }

  const btnRunWindInference = document.getElementById("btn-run-wind-inference");
  if (btnRunWindInference) {
    btnRunWindInference.addEventListener("click", async () => {
      const pwr = parseFloat(document.getElementById("input-wind-pwr").value);
      const ws = parseFloat(document.getElementById("input-wind-ws").value);
      const wd = parseFloat(document.getElementById("input-wind-wd").value);
      const prev = parseFloat(document.getElementById("input-wind-prev").value);

      try {
        const res = await fetch(`${API_BASE}/api/wind/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            current_power: pwr,
            wind_speed: ws,
            wind_direction: wd,
            prev_power: prev,
            threshold: state.windThreshold
          })
        });
        const result = await res.json();
        renderWindInferenceResult(result);
      } catch (err) {
        console.error("Wind inference error:", err);
      }
    });
  }
}

function renderInferenceResult(result) {
  // Render output cards
  const rampRate = document.getElementById("res-pv-max-ramp");
  const rampStatus = document.getElementById("res-pv-status");
  const latency = document.getElementById("res-pv-latency");
  const jsonBox = document.getElementById("res-pv-json");

  if (rampRate) rampRate.innerText = `${result.max_ramp_w_per_min} W/min`;
  if (rampStatus) {
    rampStatus.innerText = result.ramp_direction;
    rampStatus.className = `badge ${result.is_ramp_event ? (result.alert_severity === "CRITICAL" ? "badge-red" : "badge-amber") : "badge-green"}`;
  }
  if (latency) latency.innerText = `${result.latency_ms} ms (${result.device})`;
  if (jsonBox) jsonBox.innerText = JSON.stringify(result, null, 2);

  // Render 8-step predicted power trajectory chart
  renderTrajectoryChart(result.forecast_steps, result.forecast_power_w);
}

function renderWindInferenceResult(result) {
  const deltaP = document.getElementById("res-wind-delta");
  const rampStatus = document.getElementById("res-wind-status");
  const latency = document.getElementById("res-wind-latency");
  const jsonBox = document.getElementById("res-wind-json");

  if (deltaP) deltaP.innerText = `${result.delta_power_kw > 0 ? '+' : ''}${result.delta_power_kw} kW`;
  if (rampStatus) {
    rampStatus.innerText = `${result.ramp_direction} (${result.severity})`;
    rampStatus.className = `badge ${result.is_ramp_event ? "badge-red" : "badge-green"}`;
  }
  if (latency) latency.innerText = `${result.latency_ms} ms`;
  if (jsonBox) jsonBox.innerText = JSON.stringify(result, null, 2);
}

// ============================================================
// CHART RENDERING (Chart.js + Resilient Canvas Fallback)
// ============================================================

function renderPVTimeseriesChart() {
  const canvas = document.getElementById("chart-pv-timeseries");
  if (!canvas || !state.pvTimeseries.length) return;

  const labels = state.pvTimeseries.map(d => d.time.split(" ")[1] || d.time);
  const actualPower = state.pvTimeseries.map(d => d.actual_power);
  const predPower = state.pvTimeseries.map(d => d.pred_power);
  const actualRamp = state.pvTimeseries.map(d => d.actual_max_ramp);

  if (window.Chart) {
    if (state.charts.pvTimeseries) state.charts.pvTimeseries.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.pvTimeseries = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Actual PV Power (W)",
            data: actualPower,
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.1)",
            borderWidth: 2,
            tension: 0.2,
            fill: true,
            yAxisID: "y"
          },
          {
            label: "Predicted Multi-step Power (W)",
            data: predPower,
            borderColor: "#a855f7",
            borderDash: [4, 4],
            borderWidth: 2,
            tension: 0.2,
            yAxisID: "y"
          },
          {
            label: "Ramp Rate (W/min)",
            data: actualRamp,
            borderColor: "#f59e0b",
            borderWidth: 1.5,
            pointRadius: 0,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", maxTicksLimit: 12 } },
          y: {
            title: { display: true, text: "Power (W)", color: "#06b6d4" },
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y1: {
            position: "right",
            title: { display: true, text: "Ramp Rate (W/min)", color: "#f59e0b" },
            grid: { drawOnChartArea: false },
            ticks: { color: "#f59e0b" }
          }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  } else {
    drawFallbackLineChart(canvas, labels, [
      { name: "Actual Power", data: actualPower, color: "#06b6d4" },
      { name: "Predicted Power", data: predPower, color: "#a855f7" }
    ]);
  }
}

function renderPVSweepChart() {
  const canvas = document.getElementById("chart-pv-sweep");
  if (!canvas || !state.pvSweepData) return;

  const thresholds = state.pvSweepData.thresholds;
  const adamF1 = state.pvSweepData.adam_mse_f1;
  const psoF1 = state.pvSweepData.pso_warm_start_f1;

  if (window.Chart) {
    if (state.charts.pvSweep) state.charts.pvSweep.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.pvSweep = new Chart(ctx, {
      type: "line",
      data: {
        labels: thresholds,
        datasets: [
          {
            label: "PSO Warm-Start (Ramp-Aware F1)",
            data: psoF1,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.15)",
            borderWidth: 2.5,
            fill: true,
            tension: 0.3
          },
          {
            label: "Adam MSE Baseline (Smooth Level Only)",
            data: adamF1,
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244, 63, 94, 0.1)",
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: "Decision Threshold (W/min)", color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#94a3b8", maxTicksLimit: 10 }
          },
          y: {
            title: { display: true, text: "Validation F1 Score", color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#94a3b8" },
            min: 0,
            max: 0.5
          }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  } else {
    drawFallbackLineChart(canvas, thresholds, [
      { name: "PSO Warm-Start", data: psoF1, color: "#10b981" },
      { name: "Adam Baseline", data: adamF1, color: "#f43f5e" }
    ]);
  }
}

function renderWindTimeseriesChart() {
  const canvas = document.getElementById("chart-wind-timeseries");
  if (!canvas || !state.windTimeseries.length) return;

  const labels = state.windTimeseries.map(d => d.time.split(" ")[1] || d.time);
  const power = state.windTimeseries.map(d => d.power);
  const windSpeed = state.windTimeseries.map(d => d.wind_speed);

  if (window.Chart) {
    if (state.charts.windTimeseries) state.charts.windTimeseries.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.windTimeseries = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Turbine Active Power (kW)",
            data: power,
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.1)",
            borderWidth: 2,
            tension: 0.2,
            fill: true,
            yAxisID: "y"
          },
          {
            label: "Wind Speed (m/s)",
            data: windSpeed,
            borderColor: "#10b981",
            borderWidth: 1.5,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8", maxTicksLimit: 12 } },
          y: {
            title: { display: true, text: "Active Power (kW)", color: "#38bdf8" },
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y1: {
            position: "right",
            title: { display: true, text: "Wind Speed (m/s)", color: "#10b981" },
            grid: { drawOnChartArea: false },
            ticks: { color: "#10b981" }
          }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  } else {
    drawFallbackLineChart(canvas, labels, [
      { name: "Power (kW)", data: power, color: "#38bdf8" },
      { name: "Wind Speed", data: windSpeed, color: "#10b981" }
    ]);
  }
}

function renderWindDirectionChart() {
  const canvas = document.getElementById("chart-wind-direction");
  if (!canvas) return;

  const models = ["MSE Baseline", "Soft-F1 (phi_soft)", "Signed Loss (phi_signed)", "Selected Lag-6"];
  const f1Up = [0.4, 0.0, 16.22, 25.09];
  const f1Down = [0.01, 31.42, 25.00, 28.51];
  const dirAcc = [48.58, 54.49, 61.03, 66.31];

  if (window.Chart) {
    if (state.charts.windDirection) state.charts.windDirection.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.windDirection = new Chart(ctx, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Ramp-Up F1 Score (%)",
            data: f1Up,
            backgroundColor: "#06b6d4"
          },
          {
            label: "Ramp-Down F1 Score (%)",
            data: f1Down,
            backgroundColor: "#f59e0b"
          },
          {
            label: "Direction Accuracy (%)",
            data: dirAcc,
            backgroundColor: "#10b981"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" }, max: 100 }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  }
}

function renderWeightSweepChart(sweepData) {
  const canvas = document.getElementById("chart-weight-sweep");
  if (!canvas) return;

  // Use provided or standard weight sweep values from research
  const w1Vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const testF1 = [38.88, 47.17, 46.98, 47.05, 46.93, 46.80, 45.98, 45.63, 44.09, 33.49, 0.42];
  const testRMSE = [42.50, 11.60, 10.82, 10.56, 10.42, 10.10, 9.77, 9.57, 9.32, 9.28, 7.85];

  if (window.Chart) {
    if (state.charts.weightSweep) state.charts.weightSweep.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.weightSweep = new Chart(ctx, {
      type: "line",
      data: {
        labels: w1Vals.map(w => `w1 = ${w.toFixed(1)}`),
        datasets: [
          {
            label: "Test F1 Score (%)",
            data: testF1,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.15)",
            borderWidth: 2.5,
            fill: true,
            yAxisID: "y"
          },
          {
            label: "Test RMSE (% Rated)",
            data: testRMSE,
            borderColor: "#f43f5e",
            borderWidth: 2,
            borderDash: [4, 4],
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: {
            title: { display: true, text: "F1 Score (%)", color: "#10b981" },
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#10b981" }
          },
          y1: {
            position: "right",
            title: { display: true, text: "RMSE (% Rated)", color: "#f43f5e" },
            grid: { drawOnChartArea: false },
            ticks: { color: "#f43f5e" }
          }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  }
}

function renderTrajectoryChart(steps, powers) {
  const canvas = document.getElementById("chart-trajectory");
  if (!canvas) return;

  if (window.Chart) {
    if (state.charts.trajectory) state.charts.trajectory.destroy();
    const ctx = canvas.getContext("2d");
    state.charts.trajectory = new Chart(ctx, {
      type: "line",
      data: {
        labels: steps,
        datasets: [
          {
            label: "Predicted Horizon Power (W)",
            data: powers,
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.2)",
            borderWidth: 3,
            pointBackgroundColor: "#38bdf8",
            pointRadius: 5,
            fill: true,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
          y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
        },
        plugins: {
          legend: { labels: { color: "#f1f5f9" } }
        }
      }
    });
  }
}

// Fallback HTML5 Canvas drawer for standalone execution without internet
function drawFallbackLineChart(canvas, labels, datasets) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight || 300;
  
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1e293b";
  ctx.font = "11px sans-serif";

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = 30 + (h - 60) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();
  }

  // Draw lines
  datasets.forEach(ds => {
    if (!ds.data || !ds.data.length) return;
    const minVal = Math.min(...ds.data);
    const maxVal = Math.max(...ds.data) || 1;
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    ds.data.forEach((val, idx) => {
      const x = 40 + (w - 60) * (idx / (ds.data.length - 1));
      const y = h - 30 - (h - 60) * ((val - minVal) / (maxVal - minVal || 1));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

// ============================================================
// DATA TABLES RENDERING
// ============================================================
function renderPVTable(models) {
  const tbody = document.getElementById("table-pv-models-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  models.forEach(m => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.tag}</strong></td>
      <td>${parseFloat(m.val_mse_scaled).toFixed(4)}</td>
      <td>${parseFloat(m.val_f1_at_A).toFixed(3)}</td>
      <td><span class="badge badge-blue">${m.A_threshold} W/min</span></td>
      <td>${parseFloat(m.test_mse_scaled).toFixed(4)}</td>
      <td>${parseFloat(m.test_mae_w).toFixed(2)} W</td>
      <td><strong style="color:var(--accent-emerald)">${parseFloat(m.test_f1_at_A).toFixed(3)}</strong></td>
      <td>${parseFloat(m.test_AP).toFixed(3)}</td>
      <td>${parseFloat(m.test_day_AUC).toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderWindTable(models) {
  const tbody = document.getElementById("table-wind-models-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  models.forEach(m => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.featset || m.model}</strong></td>
      <td>${m["config(lag,hidden)"] || "N/A"}</td>
      <td>${m.MAE || "-"}</td>
      <td>${m.RMSE || "-"}</td>
      <td>${m.MBE || "-"}</td>
      <td><strong style="color:var(--accent-emerald)">${m.F1 || "-"}</strong></td>
      <td>${m.Recall || "-"}</td>
      <td>${m.Precision || "-"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderWindDirectionTable(directions) {
  const tbody = document.getElementById("table-wind-direction-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  directions.forEach(d => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${d.model}</strong></td>
      <td>${d.F1 || "-"}</td>
      <td><strong style="color:${parseFloat(d.F1_up) === 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">${d.F1_up || "0.0"}</strong></td>
      <td>${d.F1_down || "-"}</td>
      <td><span class="badge badge-purple">${d.DirAcc || "-"}%</span></td>
      <td>${d.RMSE || "-"}</td>
      <td>${d.MBE_up || "-"}</td>
      <td>${d.MBE_down || "-"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationTable(ablation) {
  const tbody = document.getElementById("table-ablation-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  ablation.forEach(a => {
    const tr = document.createElement("tr");
    const isSelected = a.model.includes("selected");
    tr.innerHTML = `
      <td>${isSelected ? '🌟 <strong>' + a.model + '</strong>' : a.model}</td>
      <td>${a.RMSE}</td>
      <td><strong>${a.F1}</strong></td>
      <td><strong style="color:var(--accent-emerald)">${a.F1_signed}</strong></td>
      <td>${a.F1_up}</td>
      <td>${a.F1_down}</td>
      <td><span class="badge ${isSelected ? 'badge-green' : 'badge-blue'}">${a.DirAcc}%</span></td>
      <td>${a.sec}s</td>
    `;
    tbody.appendChild(tr);
  });
}
