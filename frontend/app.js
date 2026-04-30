// app.js — HPC Model Comparison Frontend

const API = "http://127.0.0.1:8001";

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  quantMode: /** @type {"int4_base"|"int4"} */ ("int4"),
  switching: false,
  running: false,
  lastBaseline: null,
  lastOptimized: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const promptEl   = document.getElementById("prompt");
const tokenSlider = document.getElementById("max-tokens");
const tokenLabel  = document.getElementById("token-val");
const runBtn      = document.getElementById("run-btn");
const statusDot   = document.getElementById("status-dot");
const toast       = document.getElementById("toast");

// Panel refs
const sides = {
  baseline:  buildPanelRefs("left"),
  optimized: buildPanelRefs("right"),
};

function buildPanelRefs(side) {
  return {
    output:     document.getElementById(`${side}-output`),
    latency:    document.getElementById(`${side}-latency`),
    throughput: document.getElementById(`${side}-throughput`),
    memory:     document.getElementById(`${side}-memory`),
    gflops:     document.getElementById(`${side}-gflops`),
    // delta badges (only on right side)
    ...(side === "right" ? {
      deltaLatency:    document.getElementById("delta-latency"),
      deltaThroughput: document.getElementById("delta-throughput"),
      deltaMemory:     document.getElementById("delta-memory"),
      deltaGflops:     document.getElementById("delta-gflops"),
    } : {}),
  };
}

// ── Charts (Chart.js) ──────────────────────────────────────────────────────
let charts = {};

function initCharts() {
  Chart.defaults.color = "#7c8499";
  Chart.defaults.borderColor = "#2a2d3d";
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";

  const chartDefs = [
    { id: "chart-latency",    label: "Latency (ms)",       lowerBetter: true  },
    { id: "chart-throughput", label: "Throughput (tok/s)", lowerBetter: false },
    { id: "chart-memory",     label: "GPU Memory (MB)",    lowerBetter: true  },
    { id: "chart-gflops",     label: "GFLOPs/token",       lowerBetter: true  },
  ];

  chartDefs.forEach(({ id, label }) => {
    const ctx = document.getElementById(id).getContext("2d");
    charts[id] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Baseline (FP16)", "Optimized"],
        datasets: [{
          data: [null, null],
          backgroundColor: ["rgba(79,142,247,0.7)", "rgba(167,139,250,0.7)"],
          borderColor:     ["#4f8ef7", "#a78bfa"],
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: "#2a2d3d" } },
          x: { grid: { display: false } },
        },
      },
    });
  });
}

function updateCharts(baseline, optimized) {
  const mapping = {
    "chart-latency":    [baseline.latency_ms,      optimized.latency_ms     ],
    "chart-throughput": [baseline.throughput_tps,  optimized.throughput_tps ],
    "chart-memory":     [baseline.gpu_memory_mb,   optimized.gpu_memory_mb  ],
    "chart-gflops":     [baseline.gflops_per_tok,  optimized.gflops_per_tok ],
  };
  Object.entries(mapping).forEach(([id, [bVal, oVal]]) => {
    charts[id].data.datasets[0].data = [bVal ?? 0, oVal ?? 0];
    // update optimized label to show current quant mode
    charts[id].data.labels[1] = `Optimized (${state.quantMode.toUpperCase()})`;
    charts[id].update();
  });
}

// ── Status poll ────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch(`${API}/status`);
    if (r.ok) {
      const s = await r.json();
      if (s.baseline_loaded && s.optimized_loaded) {
        statusDot.classList.add("ready");
        statusDot.title = `Ready — device: ${s.device}`;
      }
    }
  } catch { /* server not up yet */ }
}

// ── Quant toggle ───────────────────────────────────────────────────────────
async function switchQuant(mode) {
  if (mode === state.quantMode || state.switching) return;

  state.switching = true;
  setQuantBtnsDisabled(true);
  showLoadingOverlay(true, `Loading ${mode.toUpperCase()} model…`);

  try {
    const r = await fetch(`${API}/switch_quant`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quant_mode: mode }),
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    state.quantMode = mode;
    updateQuantBtns();
    showToast(`Switched to ${mode.toUpperCase()}`);
  } catch (e) {
    showToast(`Switch failed: ${e.message}`, true);
  } finally {
    state.switching = false;
    setQuantBtnsDisabled(false);
    showLoadingOverlay(false);
  }
}

function updateQuantBtns() {
  document.querySelectorAll(".quant-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === state.quantMode);
  });
}

function setQuantBtnsDisabled(val) {
  document.querySelectorAll(".quant-btn").forEach(b => b.disabled = val);
}

// ── Run inference ──────────────────────────────────────────────────────────
async function runComparison() {
  const prompt = promptEl.value.trim();
  if (!prompt) { showToast("Enter a prompt first", true); return; }
  if (state.running) return;

  state.running = true;
  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="spinner"></span>Running…';

  setMetricsLoading();

  try {
    const max_new_tokens = parseInt(tokenSlider.value, 10);
    const body = JSON.stringify({ prompt, max_new_tokens });
    const headers = { "Content-Type": "application/json" };

    // Fire both requests in parallel
    const [bRes, oRes] = await Promise.all([
      fetch(`${API}/infer/baseline`,  { method: "POST", headers, body }),
      fetch(`${API}/infer/optimized`, { method: "POST", headers, body }),
    ]);

    if (!bRes.ok) throw new Error("Baseline: " + (await bRes.json()).detail);
    if (!oRes.ok) throw new Error("Optimized: " + (await oRes.json()).detail);

    const baseline  = await bRes.json();
    const optimized = await oRes.json();

    state.lastBaseline  = baseline;
    state.lastOptimized = optimized;

    renderMetrics(baseline, optimized);
    updateCharts(baseline, optimized);


  } catch (e) {
    showToast(`Error: ${e.message}`, true);
    resetMetrics();
  } finally {
    state.running = false;
    runBtn.disabled = false;
    runBtn.textContent = "Run Comparison";
  }
}

// ── Render helpers ─────────────────────────────────────────────────────────
function renderMetrics(b, o) {
  // output text
  sides.baseline.output.textContent  = (b.output_text || "(empty)").trim();
  sides.baseline.output.classList.remove("placeholder");
  sides.optimized.output.textContent = (o.output_text || "(empty)").trim();
  sides.optimized.output.classList.remove("placeholder");

  // baseline values
  sides.baseline.latency.textContent    = fmt(b.latency_ms,     1);
  sides.baseline.throughput.textContent = fmt(b.throughput_tps, 1);
  sides.baseline.memory.textContent     = fmt(b.gpu_memory_mb,  1);
  sides.baseline.gflops.textContent     = b.gflops_per_tok != null ? fmt(b.gflops_per_tok, 2) : "N/A";

  // optimized values + delta badges
  const metrics = [
    { ref: "latency",    bVal: b.latency_ms,     oVal: o.latency_ms,     unit: "ms",      lowerBetter: true  },
    { ref: "throughput", bVal: b.throughput_tps, oVal: o.throughput_tps, unit: "tok/s",   lowerBetter: false },
    { ref: "memory",     bVal: b.gpu_memory_mb,  oVal: o.gpu_memory_mb,  unit: "MB",      lowerBetter: true  },
    { ref: "gflops",     bVal: b.gflops_per_tok, oVal: o.gflops_per_tok, unit: "GFLOPs",  lowerBetter: true  },
  ];

  metrics.forEach(({ ref, bVal, oVal, lowerBetter }) => {
    const valueEl = sides.optimized[ref];
    const deltaEl = sides.optimized[`delta${capitalize(ref)}`];

    valueEl.textContent = (oVal != null) ? fmt(oVal, 1) : "N/A";


    if (bVal != null && oVal != null && deltaEl) {
      const pct = ((oVal - bVal) / bVal) * 100;
      const better = lowerBetter ? pct < 0 : pct > 0;
      const sign   = pct >= 0 ? "+" : "";
      deltaEl.textContent = `${sign}${pct.toFixed(1)}% vs baseline`;
      deltaEl.className = "metric-delta " + (better ? "delta-better" : "delta-worse");
    }
  });
}

function setMetricsLoading() {
  const dash = "…";
  ["latency","throughput","memory","gflops"].forEach(k => {
    sides.baseline.output.textContent  = "Generating…";
    sides.optimized.output.textContent = "Generating…";
    if (sides.baseline[k])  sides.baseline[k].textContent  = dash;
    if (sides.optimized[k]) sides.optimized[k].textContent = dash;
    const dKey = `delta${capitalize(k)}`;
    if (sides.optimized[dKey]) sides.optimized[dKey].textContent = "";
  });
}

function resetMetrics() {
  sides.baseline.output.textContent  = "Awaiting prompt…";
  sides.baseline.output.classList.add("placeholder");
  sides.optimized.output.textContent = "Awaiting prompt…";
  sides.optimized.output.classList.add("placeholder");
  ["latency","throughput","memory","gflops"].forEach(k => {
    if (sides.baseline[k])  sides.baseline[k].textContent  = "—";
    if (sides.optimized[k]) sides.optimized[k].textContent = "—";
  });
}

function showLoadingOverlay(show, text = "") {
  let overlay = document.getElementById("right-overlay");
  if (show) {
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "right-overlay";
      overlay.className = "loading-overlay";
      document.querySelector(".panel.right").appendChild(overlay);
    }
    overlay.textContent = text;
  } else {
    overlay?.remove();
  }
}

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, isError = false) {
  toast.textContent = msg;
  toast.style.background = isError ? "#7f1d1d" : "#1e293b";
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

// ── Utils ──────────────────────────────────────────────────────────────────
const fmt = (v, dp = 1) => (v == null ? "N/A" : Number(v).toFixed(dp));
const capitalize = s => s.charAt(0).toUpperCase() + s.slice(1);

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  updateQuantBtns();
  pollStatus();
  setInterval(pollStatus, 5000);

  // Token slider label
  tokenSlider.addEventListener("input", () => {
    tokenLabel.textContent = tokenSlider.value;
  });

  // Run button
  runBtn.addEventListener("click", runComparison);

  // Allow Ctrl+Enter to submit
  promptEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runComparison();
  });

  // Quant toggle buttons
  document.querySelectorAll(".quant-btn").forEach(btn => {
    btn.addEventListener("click", () => switchQuant(btn.dataset.mode));
  });
});
