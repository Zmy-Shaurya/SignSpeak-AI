"use strict";
const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replace(/_/g, " ");

const STATUS_TEXT = {
  live: "Live", starting: "Starting", error: "Camera error", no_signal: "No picture", connecting: "Connecting", offline: "Server offline",
};

function setStatus(state, text, fps) {
  $("status").dataset.state = state;
  $("status-text").textContent = text || STATUS_TEXT[state];
  $("fps").textContent = fps ? `${fps} fps` : "";
}

async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

/* ---------- cameras ---------- */
let cameras = [];
let activeCamera = null;

function renderCameras() {
  const select = $("camera");
  select.replaceChildren(...cameras.map((c) => {
    const o = document.createElement("option");
    o.value = c.index;
    o.textContent = `Camera ${c.index}${c.available ? "" : " (not found)"}`;  // never disabled: let the user try anyway
    o.selected = c.index === activeCamera;
    return o;
  }));
}

async function scan() {
  $("rescan").disabled = true;
  $("camera-msg").textContent = "Scanning...";
  try {
    cameras = await (await fetch("/api/cameras")).json();
    $("camera-msg").textContent = "";
    renderCameras();
  } catch {
    $("camera-msg").textContent = "Scan failed";
  } finally {
    $("rescan").disabled = false;
  }
}

$("camera").addEventListener("change", (e) => post("/api/camera", { index: Number(e.target.value) }));
$("rescan").addEventListener("click", scan);

/* ---------- settings ---------- */
const SLIDERS = { threshold: (v) => v.toFixed(2), hold: (v) => v, pause: (v) => `${v.toFixed(1)}` };
let editing = null;  // slider being dragged, so incoming state doesn't fight the user

for (const [name, fmt] of Object.entries(SLIDERS)) {
  const input = $(name);
  input.addEventListener("input", () => {
    $(`${name}-out`).textContent = fmt(Number(input.value));
    editing = name;
  });
  input.addEventListener("change", async () => {
    await post("/api/settings", { [name]: Number(input.value) });
    editing = null;
  });
}
$("mute").addEventListener("change", (e) => post("/api/settings", { mute: e.target.checked }));
$("clear").addEventListener("click", () => post("/api/clear"));

/* ---------- live state ---------- */
let probRows = {};

function renderProbs(probs, top) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  if (!entries.length) { $("probs").replaceChildren(); probRows = {}; return; }
  if (Object.keys(probRows).length !== entries.length) {
    probRows = {};
    $("probs").replaceChildren(...entries.map(([name]) => {
      const row = document.createElement("div");
      row.className = "prob";
      row.innerHTML = '<span class="name"></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
      row.querySelector(".name").textContent = pretty(name);
      probRows[name] = row;
      return row;
    }));
  }
  entries.forEach(([name, p], i) => {
    const row = probRows[name];
    row.style.order = i;
    row.classList.toggle("top", i === 0);
    row.querySelector(".fill").style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector(".pct").textContent = `${Math.round(p * 100)}%`;
  });
}

function render(s) {
  activeCamera = s.camera.index;
  setStatus(s.camera.status, STATUS_TEXT[s.camera.status], s.camera.status === "live" ? s.camera.fps : null);
  $("camera-msg").textContent = s.camera.status === "live" ? "" : s.camera.message;
  if (cameras.length && $("camera").value !== String(activeCamera)) renderCameras();

  const sign = $("sign");
  sign.textContent = s.label ? pretty(s.label) : s.warming ? "Warming up" : "–";
  sign.classList.toggle("idle", !s.label);
  $("sign-bar").style.width = `${s.confidence * 100}%`;
  $("sign-hint").textContent = s.warming ? "Filling the analysis window" : `Confidence ${Math.round(s.confidence * 100)}%`;
  renderProbs(s.probs);

  $("words").replaceChildren(...(s.words.length ? s.words.map((w) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = pretty(w);
    return chip;
  }) : [Object.assign(document.createElement("span"), { className: "empty", textContent: "No words yet" })]));
  $("sentence").textContent = s.sentence;

  for (const name of Object.keys(SLIDERS)) {
    if (editing === name) continue;
    $(name).value = s.settings[name];
    $(`${name}-out`).textContent = SLIDERS[name](s.settings[name]);
  }
  if (document.activeElement !== $("mute")) $("mute").checked = s.settings.mute;
}

function connect() {
  const events = new EventSource("/api/events");
  events.onmessage = (e) => render(JSON.parse(e.data));
  events.onerror = () => setStatus("offline");  // EventSource reconnects by itself
}

connect();
scan();
