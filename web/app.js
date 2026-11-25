"use strict";
const $ = (id) => document.getElementById(id);
const pretty = (s) => s.replace(/_/g, " ");
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const STATUS_TEXT = {
  live: "Live", starting: "Starting", error: "Camera error", no_signal: "No picture", connecting: "Connecting", offline: "Server offline",
};

function setStatus(state, fps) {
  $("status").dataset.state = state;
  $("status-text").textContent = STATUS_TEXT[state];
  $("fps").textContent = fps ? `${fps} fps` : "";
}

async function send(method, url, body) {
  const res = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!res.ok) throw new Error(`${method} ${url}: ${res.status}`);
  return res.json();
}
const post = (url, body) => send("POST", url, body);

/* ---------- theme ---------- */
$("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = (root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* private mode: theme just won't persist */ }
});

/* ---------- cameras ---------- */
let cameras = [];
let activeCamera = null;

function renderCameras() {
  $("camera").replaceChildren(...cameras.map((c) => {
    const o = el("option", "", `Camera ${c.index}${c.available ? "" : " (not found)"}`);  // never disabled: let the user try anyway
    o.value = c.index;
    o.selected = c.index === activeCamera;
    return o;
  }));
}

async function scan() {
  $("rescan").disabled = true;
  $("rescan").textContent = "Scanning...";
  try {
    cameras = await (await fetch("/api/cameras")).json();
    renderCameras();
  } catch {
    $("warmup").textContent = "Camera scan failed";
  } finally {
    $("rescan").disabled = false;
    $("rescan").textContent = "Rescan";
  }
}

function nextCamera() {
  const count = cameras.length || 5;
  post("/api/camera", { index: ((activeCamera ?? 0) + 1) % count });
}

$("camera").addEventListener("change", (e) => post("/api/camera", { index: Number(e.target.value) }));
$("rescan").addEventListener("click", scan);
$("next-camera").addEventListener("click", nextCamera);

/* ---------- settings and actions ---------- */
const SLIDERS = { threshold: (v) => v.toFixed(2), hold: (v) => v, pause: (v) => v.toFixed(1), stride: (v) => (v === 1 ? "every" : `1 in ${v}`) };
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
$("clear").addEventListener("click", () => editWords([]));
$("speak").addEventListener("click", () => post("/api/speak").catch(() => {}));  // 409 = no words: the button is disabled then anyway
$("repeat").addEventListener("click", () => post("/api/repeat").catch(() => {}));

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  const key = e.key.toLowerCase();
  if (key === "s") $("speak").click();
  else if (key === "c") editWords([]);
  else if (key === "m") post("/api/settings", { mute: !$("mute").checked });
  else if (key === "n") nextCamera();
});

/* ---------- sentence builder: edit the accepted words ---------- */
let words = [];         // what the server last told us
let shownWords = null;  // what the chips currently show (JSON), so we only rebuild them on a real change
let dragFrom = null;

let editQueue = Promise.resolve();  // edits go to the server one at a time, in order
let editsPending = 0;
let trustServerAfter = 0;          // ignore the server's word list until shortly after the last edit lands

function editWords(next, focusIndex) {
  words = next;
  renderWords(focusIndex);  // optimistic: show the edit now
  editsPending += 1;
  editQueue = editQueue
    .then(() => send("PUT", "/api/words", { words: next }))
    .catch(() => { shownWords = null; })  // rejected: redraw from the server's list
    .finally(() => {
      editsPending -= 1;
      trustServerAfter = performance.now() + 300;  // a state snapshot taken before the edit may still be in flight
    });
}

function moveWord(from, to) {
  if (to < 0 || to >= words.length || from === to) return;
  const next = [...words];
  next.splice(to, 0, next.splice(from, 1)[0]);
  editWords(next, to);
}

function chip(word, i) {
  const node = el("span", "chip", pretty(word));
  node.tabIndex = 0;
  node.draggable = true;
  node.setAttribute("aria-label", `${pretty(word)}, word ${i + 1} of ${words.length}`);
  const remove = el("button", "", "×");
  remove.type = "button";
  remove.setAttribute("aria-label", `Remove ${pretty(word)}`);
  remove.addEventListener("click", () => editWords(words.filter((_, j) => j !== i), Math.min(i, words.length - 2)));
  node.append(remove);

  node.addEventListener("keydown", (e) => {
    if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); remove.click(); }
    else if (e.shiftKey && e.key === "ArrowLeft") { e.preventDefault(); moveWord(i, i - 1); }
    else if (e.shiftKey && e.key === "ArrowRight") { e.preventDefault(); moveWord(i, i + 1); }
  });
  node.addEventListener("dragstart", (e) => { dragFrom = i; node.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; });
  node.addEventListener("dragend", () => { dragFrom = null; node.classList.remove("dragging"); });
  node.addEventListener("dragover", (e) => { if (dragFrom !== null) { e.preventDefault(); node.classList.add("drop-target"); } });
  node.addEventListener("dragleave", () => node.classList.remove("drop-target"));
  node.addEventListener("drop", (e) => { e.preventDefault(); node.classList.remove("drop-target"); moveWord(dragFrom, i); });
  return node;
}

function renderWords(focusIndex) {
  const key = JSON.stringify(words);
  if (key === shownWords && focusIndex === undefined) return;  // unchanged: keep the DOM (and focus, drag) intact
  shownWords = key;
  $("words").replaceChildren(...(words.length ? words.map(chip) : [el("span", "empty-words", "Sign to add words, or pick them from the vocabulary")]));
  if (focusIndex !== undefined && focusIndex >= 0) $("words").children[focusIndex]?.focus();
  $("speak").disabled = !words.length;
}

/* ---------- live state ---------- */
const tiles = {};  // sign name -> tile element, built once so the order never jumps
let history = [];
let lastSentence = "";

function buildVocab(names) {
  const sorted = [...names].sort();
  $("vocab").replaceChildren(...sorted.map((name) => {
    const tile = el("button", "sign-tile");
    tile.type = "button";
    tile.append(el("span", "", pretty(name)), el("span", "pct"));
    if (name === "nothing") tile.disabled = true;
    else {
      tile.title = `Add "${pretty(name)}" to the words`;
      tile.addEventListener("click", () => editWords([...words, name]));
    }
    tiles[name] = tile;
    return tile;
  }));
}

function renderVocab(s) {
  const names = s.labels;  // always present, so words can be added by hand even without a camera picture
  if (Object.keys(tiles).length !== names.length) buildVocab(names);
  const hasProbs = Object.keys(s.probs).length > 0;
  const topName = hasProbs ? names.reduce((a, b) => (s.probs[a] >= s.probs[b] ? a : b)) : null;
  for (const name of names) {
    const p = s.probs[name] || 0;
    const tile = tiles[name];
    tile.style.setProperty("--p", `${(p * 100).toFixed(0)}%`);
    tile.lastChild.textContent = p >= 0.01 ? `${Math.round(p * 100)}%` : "";
    tile.classList.toggle("top", name === topName);
    tile.classList.toggle("active", name === s.label);
    tile.classList.toggle("neutral", name === "nothing");
  }
  $("vocab-hint").textContent = `${names.length} classes`;
}

function renderHistory() {
  $("history").replaceChildren(...history.slice(0, 5).map((h) => {
    const li = el("li");
    li.append(el("time", "", h.time), el("span", "", h.text));
    return li;
  }));
}

function renderTranscript(s) {
  if (dragFrom === null && editsPending === 0 && performance.now() > trustServerAfter) {  // not mid-drag or mid-edit
    words = s.words;
    renderWords();
  }

  const sentence = $("sentence");
  const busy = s.sentence === "...";
  sentence.textContent = busy ? "Composing sentence" : s.sentence || "Your sentence appears here when you pause";
  sentence.classList.toggle("empty", !s.sentence);
  sentence.classList.toggle("busy", busy);
  $("repeat").disabled = !s.sentence || busy;
  if (s.sentence && !busy && s.sentence !== lastSentence) {
    history.unshift({ text: s.sentence, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    renderHistory();
  }
  if (!busy) lastSentence = s.sentence;

  const pausing = s.words.length && !s.label && !s.warming;  // the engine is counting a pause
  $("pause-fill").style.width = pausing ? `${Math.min(100, (s.idle / s.settings.pause) * 100)}%` : "0%";
}

function renderCamera(s) {
  const cam = s.camera;
  activeCamera = cam.index;
  setStatus(cam.status, cam.status === "live" ? cam.fps : null);
  if (cameras.length && $("camera").value !== String(activeCamera)) renderCameras();

  const problem = cam.status === "error" || cam.status === "no_signal";
  $("notice").hidden = cam.status === "live";
  $("notice-title").textContent = problem ? (cam.status === "error" ? "Camera unavailable" : "No picture from this camera") : "Opening camera";
  $("notice-text").textContent = problem ? cam.message : `Connecting to camera ${cam.index}...`;
  $("next-camera").hidden = !problem;

  $("warmup").textContent = cam.status !== "live" ? STATUS_TEXT[cam.status]
    : s.warming ? "Filling the analysis window" : s.label ? "Sign detected" : "Ready, start signing";
  $("m-stride").textContent = s.settings.stride === 1 ? "Every frame" : `1 in ${s.settings.stride} frames`;
  $("m-model").textContent = s.model.version;
}

async function pollMetrics() {
  try {
    const m = await (await fetch("/api/metrics")).json();
    $("m-latency").textContent = m.inference_ms.samples ? `${m.inference_ms.p50} ms` : "-";
  } catch { /* server restarting: the status pill already says offline */ }
}
setInterval(pollMetrics, 2000);

let shouldersMissing = 0;  // consecutive states without shoulders; debounced so one bad frame doesn't flash a warning

function renderTracking(s) {
  const q = s.camera.status === "live" ? s.quality : null;
  $("tracking").hidden = !q;
  shouldersMissing = q && !q.body ? shouldersMissing + 1 : 0;
  $("framing").hidden = shouldersMissing < 10;  // about 1 s at 10 states per second
  if (!q) return;
  for (const li of $("tracking").children) li.classList.toggle("on", q[li.dataset.part]);
}

function renderCaption(s) {
  const show = Boolean(s.label);
  $("caption").hidden = !show;
  if (!show) return;
  $("caption-sign").textContent = pretty(s.label);
  $("caption-conf").textContent = `${Math.round(s.confidence * 100)}%`;
  $("hold-fill").style.width = `${Math.min(100, (s.streak / s.settings.hold) * 100)}%`;
}

function render(s) {
  renderCamera(s);
  renderTracking(s);
  renderCaption(s);
  renderTranscript(s);
  renderVocab(s);
  for (const name of Object.keys(SLIDERS)) {
    if (editing === name) continue;
    $(name).value = s.settings[name];
    $(`${name}-out`).textContent = SLIDERS[name](s.settings[name]);
  }
  if (document.activeElement !== $("mute")) $("mute").checked = s.settings.mute;
  $("model-info").textContent = `Model ${s.model.version}, ${s.model.classes} classes, trained ${s.model.trained.slice(0, 10)}`;
}

function connect() {
  const events = new EventSource("/api/events");
  events.onmessage = (e) => render(JSON.parse(e.data));
  events.onerror = () => setStatus("offline");  // EventSource reconnects by itself
}

connect();
scan();
