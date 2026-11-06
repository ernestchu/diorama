'use strict';
// Stop-line labeller. Frames are openpilot medmodel frames rendered at IMAGE_SCALE x; labels are stored in
// medmodel pixels (512x256): a frame has at most one stop line, in the ego lane, two end points [x1, y1, x2, y2].
// A frame may come with a suggested label (prefill.json), which counts once confirmed or edited here.

const STORE = 'stopline-session-v2';
const HANDLE_MOUSE = 10, HANDLE_TOUCH = 24; // css px to grab an end point
const TAP = 6; // css px a press may move and still be a tap
const MAX_ZOOM = 16;
// a border between the picture and the screen edge, wider on touch screens, so an end point at the
// picture's edge can be reached without touching the screen's edge (drags past the picture snap to it)
const margin = () => (matchMedia('(pointer: coarse)').matches ? 32 : 12);

const $ = (id) => document.getElementById(id);
const canvas = $('view'), ctx = canvas.getContext('2d');
const loupe = $('loupe'), lctx = loupe.getContext('2d');

let manifest = null; // {version, frame_size, image_scale, frames: [id]}
let session = null; // {started, labels: {id: {status, lines, t}}}
let prefill = {}; // {id: {status, lines, why?, lead?, lanes?, orig?}}: suggested labels and what to check them against
let overlays = true; // draw the lead box, lane boundaries and the original line
let index = 0;
let lines = []; // the current frame's lines, in image pixels
let status = null; // the current frame's explicit status: 'none' | null
let undo = [];
let confirmed = false; // the current frame already had a label from this session when opened
let selected = -1;
let img = null;
const cache = new Map();
let view = { k: 1, x: 0, y: 0 }; // screen = image * k + (x, y), in css px
let fit = 1;
let spaceDown = false;

// ---------- storage ----------
function loadSession() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE));
    if (s && s.labels) return s;
  } catch (e) { /* storage blocked or bad: start fresh */ }
  return { started: new Date().toISOString(), labels: {} };
}
function saveSession() {
  try { localStorage.setItem(STORE, JSON.stringify(session)); } catch (e) { /* in memory only */ }
}

// ---------- frames ----------
const scale = () => manifest.image_scale;
const fid = () => manifest.frames[index];
function image(id) {
  if (!cache.has(id)) {
    const im = new Image();
    im.decoding = 'async';
    im.src = `frames/${id}.jpg`;
    cache.set(id, im);
  }
  return cache.get(id);
}
function preload() {
  for (let d = 1; d <= 4; d++) if (index + d < manifest.frames.length) image(manifest.frames[index + d]);
  if (cache.size > 40) for (const k of [...cache.keys()].slice(0, cache.size - 40)) cache.delete(k);
}

function commit() {
  // store the current frame's label from `lines` and `status`
  if (!manifest.frames.length) return;
  const id = fid();
  const s = scale();
  const t = new Date().toISOString();
  if (lines.length) {
    session.labels[id] = { status: 'lines', lines: lines.map((l) => l.map((v) => Math.round((v / s) * 100) / 100)), t };
  } else if (status) {
    session.labels[id] = { status, lines: [], t };
  } else {
    delete session.labels[id];
  }
  saveSession();
}

function open(i) {
  if (!manifest.frames.length) { $('empty').hidden = false; updateUI(); return; }
  index = Math.max(0, Math.min(manifest.frames.length - 1, i));
  confirmed = !!session.labels[fid()];
  const lab = session.labels[fid()] || prefill[fid()];
  const s = scale();
  lines = lab ? lab.lines.map((l) => l.map((v) => v * s)) : [];
  status = lab && lab.status !== 'lines' ? lab.status : null;
  undo = []; selected = -1;
  img = image(fid());
  if (img.complete && img.naturalWidth) resetView(); else img.onload = () => { if (img === image(fid())) { resetView(); } };
  img.onerror = () => { $('badge').textContent = 'image failed to load'; };
  preload();
  updateUI();
}

function nextUnlabeled(from) {
  for (let i = from; i < manifest.frames.length; i++) if (!session.labels[manifest.frames[i]]) return i;
  for (let i = 0; i < from; i++) if (!session.labels[manifest.frames[i]]) return i;
  return Math.min(from, manifest.frames.length - 1);
}

// stepping past a frame only keeps what was changed there: a suggested label is confirmed with Next or No line
function go(delta) { if (undo.length) commit(); open(index + delta); }
function done(st) {
  if (st) { if (lines.length) { pushUndo(); lines = []; } status = st; }
  commit();
  if (index + 1 < manifest.frames.length) open(nextUnlabeled(index + 1)); else open(index);
}

// ---------- view ----------
function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  if (img && img.naturalWidth) clampView();
  draw();
}
function fitScale(r) {
  const m = margin();
  return Math.min((r.width - 2 * m) / img.naturalWidth, (r.height - 2 * m) / img.naturalHeight);
}
function resetView() {
  const r = canvas.getBoundingClientRect();
  fit = fitScale(r);
  view.k = fit;
  view.x = (r.width - img.naturalWidth * fit) / 2;
  view.y = (r.height - img.naturalHeight * fit) / 2;
  draw();
}
function clampView() {
  const r = canvas.getBoundingClientRect();
  fit = fitScale(r);
  view.k = Math.max(fit, Math.min(fit * MAX_ZOOM, view.k));
  const m = margin(), w = img.naturalWidth * view.k, h = img.naturalHeight * view.k;
  view.x = w <= r.width - 2 * m ? (r.width - w) / 2 : Math.min(m, Math.max(r.width - m - w, view.x));
  view.y = h <= r.height - 2 * m ? (r.height - h) / 2 : Math.min(m, Math.max(r.height - m - h, view.y));
}
function zoomAt(sx, sy, factor) {
  const ix = (sx - view.x) / view.k, iy = (sy - view.y) / view.k;
  view.k *= factor;
  clampView();
  view.x = sx - ix * view.k; view.y = sy - iy * view.k;
  clampView();
  draw();
}
const toScreen = (x, y) => [x * view.k + view.x, y * view.k + view.y];
const toImage = (sx, sy) => [(sx - view.x) / view.k, (sy - view.y) / view.k];

function draw() {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!img || !img.naturalWidth) return;
  ctx.setTransform(dpr * view.k, 0, 0, dpr * view.k, dpr * view.x, dpr * view.y);
  ctx.imageSmoothingEnabled = view.k < 2.5 * fit * 1.0 ? true : false;
  ctx.drawImage(img, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (overlays) drawOverlays();
  lines.forEach((l, i) => drawLine(ctx, l, i === selected, toScreen));
}
function drawOverlays() {
  const p = prefill[fid()];
  if (!p) return;
  const s = scale(), map = (x, y) => toScreen(x * s, y * s);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = '#3ddc84';
  for (const curve of p.lanes || []) {
    ctx.beginPath();
    curve.forEach(([x, y], i) => { const [a, b] = map(x, y); if (i) ctx.lineTo(a, b); else ctx.moveTo(a, b); });
    ctx.stroke();
  }
  if (p.orig) {
    const [ax, ay] = map(p.orig[0], p.orig[1]), [bx, by] = map(p.orig[2], p.orig[3]);
    ctx.strokeStyle = '#ff6b6b'; ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (p.lead) {
    const [ax, ay] = map(p.lead[0], p.lead[1]), [bx, by] = map(p.lead[2], p.lead[3]);
    ctx.strokeStyle = '#ff9f1c'; ctx.lineWidth = 2;
    ctx.strokeRect(ax, ay, bx - ax, by - ay);
  }
}
function drawLine(c, l, sel, map) {
  const [ax, ay] = map(l[0], l[1]), [bx, by] = map(l[2], l[3]);
  c.lineCap = 'round';
  c.strokeStyle = '#000a'; c.lineWidth = 4;
  c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
  c.strokeStyle = sel ? '#ff4fd8' : '#ffd400'; c.lineWidth = 2;
  c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
  for (const [x, y] of [[ax, ay], [bx, by]]) {
    c.beginPath(); c.arc(x, y, sel ? 6 : 4, 0, Math.PI * 2);
    c.fillStyle = sel ? '#ff4fd8' : '#ffd400'; c.fill();
    c.lineWidth = 1.5; c.strokeStyle = '#000'; c.stroke();
  }
}

// the loupe: 4x of the current view around an image point, placed away from the finger
function showLoupe(ix, iy, sx) {
  const r = canvas.getBoundingClientRect();
  loupe.style.display = 'block';
  loupe.style.left = sx < r.width / 2 ? `${r.width - 168}px` : '8px';
  const dpr = window.devicePixelRatio || 1;
  loupe.width = loupe.height = 160 * dpr;
  const z = Math.max(view.k * 4, 2); // screen px per image px in the loupe
  lctx.setTransform(1, 0, 0, 1, 0, 0);
  lctx.fillStyle = '#000'; lctx.fillRect(0, 0, loupe.width, loupe.height);
  lctx.imageSmoothingEnabled = false;
  lctx.setTransform(dpr * z, 0, 0, dpr * z, dpr * (80 - ix * z), dpr * (80 - iy * z));
  lctx.drawImage(img, 0, 0);
  lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const map = (x, y) => [80 + (x - ix) * z, 80 + (y - iy) * z];
  lines.forEach((l, i) => drawLine(lctx, l, i === selected, map));
  lctx.strokeStyle = '#fff'; lctx.lineWidth = 1;
  lctx.beginPath(); lctx.moveTo(80, 64); lctx.lineTo(80, 96); lctx.moveTo(64, 80); lctx.lineTo(96, 80); lctx.stroke();
}
function hideLoupe() { loupe.style.display = 'none'; }

// ---------- editing ----------
function pushUndo() { undo.push(JSON.stringify({ lines, status })); if (undo.length > 100) undo.shift(); }
function doUndo() {
  const u = undo.pop();
  if (!u) return;
  ({ lines, status } = JSON.parse(u));
  selected = -1;
  if (!undo.length && !confirmed) { delete session.labels[fid()]; saveSession(); } else commit(); // back to how it arrived
  updateUI(); draw();
}
function deleteSelected() {
  if (selected < 0) return;
  pushUndo(); lines.splice(selected, 1); selected = -1;
  commit(); updateUI(); draw();
}
function hitHandle(sx, sy, touch) {
  const rad = touch ? HANDLE_TOUCH : HANDLE_MOUSE;
  let best = null, bd = rad;
  lines.forEach((l, i) => {
    for (const e of [0, 1]) {
      const [x, y] = toScreen(l[2 * e], l[2 * e + 1]);
      const d = Math.hypot(x - sx, y - sy);
      if (d <= bd) { bd = d; best = { line: i, end: e }; }
    }
  });
  return best;
}
function hitLine(sx, sy, touch) {
  const rad = touch ? 18 : 8;
  let best = -1, bd = rad;
  lines.forEach((l, i) => {
    const [ax, ay] = toScreen(l[0], l[1]), [bx, by] = toScreen(l[2], l[3]);
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    const t = L ? Math.max(0, Math.min(1, ((sx - ax) * dx + (sy - ay) * dy) / L)) : 0;
    const d = Math.hypot(ax + t * dx - sx, ay + t * dy - sy);
    if (d <= bd) { bd = d; best = i; }
  });
  return best;
}
const clampImg = (x, y) => [Math.max(0, Math.min(img.naturalWidth, x)), Math.max(0, Math.min(img.naturalHeight, y))];

// ---------- input ----------
const pointers = new Map(); // pointerId -> {x, y}
let gesture = null; // {kind: 'draw'|'end'|'pan'|'pinch', ...}
let lastTap = 0;

function local(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  if (!img || !img.naturalWidth) return;
  canvas.setPointerCapture(e.pointerId);
  const [sx, sy] = local(e);
  pointers.set(e.pointerId, { x: sx, y: sy });
  const touch = e.pointerType !== 'mouse';

  if (pointers.size === 2) { // second finger: abandon any line in progress, pinch/pan
    if (gesture && gesture.kind === 'draw' && gesture.added) ({ lines, status } = JSON.parse(undo.pop()));
    if (gesture && gesture.kind === 'end' && gesture.before) { lines = JSON.parse(gesture.before); undo.pop(); }
    hideLoupe();
    const [p, q] = [...pointers.values()];
    gesture = { kind: 'pinch', d: Math.hypot(p.x - q.x, p.y - q.y), mx: (p.x + q.x) / 2, my: (p.y + q.y) / 2 };
    draw();
    return;
  }
  if (pointers.size > 2) return;

  if (e.button === 2 || e.button === 1 || (e.button === 0 && spaceDown)) {
    gesture = { kind: 'pan', sx, sy, x: view.x, y: view.y };
    canvas.style.cursor = 'grabbing';
    return;
  }
  if (e.button !== 0) return;
  const h = hitHandle(sx, sy, touch);
  if (h) {
    gesture = { kind: 'end', ...h, before: JSON.stringify(lines), sx, sy, moved: false };
    selected = h.line;
    updateUI(); draw();
    return;
  }
  const [ix, iy] = clampImg(...toImage(sx, sy));
  gesture = { kind: 'draw', sx, sy, ix, iy, added: false, touch };
});

canvas.addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId)) {
    if (e.pointerType === 'mouse' && img) canvas.style.cursor = spaceDown ? 'grab' : hitHandle(...local(e), false) ? 'move' : 'crosshair';
    return;
  }
  const [sx, sy] = local(e);
  pointers.set(e.pointerId, { x: sx, y: sy });
  if (!gesture) return;

  if (gesture.kind === 'pinch' && pointers.size >= 2) {
    const [p, q] = [...pointers.values()];
    const d = Math.hypot(p.x - q.x, p.y - q.y), mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
    view.x += mx - gesture.mx; view.y += my - gesture.my;
    zoomAt(mx, my, gesture.d ? d / gesture.d : 1);
    Object.assign(gesture, { d, mx, my });
  } else if (gesture.kind === 'pan') {
    view.x = gesture.x + sx - gesture.sx; view.y = gesture.y + sy - gesture.sy;
    clampView(); draw();
  } else if (gesture.kind === 'draw') {
    if (!gesture.added && Math.hypot(sx - gesture.sx, sy - gesture.sy) < TAP) return;
    const [bx, by] = clampImg(...toImage(sx, sy));
    if (!gesture.added) {
      pushUndo();
      lines = [[gesture.ix, gesture.iy, bx, by]]; // one stop line in a frame: a new one replaces it
      gesture.added = true; status = null;
      selected = 0;
    } else {
      lines[0] = [gesture.ix, gesture.iy, bx, by];
    }
    showLoupe(bx, by, sx); draw();
  } else if (gesture.kind === 'end') {
    if (!gesture.moved) { if (Math.hypot(sx - gesture.sx, sy - gesture.sy) < 2) return; pushUndo(); gesture.moved = true; }
    const [x, y] = clampImg(...toImage(sx, sy));
    const l = lines[gesture.line];
    l[2 * gesture.end] = x; l[2 * gesture.end + 1] = y;
    showLoupe(x, y, sx); draw();
  }
});

function endPointer(e) {
  if (!pointers.has(e.pointerId)) return;
  const [sx, sy] = local(e);
  pointers.delete(e.pointerId);
  if (!gesture) return;
  if (gesture.kind === 'pinch') {
    if (pointers.size === 0) gesture = null;
    else { const [p] = [...pointers.values()]; gesture = { kind: 'pan', sx: p.x, sy: p.y, x: view.x, y: view.y }; }
    return;
  }
  hideLoupe();
  if (gesture.kind === 'draw') {
    if (gesture.added) {
      const l = lines[0];
      if (Math.hypot((l[2] - l[0]) * view.k, (l[3] - l[1]) * view.k) < TAP) { ({ lines, status } = JSON.parse(undo.pop())); selected = -1; }
      commit();
    } else { // a tap: select a line, or double-tap to reset the view
      const now = performance.now();
      if (now - lastTap < 300) { resetView(); lastTap = 0; } else lastTap = now;
      selected = hitLine(sx, sy, gesture.touch);
    }
  } else if (gesture.kind === 'end') {
    if (gesture.moved) commit();
  }
  canvas.style.cursor = 'crosshair';
  gesture = null;
  updateUI(); draw();
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('dblclick', (e) => { e.preventDefault(); if (img) resetView(); });

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!img || !img.naturalWidth) return;
  const [sx, sy] = local(e);
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  zoomAt(sx, sy, Math.exp(-e.deltaY * unit * (e.ctrlKey ? 0.01 : 0.0015)));
}, { passive: false });
// Safari's trackpad pinch
canvas.addEventListener('gesturestart', (e) => e.preventDefault());
canvas.addEventListener('gesturechange', (e) => e.preventDefault());

document.addEventListener('keydown', (e) => {
  if ($('help').open) return;
  const mod = e.metaKey || e.ctrlKey;
  if (e.key === ' ') { spaceDown = true; canvas.style.cursor = 'grab'; e.preventDefault(); return; }
  if (mod && e.key.toLowerCase() === 'z') { doUndo(); e.preventDefault(); return; }
  if (mod) return;
  switch (e.key) {
    case 'Enter': done(null); break;
    case 'ArrowRight': go(1); break;
    case 'ArrowLeft': go(-1); break;
    case 'n': case 'N': done('none'); break;
    case 'z': case 'Z': doUndo(); break;
    case 'Delete': case 'Backspace': deleteSelected(); break;
    case '0': case 'f': case 'F': if (img) resetView(); break;
    case 'v': case 'V': overlays = !overlays; draw(); break;
    case 'Escape': selected = -1; updateUI(); draw(); break;
    default: return;
  }
  e.preventDefault();
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') { spaceDown = false; canvas.style.cursor = 'crosshair'; } });

// ---------- UI ----------
function updateUI() {
  const total = manifest ? manifest.frames.length : 0;
  const labeled = manifest ? manifest.frames.filter((id) => session.labels[id]).length : 0;
  $('progress').textContent = total ? `${labeled} / ${total} labelled · frame ${index + 1}` : 'No frames';
  $('frameId').textContent = total ? fid() : '';
  const p = total ? prefill[fid()] : null;
  $('badge').textContent = [lines.length ? 'line' : status === 'none' ? 'no line' : '',
    p && p.why && !(total && session.labels[fid()]) ? p.why : ''].filter(Boolean).join(' · ');
  $('noneBtn').classList.toggle('on', status === 'none' && !lines.length);
  $('delBtn').disabled = selected < 0;
  $('undoBtn').disabled = !undo.length;
  $('prevBtn').disabled = !total || index === 0;
  $('fwdBtn').disabled = !total || index === total - 1;
  for (const b of ['noneBtn', 'nextBtn']) $(b).disabled = !total;
  $('downloadBtn').disabled = !Object.keys(session.labels).length;
}

function download() {
  commit();
  const out = {
    format: 'stopline-labels/1',
    manifest_version: manifest.version,
    coordinates: 'openpilot medmodel frame pixels, 512x256 (x right, y down); a line is [x1, y1, x2, y2]',
    session_started: session.started,
    exported: new Date().toISOString(),
    labels: session.labels,
  };
  const text = JSON.stringify(out, null, 1);
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, '').replace('T', '-');
  const name = `stopline-labels-${stamp}.json`;
  const file = new File([text], name, { type: 'application/json' });
  const url = URL.createObjectURL(file);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

$('prevBtn').onclick = () => go(-1);
$('fwdBtn').onclick = () => go(1);
$('nextBtn').onclick = () => done(null);
$('noneBtn').onclick = () => done('none');
$('undoBtn').onclick = doUndo;
$('delBtn').onclick = deleteSelected;
$('downloadBtn').onclick = download;
$('helpBtn').onclick = () => {
  const n = Object.keys(session.labels).length;
  $('sessionInfo').textContent = `This session: ${n} frame${n === 1 ? '' : 's'} labelled since ${new Date(session.started).toLocaleString()}.`;
  $('help').showModal();
};
$('closeHelp').onclick = () => $('help').close();
$('clearBtn').onclick = () => {
  const n = Object.keys(session.labels).length;
  if (n && !confirm(`Delete this browser's ${n} labels? Download them first if you haven't sent them.`)) return;
  session = { started: new Date().toISOString(), labels: {} };
  saveSession(); $('help').close(); open(nextUnlabeled(0));
};
window.addEventListener('resize', resize);

(async function main() {
  session = loadSession();
  try {
    manifest = await (await fetch(`manifest.json?t=${Date.now()}`, { cache: 'no-store' })).json();
  } catch (e) {
    $('progress').textContent = 'Could not load manifest.json';
    return;
  }
  try {
    prefill = await (await fetch(`prefill.json?v=${manifest.version}`)).json();
  } catch (e) {
    $('progress').textContent = 'Could not load prefill.json';
    return;
  }
  // labels for frames no longer listed were delivered and merged: drop them
  const listed = new Set(manifest.frames);
  for (const id of Object.keys(session.labels)) if (!listed.has(id)) delete session.labels[id];
  for (const lab of Object.values(session.labels)) { // from before unclear and hard were retired
    if (lab.status === 'unclear') { lab.status = 'none'; lab.lines = []; }
    delete lab.hard;
  }
  saveSession();
  resize();
  open(nextUnlabeled(0));
  try { if (!localStorage.getItem('stopline-helped')) { localStorage.setItem('stopline-helped', '1'); $('helpBtn').click(); } } catch (e) { /* fine */ }
})();
