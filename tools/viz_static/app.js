// Climb Pipeline Viewer – vanilla-JS SPA.
// Loads /api/{summary,holds,pose,contacts,events,route} on boot, renders three
// tabs (Hold-level, Route-level, Contacts Gantt). Keeps state on window.VIZ
// for easy debugging from the devtools console.

(() => {

// Climbing-pigment palette tuned for the paper substrate.
const COCO_COLOR = {
  Orange: "#d83a1c", Green: "#4f7a4a", Blue: "#28637a", Red: "#a72710",
  Yellow: "#c89a2a", Pink: "#b14b65", Purple: "#5e4f7a", Black: "#1a1814",
  White: "#ebe6da", Gray: "#6e695d", UNKNOWN: "#9a9587",
};
// Earthy hold-state pigments — matches `--bronze`, `--moss`, `--slate` in CSS.
const STATE_COLOR = {
  core: "#b78343", possible: "#6b7a4a", rejected: "#5d5852", unknown: "#9a9587",
};
const LIMB_ORDER = ["left_hand", "right_hand", "left_foot", "right_foot"];
const LIMB_HEX = {
  left_hand: "#c84a2a",
  right_hand: "#d6a73c",
  left_foot:  "#4f7a4a",
  right_foot: "#28637a",
};
const SKELETON_COLOR = "#f3efe6";   // paper for dark video
const SKELETON_STROKE_W = 2.6;

// COCO-WholeBody + Goliath mapping → pairs we actually want to draw. Body +
// feet; hands shown as wrist dots only to avoid clutter.
const SKELETON_EDGES = [
  ["left_shoulder","right_shoulder"],
  ["left_shoulder","left_elbow"], ["left_elbow","left_wrist"],
  ["right_shoulder","right_elbow"], ["right_elbow","right_wrist"],
  ["left_shoulder","left_hip"], ["right_shoulder","right_hip"],
  ["left_hip","right_hip"],
  ["left_hip","left_knee"], ["left_knee","left_ankle"],
  ["right_hip","right_knee"], ["right_knee","right_ankle"],
  ["left_ankle","left_big_toe"], ["left_ankle","left_heel"],
  ["right_ankle","right_big_toe"], ["right_ankle","right_heel"],
];
const KP_DOTS = [
  "nose","left_eye","right_eye",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist",
  "left_hip","right_hip","left_knee","right_knee",
  "left_ankle","right_ankle","left_heel","right_heel",
  "left_big_toe","right_big_toe",
];
// Hand pivot points the pipeline actually treats as "contact fingers"
const HAND_TIP_NAMES = {
  left_hand:  ["left_wrist","left_hand_5","left_hand_9"],
  right_hand: ["right_wrist","right_hand_5","right_hand_9"],
};

const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));

const V = window.VIZ = {
  attempts: [], attemptId: null,
  summary: null, holds: [], pose: [], contacts: {}, events: {}, route: {},
  holdsById: new Map(),
  poseByFrame: new Map(),
  filterState: new Set(["all"]),
  filterColor: new Set(["all"]),
  activeKf: null,
  // ---- Hold-level photo-set (independent from video attempts) ----
  holdsMode: "keyframes",   // "keyframes" or "photos"
  photos: [],
  photoCache: new Map(),    // pid -> {holds, width, height}
  activePhoto: null,        // pid
};

function api(path) {
  return `/api/${V.attemptId}${path.startsWith("/") ? path : "/" + path}`;
}

// ---------- bootstrap ----------
(async function init() {
  try {
    const [attempts, photos] = await Promise.all([
      fetch("/api/attempts").then(r => r.json()),
      fetch("/api/photos").then(r => r.ok ? r.json() : []).catch(() => []),
    ]);
    V.attempts = attempts;
    V.photos = photos || [];
    V.holdsMode = (V.photos.length > 0) ? "photos" : "keyframes";
    if (!attempts.length) throw new Error("no attempts registered on server");
    V.attemptId = attempts[0].id;

    setupAttemptPicker(attempts);
    bindTabs();
    bindVideoPanel();       // attaches event listeners once
    bindHoldFilters();      // ditto — handlers read V state dynamically

    await loadAttempt(V.attemptId);
    if (V.holdsMode === "photos") {
      // Hold-level draws from the photo-set, independent of attempt selection.
      renderPhotoList();
      const first = V.photos[0]?.id;
      if (first) await selectPhoto(first);
    }
  } catch (err) {
    console.error(err);
    document.body.innerHTML =
      `<pre style="color:#ea5a5a; padding:20px">viewer bootstrap failed: ${err.message}</pre>`;
  }
})();

function setupAttemptPicker(attempts) {
  const picker = $("#attemptPicker");
  const sel = $("#attemptSelect");
  sel.innerHTML = "";
  attempts.forEach(a => {
    const opt = document.createElement("option");
    opt.value = a.id;
    const hbs = a.holds_by_state || {};
    const suffix = ` — ${a.holds_total} holds · ${a.target_color ?? "?"}`;
    opt.textContent = `${a.label}${suffix}`;
    sel.appendChild(opt);
  });
  picker.hidden = attempts.length < 2;
  sel.addEventListener("change", async () => {
    V.attemptId = sel.value;
    await loadAttempt(V.attemptId);
  });
}

async function loadAttempt(aid) {
  V.attemptId = aid;
  // Drop state that must be rebuilt:
  V.holdsById = new Map();
  V.poseByFrame = new Map();
  V.filterState = new Set(["all"]);
  V.filterColor = new Set(["all"]);
  V.activeKf = null;

  const [summary, holds, pose, contacts, events, route] = await Promise.all([
    fetch(api("/summary")).then(r => r.json()),
    fetch(api("/holds")).then(r => r.json()),
    fetch(api("/pose")).then(r => r.json()),
    fetch(api("/contacts")).then(r => r.json()),
    fetch(api("/events")).then(r => r.json()),
    fetch(api("/route")).then(r => r.json()),
  ]);
  V.summary = summary;
  V.holds = holds;
  V.pose = pose;
  V.contacts = contacts || {};
  V.events = events || {};
  V.route = route;
  holds.forEach(h => V.holdsById.set(h.id, h));
  pose.forEach(p => V.poseByFrame.set(p.frame, p));

  renderHeader();
  resetVideoSource();
  renderGantt(true);
  renderEventsPanel();

  if (V.holdsMode === "photos") {
    // Hold-level driven from photo-set; only color filters depend on the
    // active photo (rebuilt when a photo is selected). State filters get
    // hidden because photo-level holds have no route_state.
    setHoldFiltersForPhotosMode();
  } else {
    renderColorFilters();
    renderKfList();
    const firstKf = collectKeyframeIndices()[0];
    if (firstKf !== undefined) selectKeyframe(firstKf);
  }
}

function setHoldFiltersForPhotosMode() {
  const stateGroup = $("#stateFilters");
  if (stateGroup && stateGroup.parentElement) {
    stateGroup.parentElement.style.display = "none";
  }
}

function resetVideoSource() {
  const v = $("#mainVideo");
  if (!v) return;
  v.pause();
  v.currentTime = 0;
  v.src = api("/video");
  v.load();
}

// ---------- header + metrics ----------
function renderHeader() {
  const vid = V.summary.video_id ?? "?";
  const target = V.summary.target_color ?? "?";
  $("#title").innerHTML = `${escapeHtml(vid)} <em>${escapeHtml(target.toLowerCase())}</em>`;
  $("#subtitle").textContent =
    `${fmtDuration(V.summary.duration_sec)} · ${V.summary.fps.toFixed(2)} fps · ${V.summary.frame_count} frames`;

  const badges = $("#metricsBadges");
  badges.innerHTML = "";
  const hbs = V.summary.holds_by_state;
  const items = [
    { k: "holds",    v: V.summary.holds_total, accent: false },
    { k: "core",     v: hbs.core ?? 0, accent: true },
    { k: "possible", v: hbs.possible ?? 0, accent: false },
    { k: "rejected", v: hbs.rejected ?? 0, accent: false },
  ];
  const m = V.summary.metrics || {};
  for (const k of ["total_moves", "mean_move_duration_sec", "pause_fraction", "mean_move_speed"]) {
    if (m[k] !== undefined) {
      const num = typeof m[k] === "number" ? (Math.abs(m[k]) < 10 ? m[k].toFixed(2) : m[k].toFixed(0)) : m[k];
      items.push({ k: k.replaceAll("_"," "), v: num, accent: false });
    }
  }
  items.forEach(({k, v, accent}) => {
    const el = document.createElement("div");
    el.className = "badge" + (accent ? " accent" : "");
    el.innerHTML = `<span class="k">${escapeHtml(k)}</span><b>${escapeHtml(String(v))}</b>`;
    badges.appendChild(el);
  });
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#39;");
}

function fmtDuration(sec) {
  const s = sec|0, m = (s/60)|0;
  return `${m}:${String(s%60).padStart(2,"0")}`;
}

// ---------- tabs ----------
function bindTabs() {
  $$("#tabs .tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#tabs .tab-btn").forEach(b => b.classList.toggle("active", b === btn));
      const key = btn.dataset.tab;
      $$("main .panel").forEach(p => p.classList.toggle("active", p.id === `panel-${key}`));
      if (key === "route") ensureVideoReady();
      if (key === "gantt") renderGantt();
    });
  });
}

// ==================== HOLD-LEVEL ====================

function collectKeyframeIndices() {
  // Keyframes = union of frames_seen across all holds, sorted.
  const set = new Set();
  V.holds.forEach(h => (h.frames_seen || []).forEach(fi => set.add(fi)));
  return Array.from(set).sort((a,b) => a-b);
}

function renderKfList() {
  const kfs = collectKeyframeIndices();
  const list = $("#kfList");
  list.innerHTML = "";
  kfs.forEach((fi) => {
    const t = (fi / V.summary.fps).toFixed(1) + "s";
    const el = document.createElement("div");
    el.className = "kf-thumb";
    el.dataset.fi = fi;
    el.innerHTML = `<img src="${api("/keyframe/")}${fi}" alt="kf ${fi}"><div class="kf-tag">${t}</div>`;
    el.addEventListener("click", () => selectKeyframe(fi));
    list.appendChild(el);
  });
}

// =================== PHOTO-SET (Hold-level only) ===================
function renderPhotoList() {
  const list = $("#kfList");
  list.dataset.kicker = "photos";
  list.innerHTML = "";
  V.photos.forEach(p => {
    const el = document.createElement("div");
    el.className = "kf-thumb photo-thumb";
    el.dataset.pid = p.id;
    el.innerHTML = `
      <img src="/api/photo/${encodeURIComponent(p.id)}/image" alt="${escapeHtml(p.label)}">
      <div class="kf-tag">${escapeHtml(p.label)}</div>
      <div class="kf-num">${p.n_holds}</div>`;
    el.addEventListener("click", () => selectPhoto(p.id));
    list.appendChild(el);
  });
}

async function selectPhoto(pid) {
  V.activePhoto = pid;
  $$("#kfList .kf-thumb").forEach(el => el.classList.toggle("active", el.dataset.pid === pid));

  let cached = V.photoCache.get(pid);
  if (!cached) {
    cached = await fetch(`/api/photo/${encodeURIComponent(pid)}/holds`).then(r => r.json());
    V.photoCache.set(pid, cached);
  }
  // Surface this photo's holds the same way attempt holds are: via V.holds /
  // V.holdsById, because renderHoldsSvg/showHoldTip read those globals.
  V.holds = cached.holds;
  V.holdsById = new Map(cached.holds.map(h => [h.id, h]));

  // Refresh colour-filter chips for this photo's distinct colours.
  renderColorFilters();

  const img = $("#kfImg");
  img.onload = () => renderHoldsSvg(pid);
  img.src = `/api/photo/${encodeURIComponent(pid)}/image`;
  const corner = $("#kfCornerLabel");
  if (corner) corner.textContent = `${cached.label} · ${cached.width}×${cached.height} · ${cached.holds.length} holds`;
}

function renderColorFilters() {
  // Use distinct colors present in holds
  const colors = new Set();
  V.holds.forEach(h => colors.add(h.color));
  const parent = $("#colorFilters");
  parent.innerHTML = "";
  const allBtn = document.createElement("button");
  allBtn.className = "chip active"; allBtn.dataset.color = "all"; allBtn.textContent = "any";
  parent.appendChild(allBtn);
  Array.from(colors).sort().forEach(c => {
    const b = document.createElement("button");
    b.className = "chip chip-color"; b.dataset.color = c;
    b.style.background = COCO_COLOR[c] ?? "#6a7180";
    b.title = c;
    parent.appendChild(b);
  });
}

function bindHoldFilters() {
  $("#stateFilters").addEventListener("click", (e) => {
    const t = e.target.closest(".chip"); if (!t) return;
    const st = t.dataset.state;
    if (st === "all") {
      V.filterState = new Set(["all"]);
    } else {
      V.filterState.delete("all");
      if (V.filterState.has(st)) V.filterState.delete(st);
      else V.filterState.add(st);
      if (V.filterState.size === 0) V.filterState.add("all");
    }
    syncChips("#stateFilters", V.filterState);
    reapplyHoldFilters();
  });
  $("#colorFilters").addEventListener("click", (e) => {
    const t = e.target.closest(".chip"); if (!t) return;
    const c = t.dataset.color;
    if (c === "all") {
      V.filterColor = new Set(["all"]);
    } else {
      V.filterColor.delete("all");
      if (V.filterColor.has(c)) V.filterColor.delete(c);
      else V.filterColor.add(c);
      if (V.filterColor.size === 0) V.filterColor.add("all");
    }
    syncChips("#colorFilters", V.filterColor, "color");
    reapplyHoldFilters();
  });
  $("#togglePolyFill").addEventListener("change", () => {
    const fill = $("#togglePolyFill").checked;
    $$("#kfSvg .hold-poly").forEach(p => p.classList.toggle("no-fill", !fill));
  });
  $("#toggleBboxes").addEventListener("change", () => {
    const show = $("#toggleBboxes").checked;
    $$("#kfSvg .hold-bbox").forEach(b => b.style.display = show ? "" : "none");
  });
}

function syncChips(sel, set, key="state") {
  $$(`${sel} .chip`).forEach(c => {
    const v = c.dataset[key];
    c.classList.toggle("active", set.has(v) || (set.has("all") && v === "all"));
  });
}

function selectKeyframe(fi) {
  V.activeKf = fi;
  $$("#kfList .kf-thumb").forEach(el => el.classList.toggle("active", Number(el.dataset.fi) === fi));
  const img = $("#kfImg");
  img.onload = () => renderHoldsSvg(fi);
  img.src = `${api("/keyframe/")}${fi}`;
  const t = (fi / V.summary.fps).toFixed(2);
  const corner = $("#kfCornerLabel");
  if (corner) corner.textContent = `frame ${fi} · ${t}s`;
}

function renderHoldsSvg(_unusedKey) {
  const img = $("#kfImg");
  const svg = $("#kfSvg");
  const wrap = $("#kfWrap");
  const natW = img.naturalWidth, natH = img.naturalHeight;
  if (!natW || !natH) return;

  // The image has a paper-card border that we must skip, and the SVG's
  // containing block is the .kf-frame (because its CSS filter establishes
  // a containing block). offsetLeft/Top measures the offset RELATIVE to
  // that same positioned ancestor — so it always lines up with the image
  // regardless of layout origin.
  const cs = getComputedStyle(img);
  const bl = parseFloat(cs.borderLeftWidth) || 0;
  const bt = parseFloat(cs.borderTopWidth)  || 0;
  const br = parseFloat(cs.borderRightWidth)|| 0;
  const bb = parseFloat(cs.borderBottomWidth)|| 0;
  const imgRect = img.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${natW} ${natH}`);
  svg.style.left = (img.offsetLeft + bl) + "px";
  svg.style.top  = (img.offsetTop  + bt) + "px";
  svg.style.width  = (imgRect.width  - bl - br) + "px";
  svg.style.height = (imgRect.height - bt - bb) + "px";

  svg.innerHTML = "";
  const fillOn = $("#togglePolyFill").checked;
  const bboxOn = $("#toggleBboxes").checked;
  // In photos mode, draw every hold from the active photo. In keyframes
  // mode, draw only holds whose frames_seen includes the current frame.
  const visible = (V.holdsMode === "photos")
    ? V.holds
    : V.holds.filter(h => (h.frames_seen || []).includes(_unusedKey));

  visible.forEach(h => {
    // In photos mode there is no route_state — colour the polygon by its
    // detected colour pigment instead so the wall reads like a map.
    const stroke = (V.holdsMode === "photos")
      ? (COCO_COLOR[h.color] || "#6e695d")
      : (STATE_COLOR[h.route_state] || "#6e695d");
    const fill = stroke;
    h.polygons.forEach(poly => {
      if (poly.length < 3) return;
      const points = poly.map(p => `${p[0]},${p[1]}`).join(" ");
      const el = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      el.setAttribute("points", points);
      el.setAttribute("class", "hold-poly" + (fillOn ? "" : " no-fill"));
      el.setAttribute("data-id", h.id);
      el.setAttribute("data-state", h.route_state || "detected");
      el.setAttribute("data-color", h.color);
      el.setAttribute("stroke", stroke);
      el.setAttribute("fill", fill);
      el.addEventListener("pointerenter", () => showHoldTip(h));
      el.addEventListener("pointermove", onHoldMouseMove);
      el.addEventListener("pointerleave", hideHoldTip);
      svg.appendChild(el);
    });
    if (bboxOn) {
      const [x1,y1,x2,y2] = h.bbox;
      const r = document.createElementNS("http://www.w3.org/2000/svg","rect");
      r.setAttribute("x", x1); r.setAttribute("y", y1);
      r.setAttribute("width", x2-x1); r.setAttribute("height", y2-y1);
      r.setAttribute("class","hold-bbox");
      r.setAttribute("stroke", stroke);
      svg.appendChild(r);
    }
  });
  reapplyHoldFilters();
}

function reapplyHoldFilters() {
  $$("#kfSvg .hold-poly").forEach(p => {
    const st = p.getAttribute("data-state");
    const col = p.getAttribute("data-color");
    const okS = V.filterState.has("all") || V.filterState.has(st);
    const okC = V.filterColor.has("all") || V.filterColor.has(col);
    p.classList.toggle("dimmed", !(okS && okC));
  });
}

function showHoldTip(h) {
  const t = $("#holdTooltip");
  const swatch = `<span class="ttl-swatch" style="background:${COCO_COLOR[h.color] ?? "#9a9587"}"></span>`;
  const idShort = h.id.startsWith("ph_") ? h.id.slice(3) : (h.id.startsWith("h_") ? h.id.slice(2) : h.id);

  const isPhotoMode = V.holdsMode === "photos";
  let head, extras;
  if (isPhotoMode) {
    const cls = h.seg_class || "hold";
    const cls_color = cls === "volume" ? "var(--teal)" : "var(--rope)";
    head = `
      <div class="ttl-head">
        <span class="ttl-id">№ ${escapeHtml(idShort)}</span>
        <span class="ttl-state" style="color:${cls_color}">${escapeHtml(cls)}</span>
      </div>`;
    extras = `
      <div class="ttl-row">
        <span class="ttl-k">det conf</span>
        <span class="ttl-v">${(h.det_conf*100).toFixed(0)}%</span>
      </div>
      <div class="ttl-row">
        <span class="ttl-k">SAM iou</span>
        <span class="ttl-v">${h.sam_iou ? h.sam_iou.toFixed(2) : "—"}</span>
      </div>
      <div class="ttl-row">
        <span class="ttl-k">fill</span>
        <span class="ttl-v">${h.fill_ratio ? (h.fill_ratio*100).toFixed(0) + "%" : "—"}</span>
      </div>`;
  } else {
    const stateColor = STATE_COLOR[h.route_state] || "var(--ink)";
    head = `
      <div class="ttl-head">
        <span class="ttl-id">№ ${escapeHtml(idShort)}</span>
        <span class="ttl-state" style="color:${stateColor}">${escapeHtml(h.route_state || "?")}</span>
      </div>`;
    extras = `
      <div class="ttl-row">
        <span class="ttl-k">class</span>
        <span class="ttl-v">${escapeHtml(h.seg_class)}</span>
      </div>
      <div class="ttl-row">
        <span class="ttl-k">route score</span>
        <span class="ttl-v">${(h.route_score ?? 0).toFixed(2)}</span>
      </div>
      <div class="ttl-row">
        <span class="ttl-k">seen in</span>
        <span class="ttl-v">${(h.frames_seen || []).length} kf</span>
      </div>`;
  }
  t.innerHTML = `${head}
    <div class="ttl-row">
      <span class="ttl-k">colour</span>
      <span class="ttl-v serif">${swatch}${escapeHtml(h.color)} <span style="color:var(--ink-soft); font-family:var(--mono); font-size:10px; font-style:normal;">${(h.color_conf*100).toFixed(0)}%</span></span>
    </div>
    <div class="ttl-row">
      <span class="ttl-k">type</span>
      <span class="ttl-v serif">${escapeHtml(h.type)} <span style="color:var(--ink-soft); font-family:var(--mono); font-size:10px; font-style:normal;">${(h.type_conf*100).toFixed(0)}%</span></span>
    </div>
    ${extras}`;
  t.hidden = false;
}
function onHoldMouseMove(e) {
  const t = $("#holdTooltip");
  const wrap = $("#kfWrap").getBoundingClientRect();
  const tw = t.offsetWidth, th = t.offsetHeight;
  // Prefer top-right of cursor; flip on right edge
  let x = e.clientX - wrap.left + 14;
  let y = e.clientY - wrap.top + 14;
  if (x + tw > wrap.width - 6)  x = e.clientX - wrap.left - tw - 14;
  if (y + th > wrap.height - 6) y = e.clientY - wrap.top - th - 14;
  t.style.left = Math.max(4, x) + "px";
  t.style.top  = Math.max(4, y) + "px";
}
function hideHoldTip() { $("#holdTooltip").hidden = true; }

// Re-position SVG overlay when image layout changes (resize).
window.addEventListener("resize", () => {
  if (V.activeKf !== null) renderHoldsSvg(V.activeKf);
});


// ==================== ROUTE-LEVEL (video + pose overlay) ====================

let poseRafId = 0;

function bindVideoPanel() {
  const v = $("#mainVideo");
  // src is assigned per-attempt in resetVideoSource()
  const cvs = $("#poseCanvas");
  const resize = () => {
    if (!v.videoWidth) return;
    // Match canvas to the displayed video rect
    const r = v.getBoundingClientRect();
    cvs.width  = v.videoWidth;
    cvs.height = v.videoHeight;
    cvs.style.width  = r.width  + "px";
    cvs.style.height = r.height + "px";
    cvs.style.left   = (r.left - v.parentElement.getBoundingClientRect().left) + "px";
    cvs.style.top    = (r.top  - v.parentElement.getBoundingClientRect().top)  + "px";
  };
  v.addEventListener("loadedmetadata", resize);
  v.addEventListener("resize", resize);
  window.addEventListener("resize", resize);

  const loop = () => {
    paintOverlay();
    poseRafId = requestAnimationFrame(loop);
  };
  v.addEventListener("play", () => { if (!poseRafId) loop(); });
  v.addEventListener("pause", () => { cancelAnimationFrame(poseRafId); poseRafId = 0; paintOverlay(); });
  v.addEventListener("seeked", paintOverlay);
  v.addEventListener("timeupdate", paintOverlay);
  // First frame:
  setTimeout(() => { resize(); paintOverlay(); }, 200);
}

function ensureVideoReady() {
  const v = $("#mainVideo");
  if (v && v.readyState === 0) v.load();
}

function currentFrameIdx() {
  const v = $("#mainVideo");
  const fps = V.summary.fps || 25;
  return Math.round(v.currentTime * fps);
}

function nearestPoseFrame(fi) {
  // pose frames are sampled (may not have every video frame). Binary search
  // the closest.
  const arr = V.pose;
  if (!arr.length) return null;
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].frame < fi) lo = mid + 1; else hi = mid;
  }
  let best = arr[lo];
  if (lo > 0 && Math.abs(arr[lo-1].frame - fi) < Math.abs(best.frame - fi)) best = arr[lo-1];
  return best;
}

function paintOverlay() {
  const v = $("#mainVideo");
  const cvs = $("#poseCanvas");
  if (!cvs.width) return;
  const ctx = cvs.getContext("2d");
  ctx.clearRect(0, 0, cvs.width, cvs.height);

  const fi = currentFrameIdx();
  $("#timecode").textContent = `${v.currentTime.toFixed(2)} s · frame ${fi}`;

  const showRouteHolds = $("#showRouteHolds").checked;
  const showPose = $("#showPose").checked;
  const showContactsDots = $("#showContactsOnVid").checked;

  if (showRouteHolds) drawRouteHoldsOverlay(ctx, cvs.width, cvs.height);
  if (showPose) {
    const pf = nearestPoseFrame(fi);
    if (pf) drawSkeleton(ctx, pf.keypoints);
  }
  if (showContactsDots) drawContactDots(ctx, fi);

  updateLiveContactsPanel(fi);
}

function drawRouteHoldsOverlay(ctx, W, H) {
  ctx.save();
  V.holds.forEach(h => {
    if (h.route_state === "rejected") return;
    const c = STATE_COLOR[h.route_state] || "#6a7180";
    ctx.fillStyle = c + "33"; // low alpha
    ctx.strokeStyle = c + "bb";
    ctx.lineWidth = 1.2;
    h.polygons.forEach(poly => {
      if (poly.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
  });
  ctx.restore();
}

function drawSkeleton(ctx, kps) {
  ctx.save();
  ctx.strokeStyle = "#4bc07a";
  ctx.lineWidth = 2.4;
  ctx.lineCap = "round";
  SKELETON_EDGES.forEach(([a, b]) => {
    const pa = kps[a], pb = kps[b];
    if (!pa || !pb) return;
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  });
  ctx.fillStyle = "#fff";
  KP_DOTS.forEach(n => {
    const p = kps[n]; if (!p) return;
    ctx.beginPath(); ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2); ctx.fill();
  });
  ctx.restore();
}

function currentContactOnLimb(fi, limb) {
  const segs = V.contacts[limb] || [];
  // Segments are in video-frame space (remapped by the pipeline).
  for (const s of segs) {
    if (s.start_frame <= fi && fi <= s.end_frame) return s;
  }
  return null;
}

function drawContactDots(ctx, fi) {
  const pf = nearestPoseFrame(fi);
  if (!pf) return;
  const kps = pf.keypoints;
  const anchorByLimb = {
    left_hand:  ["left_wrist","left_hand_5","left_hand_9"],
    right_hand: ["right_wrist","right_hand_5","right_hand_9"],
    left_foot:  ["left_big_toe","left_ankle","left_heel"],
    right_foot: ["right_big_toe","right_ankle","right_heel"],
  };
  LIMB_ORDER.forEach(limb => {
    const seg = currentContactOnLimb(fi, limb);
    const color = LIMB_HEX[limb];
    const anchor = anchorByLimb[limb].map(n => kps[n]).find(Boolean);
    if (!anchor) return;
    ctx.save();
    // Ring around limb anchor:
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(anchor[0], anchor[1], 10, 0, Math.PI * 2);
    ctx.stroke();
    // Filled dot if in contact with a hold, hollow if no contact/occluded:
    if (seg && seg.state === "contact" && seg.hold_id) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(anchor[0], anchor[1], 6, 0, Math.PI * 2);
      ctx.fill();
      // Line to hold centre:
      const h = V.holdsById.get(seg.hold_id);
      if (h) {
        ctx.strokeStyle = color + "bb";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(anchor[0], anchor[1]);
        ctx.lineTo(h.center[0], h.center[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        // Target circle on the hold:
        ctx.fillStyle = color + "88";
        ctx.beginPath();
        ctx.arc(h.center[0], h.center[1], 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  });
}

function updateLiveContactsPanel(fi) {
  const panel = $("#liveContacts");
  if (!panel.dataset.ready) {
    panel.innerHTML = `<h3>current contacts</h3>
      <span class="lc-kicker">live · per-limb</span>
      <div id="limbCards"></div>
      <div class="stats-mini" id="liveStats"></div>`;
    panel.dataset.ready = "1";
  }
  const box = $("#limbCards", panel);
  box.innerHTML = "";
  LIMB_ORDER.forEach(limb => {
    const seg = currentContactOnLimb(fi, limb);
    const card = document.createElement("div"); card.className = "limb-card";
    const dot = document.createElement("div"); dot.className = "limb-dot " + limb;
    card.appendChild(dot);
    const name = document.createElement("span"); name.className = "lb-name";
    name.textContent = limb.replace("_", " ");
    card.appendChild(name);
    const val = document.createElement("span"); val.className = "lb-val";
    if (!seg || seg.state === "none") { val.textContent = "—"; val.classList.add("lb-state-none"); }
    else if (seg.state === "occluded") { val.textContent = "occluded"; val.classList.add("lb-state-occluded"); }
    else if (seg.state === "contact") {
      const h = V.holdsById.get(seg.hold_id);
      if (h) {
        const sw = `<span class="ttl-swatch" style="background:${COCO_COLOR[h.color]??"#6a7180"}"></span>`;
        val.innerHTML = `${sw}<b>${h.color}</b> ${h.type} <span style="color:#8b93a4">(${h.id.slice(-6)})</span>`;
      } else {
        val.textContent = seg.hold_id;
      }
    }
    card.appendChild(val);
    box.appendChild(card);
  });
  const m = V.summary.metrics || {};
  const st = $("#liveStats", panel);
  const rows = [
    ["frame", fi],
    ["time", ($("#mainVideo").currentTime).toFixed(2) + "s"],
    ...(m.total_moves !== undefined ? [["moves total", m.total_moves]] : []),
    ...(m.readjustments_count !== undefined ? [["readjustments", m.readjustments_count]] : []),
    ...(m.hesitations_count !== undefined ? [["hesitations", m.hesitations_count]] : []),
  ];
  st.innerHTML = rows.map(([k,v]) => `<div class="row"><span>${k}</span><span>${v}</span></div>`).join("");
}


// ==================== CONTACTS GANTT ====================

let ganttBuilt = false;

function renderGantt(force = false) {
  if (ganttBuilt && !force) return;
  ganttBuilt = false;
  const wrap = $("#ganttWrap");
  wrap.innerHTML = "";
  const total = V.summary.frame_count || 1;
  const fps = V.summary.fps || 25;

  LIMB_ORDER.forEach(limb => {
    const row = document.createElement("div"); row.className = "gantt-row";
    const lbl = document.createElement("div"); lbl.className = "gantt-label";
    lbl.textContent = limb.replace("_", " ");
    const bar = document.createElement("div"); bar.className = "gantt-bar";
    const color = LIMB_HEX[limb];
    const segs = V.contacts[limb] || [];
    segs.forEach(s => {
      const left = (s.start_frame / total) * 100;
      const width = Math.max(0.15, (s.end_frame - s.start_frame + 1) / total * 100);
      const seg = document.createElement("div");
      seg.className = "gantt-seg state-" + s.state;
      seg.style.left = left + "%";
      seg.style.width = width + "%";
      if (s.state === "contact") {
        seg.style.background = color;
        seg.style.opacity = 0.45 + 0.45 * (s.confidence || 0);
      }
      const h = s.hold_id ? V.holdsById.get(s.hold_id) : null;
      const hlabel = h ? `${h.color} ${h.type} ${h.id.slice(-6)}` : (s.hold_id || s.state);
      seg.title = `${limb}  [${s.state}]\n` +
        `frames: ${s.start_frame}–${s.end_frame} (${((s.end_frame-s.start_frame+1)/fps).toFixed(1)}s)\n` +
        `hold: ${hlabel}\nconfidence: ${(s.confidence||0).toFixed(2)}`;
      seg.addEventListener("click", () => {
        const vid = $("#mainVideo");
        vid.currentTime = s.start_frame / fps;
        $$("#tabs .tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === "route"));
        $$("main .panel").forEach(p => p.classList.toggle("active", p.id === "panel-route"));
      });
      bar.appendChild(seg);
    });
    // Cursor to track current video time
    const cursor = document.createElement("div");
    cursor.className = "gantt-cursor";
    cursor.id = `cursor-${limb}`;
    cursor.style.left = "0%";
    bar.appendChild(cursor);
    row.appendChild(lbl);
    row.appendChild(bar);
    wrap.appendChild(row);
  });
  // Time axis
  const axis = document.createElement("div"); axis.className = "gantt-axis";
  const axisLbl = document.createElement("div"); axisLbl.textContent = "";
  const ticks = document.createElement("div"); ticks.className = "gantt-ticks";
  const nTicks = 6;
  for (let i = 0; i <= nTicks; i++) {
    const f = Math.round(i / nTicks * total);
    const t = (f / fps).toFixed(1);
    const sp = document.createElement("span");
    sp.style.left = (i / nTicks * 100) + "%";
    sp.textContent = `${t}s`;
    ticks.appendChild(sp);
  }
  axis.appendChild(axisLbl);
  axis.appendChild(ticks);
  wrap.appendChild(axis);

  // Update cursor on video time change (bind once)
  const vid = $("#mainVideo");
  if (vid && !vid.dataset.ganttCursorBound) {
    vid.dataset.ganttCursorBound = "1";
    vid.addEventListener("timeupdate", () => {
      const tot = V.summary.frame_count || 1;
      const f = V.summary.fps || 25;
      const pct = (vid.currentTime * f) / tot * 100;
      LIMB_ORDER.forEach(limb => {
        const c = document.getElementById(`cursor-${limb}`);
        if (c) c.style.left = pct + "%";
      });
    });
  }
  ganttBuilt = true;
}

function renderEventsPanel() {
  const panel = $("#eventsPanel");
  const moves = V.events.move_events || [];
  const readj = V.events.readjustments || [];
  const hesit = V.events.hesitations || [];
  const fps = V.summary.fps || 25;

  const html = [];
  html.push("<h3>move events</h3>");
  moves.slice(0, 40).forEach(e => {
    const t = (e.start_frame / fps).toFixed(1);
    const fromTo = [e.from_hold, e.to_hold].filter(Boolean).map(id => id ? id.slice(-6) : "?").join(" → ");
    html.push(
      `<div class="event-item" data-t="${e.start_frame/fps}">
         <div class="ev-title">${e.limb?.replace("_"," ") ?? "move"} · ${fromTo}</div>
         <div class="ev-meta"><span>${t}s</span><span>Δ ${(e.duration_sec||0).toFixed(2)}s</span></div>
       </div>`);
  });
  if (readj.length) {
    html.push("<h3 style='margin-top:12px'>readjustments</h3>");
    readj.forEach(e => {
      const t = (e.start_frame / fps).toFixed(1);
      html.push(
        `<div class="event-item" data-t="${e.start_frame/fps}">
           <div class="ev-title">${e.limb?.replace("_"," ") ?? "readjust"} · hold ${e.hold_id ? e.hold_id.slice(-6) : "?"}</div>
           <div class="ev-meta"><span>${t}s</span></div>
         </div>`);
    });
  }
  if (hesit.length) {
    html.push("<h3 style='margin-top:12px'>hesitations</h3>");
    hesit.forEach(e => {
      const t = (e.start_frame / fps).toFixed(1);
      html.push(
        `<div class="event-item" data-t="${e.start_frame/fps}">
           <div class="ev-title">hesitation · ${(e.duration_sec||0).toFixed(2)}s</div>
           <div class="ev-meta"><span>${t}s</span></div>
         </div>`);
    });
  }
  panel.innerHTML = html.join("");
  if (!panel.dataset.eventsBound) {
    panel.dataset.eventsBound = "1";
    panel.addEventListener("click", (e) => {
      const item = e.target.closest(".event-item");
      if (!item) return;
      const vid = $("#mainVideo");
      vid.currentTime = Number(item.dataset.t) || 0;
      $$("#tabs .tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === "route"));
      $$("main .panel").forEach(p => p.classList.toggle("active", p.id === "panel-route"));
    });
  }
}

})();
