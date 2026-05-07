const canvas = document.getElementById('trajCanvas');
const ctx = canvas.getContext('2d');
const mapEl = document.getElementById('map');

const playPauseBtn = document.getElementById('playPauseBtn');
const resetBtn = document.getElementById('resetBtn');
const togglePredictBtn = document.getElementById('togglePredictBtn');
const speedSelect = document.getElementById('speedSelect');
const assocDelayRange = document.getElementById('assocDelayRange');
const assocDelayText = document.getElementById('assocDelayText');
const fileInput = document.getElementById('fileInput');

const statusText = document.getElementById('statusText');
const timeText = document.getElementById('timeText');
const pointCountText = document.getElementById('pointCountText');
const trajCountText = document.getElementById('trajCountText');
const pointsPreviewMeta = document.getElementById('pointsPreviewMeta');
const hoverTrajTag = document.getElementById('hoverTrajTag');

const trajSearchInput = document.getElementById('trajSearchInput');
const clearFilterBtn = document.getElementById('clearFilterBtn');
const trajList = document.getElementById('trajList');
const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');
const pageInfoText = document.getElementById('pageInfoText');

const DEFAULT_ASSOC_PATH = '../data/output/1_associated.csv';
const DEFAULT_PREDICT_PATHS = ['../data/input/predict_result.csv', '../data/output/predict_result.csv'];
const PAGE_SIZE = 10;

let assocPoints = [];
let predictPoints = [];
let activePoints = [];
let sortedByTime = [];
let extent = null;

let running = false;
let cursor = 0;
let simTime = 0;
let lastFrame = 0;
let displayedCount = 0;

const pendingAssociations = [];
const tracks = new Map();

let speed = Number(speedSelect.value);
let assocDelayMs = Number(assocDelayRange.value);
let canvasCssWidth = 0;
let canvasCssHeight = 0;
let map = null;
let mapReady = false;
let tileLayer = null;
let tileProviderIndex = 0;

let showPredict = true;
let selectedTrajId = null;
let searchKeyword = '';
let pageIndex = 1;
let hoveredTrajId = null;

const TILE_PROVIDERS = [
  {
    name: '高德地图',
    url: 'https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
    options: { subdomains: ['1', '2', '3', '4'], maxZoom: 18 },
  },
  {
    name: 'OpenStreetMap',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    options: { subdomains: ['a', 'b', 'c'], maxZoom: 19 },
  },
];

function parseTimeToSeconds (ts) {
  if (ts == null) return NaN;
  const raw = String(ts).trim();
  if (/^\d+(\.\d+)?$/.test(raw)) return Number(raw);

  const parts = raw.split(':');
  if (parts.length === 2) {
    const mm = Number(parts[0]);
    const ss = Number(parts[1]);
    if (!Number.isNaN(mm) && !Number.isNaN(ss)) return mm * 60 + ss;
  }
  if (parts.length === 3) {
    const hh = Number(parts[0]);
    const mm = Number(parts[1]);
    const ss = Number(parts[2]);
    if (!Number.isNaN(hh) && !Number.isNaN(mm) && !Number.isNaN(ss)) return hh * 3600 + mm * 60 + ss;
  }
  return NaN;
}

function parseCsv (text, sourceTag) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const header = lines[0].split(',').map(h => h.trim());
  const index = Object.fromEntries(header.map((h, i) => [h, i]));
  for (const k of ['ts', 'x', 'y']) {
    if (!(k in index)) throw new Error(`CSV缺少字段: ${k}`);
  }
  if (!("id" in index) && !("tra_id" in index)) {
    throw new Error('CSV缺少字段: id（或兼容字段tra_id）');
  }

  const data = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(',');
    if (cells.length < header.length) continue;

    const t = parseTimeToSeconds(cells[index.ts]);
    const x = Number(cells[index.x]);
    const y = Number(cells[index.y]);
    const rawId = (index.id !== undefined) ? cells[index.id] : cells[index.tra_id];
    const traId = Number(rawId);
    if ([t, x, y, traId].some(Number.isNaN)) continue;

    data.push({
      t,
      tsRaw: cells[index.ts],
      x,
      y,
      traId,
      source: sourceTag,
    });
  }
  return data;
}

function colorForTraj (id) {
  const hue = (id * 47) % 360;
  return `hsl(${hue} 90% 52%)`;
}

function isSelectedTraj (id) {
  return selectedTrajId == null || id === selectedTrajId;
}

function resetState () {
  cursor = 0;
  simTime = sortedByTime.length ? sortedByTime[0].t : 0;
  lastFrame = 0;
  displayedCount = 0;
  pendingAssociations.length = 0;
  tracks.clear();

  for (const p of sortedByTime) {
    if (!tracks.has(p.traId)) tracks.set(p.traId, []);
  }

  hideHoverTrajTag();

  draw();
  updateHud();
}

function setTileProvider (index) {
  if (!map) return;
  tileProviderIndex = index;

  if (tileLayer) {
    tileLayer.off();
    map.removeLayer(tileLayer);
    tileLayer = null;
  }

  const provider = TILE_PROVIDERS[tileProviderIndex];
  let tileErrors = 0;
  tileLayer = L.tileLayer(provider.url, {
    ...provider.options,
    attribution: '&copy; map data providers',
  });

  tileLayer.on('tileerror', () => {
    tileErrors += 1;
    if (tileErrors > 8 && tileProviderIndex < TILE_PROVIDERS.length - 1) {
      setTileProvider(tileProviderIndex + 1);
      statusText.textContent = `地图源切换为 ${TILE_PROVIDERS[tileProviderIndex].name}`;
    }
  });

  tileLayer.addTo(map);
}

function initMap () {
  if (typeof L === 'undefined') {
    statusText.textContent = '地图脚本加载失败，请检查网络后刷新页面。';
    return;
  }
  map = L.map(mapEl, { zoomControl: true, preferCanvas: true });
  setTileProvider(0);
  map.setView([35.8617, 104.1954], 4);
  mapReady = true;
  map.on('move zoom resize', draw);
  map.on('mousemove', handleMapMouseMove);
  map.on('mouseout', hideHoverTrajTag);
  map.on('dragstart', hideHoverTrajTag);
}

function pointToSegmentDistance (px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 <= 1e-9) {
    const dx = px - ax;
    const dy = py - ay;
    return Math.sqrt(dx * dx + dy * dy);
  }
  const t = Math.max(0, Math.min(1, (apx * abx + apy * aby) / ab2));
  const cx = ax + t * abx;
  const cy = ay + t * aby;
  const dx = px - cx;
  const dy = py - cy;
  return Math.sqrt(dx * dx + dy * dy);
}

function findHoveredTrajId (containerPoint) {
  const hitThreshold = 8;
  let best = { id: null, dist: Infinity };

  for (const [trajId, pts] of tracks.entries()) {
    if (!isSelectedTraj(trajId)) continue;

    for (let i = 0; i < pts.length; i++) {
      const p = worldToCanvas(pts[i].x, pts[i].y);
      const dx = containerPoint.x - p.x;
      const dy = containerPoint.y - p.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < best.dist) best = { id: trajId, dist: d };

      if (i > 0) {
        const p0 = worldToCanvas(pts[i - 1].x, pts[i - 1].y);
        const dl = pointToSegmentDistance(containerPoint.x, containerPoint.y, p0.x, p0.y, p.x, p.y);
        if (dl < best.dist) best = { id: trajId, dist: dl };
      }
    }
  }

  if (best.dist <= hitThreshold) return best.id;

  for (let i = cursor - 1; i >= 0; i--) {
    const p = sortedByTime[i];
    if (!isSelectedTraj(p.traId)) continue;
    const cv = worldToCanvas(p.x, p.y);
    const dx = containerPoint.x - cv.x;
    const dy = containerPoint.y - cv.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d <= hitThreshold) return p.traId;
  }

  return null;
}

function showHoverTrajTag (trajId, containerPoint) {
  hoveredTrajId = trajId;
  hoverTrajTag.style.display = 'block';
  hoverTrajTag.textContent = `轨迹ID: ${trajId}`;
  hoverTrajTag.style.left = `${containerPoint.x}px`;
  hoverTrajTag.style.top = `${containerPoint.y}px`;
}

function hideHoverTrajTag () {
  hoveredTrajId = null;
  hoverTrajTag.style.display = 'none';
}

function handleMapMouseMove (e) {
  if (!sortedByTime.length) {
    hideHoverTrajTag();
    return;
  }
  const hitId = findHoveredTrajId(e.containerPoint);
  if (hitId == null) {
    hideHoverTrajTag();
    return;
  }
  showHoverTrajTag(hitId, e.containerPoint);
}

function focusMapToData () {
  if (!mapReady || !extent) return;
  const sameLng = Math.abs(extent.maxX - extent.minX) < 1e-9;
  const sameLat = Math.abs(extent.maxY - extent.minY) < 1e-9;
  if (sameLng && sameLat) {
    map.setView([extent.minY, extent.minX], 15);
    return;
  }
  map.fitBounds(L.latLngBounds([extent.minY, extent.minX], [extent.maxY, extent.maxX]), {
    padding: [30, 30],
    maxZoom: 16,
    animate: false,
  });
}

function rebuildActiveData ({ resetAnimation = true, focusMap = false } = {}) {
  activePoints = showPredict ? [...assocPoints, ...predictPoints] : [...assocPoints];
  sortedByTime = [...activePoints].sort((a, b) => a.t - b.t);

  if (selectedTrajId != null && !sortedByTime.some(p => p.traId === selectedTrajId)) {
    selectedTrajId = null;
  }

  if (!sortedByTime.length) {
    extent = null;
    pointsPreviewMeta.textContent = '暂无数据';
    trajList.innerHTML = '<div class="empty-tip">无可显示轨迹</div>';
    hideHoverTrajTag();
    draw();
    return;
  }

  const xs = sortedByTime.map(p => p.x);
  const ys = sortedByTime.map(p => p.y);
  extent = { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };

  if (focusMap) focusMapToData();
  if (resetAnimation) resetState(); else draw();

  renderTrajectoryList();
  const assocCount = assocPoints.length;
  const predCount = predictPoints.length;
  pointsPreviewMeta.textContent = `关联点 ${assocCount}，预测点 ${predCount}，当前显示 ${sortedByTime.length}`;
}

function getFilteredIds () {
  const ids = [...new Set(sortedByTime.map(p => p.traId))].sort((a, b) => a - b);
  if (!searchKeyword) return ids;
  return ids.filter(id => String(id).includes(searchKeyword));
}

function renderTrajectoryList () {
  const ids = getFilteredIds();
  const totalPages = Math.max(1, Math.ceil(ids.length / PAGE_SIZE));
  pageIndex = Math.min(Math.max(1, pageIndex), totalPages);
  const start = (pageIndex - 1) * PAGE_SIZE;
  const pageIds = ids.slice(start, start + PAGE_SIZE);

  pageInfoText.textContent = `${pageIndex} / ${totalPages}`;
  prevPageBtn.disabled = pageIndex <= 1;
  nextPageBtn.disabled = pageIndex >= totalPages;

  if (!pageIds.length) {
    trajList.innerHTML = '<div class="empty-tip">没有匹配到轨迹ID</div>';
    return;
  }

  trajList.innerHTML = pageIds.map(id => (
    `<button class="traj-item ${selectedTrajId === id ? 'active' : ''}" data-id="${id}">轨迹 ${id}</button>`
  )).join('');

  trajList.querySelectorAll('.traj-item').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedTrajId = Number(btn.dataset.id);
      renderTrajectoryList();
      draw();
      updateHud();
      statusText.textContent = `已选中轨迹 ${selectedTrajId}，仅显示该轨迹。`;
    });
  });
}

function worldToCanvas (lng, lat) {
  if (!mapReady) return { x: 0, y: 0 };
  const p = map.latLngToContainerPoint([lat, lng]);
  return { x: p.x, y: p.y };
}

function formatSec (sec) {
  const total = Math.max(0, sec);
  const mm = Math.floor(total / 60);
  const ss = (total % 60).toFixed(1).padStart(4, '0');
  return `${String(mm).padStart(2, '0')}:${ss}`;
}

function updateHud () {
  timeText.textContent = formatSec(simTime);
  pointCountText.textContent = String(displayedCount);
  const visibleIds = new Set(sortedByTime.filter(p => p.t <= simTime && isSelectedTraj(p.traId)).map(p => p.traId));
  trajCountText.textContent = String(visibleIds.size);
}

function step (now) {
  if (!running) return;
  if (!lastFrame) lastFrame = now;
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  simTime += dt * speed;

  while (cursor < sortedByTime.length && sortedByTime[cursor].t <= simTime) {
    const p = sortedByTime[cursor++];
    displayedCount++;
    pendingAssociations.push({ dueAt: now + assocDelayMs, point: { ...p } });
  }

  while (pendingAssociations.length && pendingAssociations[0].dueAt <= now) {
    const { point } = pendingAssociations.shift();
    tracks.get(point.traId).push(point);
  }

  if (cursor >= sortedByTime.length && pendingAssociations.length === 0) {
    running = false;
    playPauseBtn.textContent = '播放';
    statusText.textContent = '播放完成。可点击“重置”重新播放。';
  }

  draw();
  updateHud();
  requestAnimationFrame(step);
}

function draw () {
  if (!extent || !mapReady) return;
  ctx.clearRect(0, 0, canvasCssWidth, canvasCssHeight);

  for (let i = 0; i < cursor; i++) {
    const p = sortedByTime[i];
    if (!isSelectedTraj(p.traId)) continue;
    const cv = worldToCanvas(p.x, p.y);
    ctx.beginPath();
    ctx.fillStyle = p.source === 'predict' ? 'rgba(231, 108, 25, 0.8)' : 'rgba(24, 35, 58, 0.75)';
    ctx.arc(cv.x, cv.y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.globalAlpha = 0.95;
  ctx.lineWidth = 2;
  for (const [trajId, pts] of tracks.entries()) {
    if (pts.length < 2 || !isSelectedTraj(trajId)) continue;
    ctx.beginPath();
    ctx.strokeStyle = colorForTraj(trajId);
    const first = worldToCanvas(pts[0].x, pts[0].y);
    ctx.moveTo(first.x, first.y);
    for (let i = 1; i < pts.length; i++) {
      const cv = worldToCanvas(pts[i].x, pts[i].y);
      ctx.lineTo(cv.x, cv.y);
    }
    ctx.stroke();
  }

  if (cursor > 0) {
    for (let i = cursor - 1; i >= 0; i--) {
      const p = sortedByTime[i];
      if (!isSelectedTraj(p.traId)) continue;
      const cv = worldToCanvas(p.x, p.y);
      ctx.beginPath();
      ctx.fillStyle = colorForTraj(p.traId);
      ctx.arc(cv.x, cv.y, 4.4, 0, Math.PI * 2);
      ctx.fill();
      break;
    }
  }
}

function resizeCanvas () {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvasCssWidth = Math.max(1, Math.floor(rect.width));
  canvasCssHeight = Math.max(1, Math.floor(rect.height));
  canvas.width = Math.max(1, Math.floor(canvasCssWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvasCssHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (mapReady) map.invalidateSize();
  draw();
}

async function fetchCsvFromCandidates (paths, sourceTag) {
  for (const path of paths) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) continue;
      const text = await resp.text();
      return parseCsv(text, sourceTag);
    } catch {
      continue;
    }
  }
  return [];
}

async function tryLoadDefaultCsv () {
  assocPoints = await fetchCsvFromCandidates([DEFAULT_ASSOC_PATH], 'assoc');
  predictPoints = await fetchCsvFromCandidates(DEFAULT_PREDICT_PATHS, 'predict');

  if (!assocPoints.length && !predictPoints.length) {
    statusText.textContent = '默认CSV加载失败，请使用“导入CSV”。';
    return;
  }

  rebuildActiveData({ resetAnimation: true, focusMap: true });
  statusText.textContent = `加载完成：关联 ${assocPoints.length} 点，预测 ${predictPoints.length} 点。`;
}

playPauseBtn.addEventListener('click', () => {
  if (!sortedByTime.length) return;
  running = !running;
  playPauseBtn.textContent = running ? '暂停' : '播放';
  statusText.textContent = running ? '播放中…' : '已暂停。';
  if (running) {
    lastFrame = 0;
    requestAnimationFrame(step);
  }
});

resetBtn.addEventListener('click', () => {
  running = false;
  playPauseBtn.textContent = '播放';
  resetState();
  statusText.textContent = '已重置。点击“播放”开始。';
});

togglePredictBtn.addEventListener('click', () => {
  showPredict = !showPredict;
  togglePredictBtn.textContent = showPredict ? '隐藏预测' : '显示预测';
  running = false;
  playPauseBtn.textContent = '播放';
  rebuildActiveData({ resetAnimation: true, focusMap: true });
  statusText.textContent = showPredict ? '已显示预测轨迹。' : '已隐藏预测轨迹。';
});

speedSelect.addEventListener('change', () => {
  speed = Number(speedSelect.value);
});

assocDelayRange.addEventListener('input', () => {
  assocDelayMs = Number(assocDelayRange.value);
  assocDelayText.textContent = `${assocDelayMs}ms`;
});

trajSearchInput.addEventListener('input', () => {
  searchKeyword = trajSearchInput.value.trim();
  pageIndex = 1;
  renderTrajectoryList();
});

clearFilterBtn.addEventListener('click', () => {
  selectedTrajId = null;
  searchKeyword = '';
  trajSearchInput.value = '';
  pageIndex = 1;
  renderTrajectoryList();
  draw();
  updateHud();
  statusText.textContent = '已显示全部轨迹。';
});

prevPageBtn.addEventListener('click', () => {
  pageIndex = Math.max(1, pageIndex - 1);
  renderTrajectoryList();
});

nextPageBtn.addEventListener('click', () => {
  pageIndex += 1;
  renderTrajectoryList();
});

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    assocPoints = parseCsv(text, 'assoc');
    predictPoints = [];
    showPredict = false;
    togglePredictBtn.textContent = '显示预测';
    rebuildActiveData({ resetAnimation: true, focusMap: true });
    statusText.textContent = `已导入 ${file.name}，点数 ${assocPoints.length}。`;
  } catch (err) {
    statusText.textContent = `导入失败：${err.message}`;
  }
});

window.addEventListener('resize', resizeCanvas);

(async function init () {
  initMap();
  resizeCanvas();
  await tryLoadDefaultCsv();
})();
