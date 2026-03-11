#include "report_writer.hpp"
#include "db/db_manager.hpp"
#include <duckdb.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace derep {

namespace {

static std::string esc_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20)  { /* skip control chars */ }
        else                out += c;
    }
    return out;
}

static std::string fmt_d(double v, int prec = 4) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm visualization page  (__PREFIX__ substituted at write time)
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
// Algorithm visualization page  (__PREFIX__ substituted at write time)
// ─────────────────────────────────────────────────────────────────────────────
static const char* ALGORITHM_HTML = R"ALGO(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geodesic · __PREFIX__ · algorithm</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=Outfit:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#fff;--bg2:#f8f8f6;--bg3:#f0f0ed;
  --border:#e5e5e2;--border-dark:#ccccc8;
  --text:#0a0a08;--text2:#52524e;--text3:#a8a8a2;
  --ink:#0a0a08;--green:#15803d;--amber:#b45309;--rose:#9f1239;
  --green-d:rgba(21,128,61,.07);--ink-d:rgba(10,10,8,.05);
}
html,body{height:100%;overflow:hidden;display:flex;flex-direction:column}
body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;font-size:14px;line-height:1.6}
.nav{display:flex;align-items:center;gap:16px;padding:0 24px;height:44px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0}
.nav-logo{font-family:'Geist Mono',monospace;font-size:13px;font-weight:500;color:var(--text)}
.nav-back{font-size:12px;color:var(--text3);text-decoration:none;transition:color .15s}
.nav-back:hover{color:var(--text)}
.nav-prefix{font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3);margin-left:auto}
#algorithm{flex:1;display:flex;overflow:hidden;background:var(--bg)}
#algorithm .alg-panel{
  width:320px;min-width:320px;background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;z-index:10;
}
#algorithm .alg-panel-hdr{padding:24px 24px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
#algorithm .alg-logo{font-family:'Geist Mono',monospace;font-size:15px;font-weight:500;color:var(--green);letter-spacing:-.02em}
#algorithm .alg-logo-sub{margin-top:4px;font-size:11px;color:var(--text3);letter-spacing:.08em;text-transform:uppercase}
#algorithm .alg-steps{flex:1;overflow-y:auto;padding:16px 0}
#algorithm .alg-steps::-webkit-scrollbar{width:4px}
#algorithm .alg-steps::-webkit-scrollbar-thumb{background:var(--border-dark);border-radius:2px}
#algorithm .alg-step{padding:14px 24px;border-left:2px solid transparent;cursor:pointer;transition:all .15s;opacity:.5}
#algorithm .alg-step.active{border-left-color:var(--green);opacity:1;background:var(--green-d)}
#algorithm .alg-step:hover{opacity:.8}
#algorithm .alg-step-num{font-family:'Geist Mono',monospace;font-size:10px;color:var(--green);letter-spacing:.12em;margin-bottom:4px}
#algorithm .alg-step-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px}
#algorithm .alg-step-body{font-size:12px;color:var(--text2);line-height:1.6;display:none}
#algorithm .alg-step.active .alg-step-body{display:block}
#algorithm .alg-doc-link{display:block;margin-top:10px;font-size:11px;color:var(--text3);text-decoration:none;letter-spacing:.04em;transition:color .15s}
#algorithm .alg-doc-link:hover{color:var(--green)}
#algorithm .alg-controls{padding:16px 24px;border-top:1px solid var(--border);flex-shrink:0}
#algorithm .alg-ctrl-row{display:flex;gap:8px;margin-bottom:8px;align-items:center}
#algorithm .alg-ctrl-row:last-child{margin-bottom:0}
#algorithm .alg-btn{
  background:var(--bg);border:1px solid var(--border);border-radius:5px;
  padding:7px 14px;color:var(--text2);font-family:'Outfit',sans-serif;
  font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;white-space:nowrap;
}
#algorithm .alg-btn:hover{color:var(--text);border-color:var(--border-dark)}
#algorithm .alg-btn.primary{color:var(--green);border-color:rgba(21,128,61,.3);background:var(--green-d)}
#algorithm .alg-btn.primary:hover{background:rgba(21,128,61,.12)}
#algorithm .alg-btn:disabled{opacity:.35;cursor:not-allowed}
#algorithm .alg-step-nav{display:flex;align-items:center;gap:8px;flex:1}
#algorithm .alg-step-lbl{flex:1;text-align:center;font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3)}
#algorithm .alg-info-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:8px;
}
#algorithm .alg-info-cell{background:var(--bg);padding:10px 12px}
#algorithm .alg-info-key{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:2px}
#algorithm .alg-info-val{font-family:'Geist Mono',monospace;font-size:13px;font-weight:500;color:var(--text)}
#algorithm .alg-canvas-wrap{flex:1;position:relative;overflow:hidden;background:var(--bg3)}
#algorithm #sphere-canvas{width:100%;height:100%;display:block}
#algorithm .alg-canvas-lbl{
  position:absolute;top:20px;right:24px;
  font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3);
  letter-spacing:.08em;text-transform:uppercase;
}
#algorithm .alg-legend{position:absolute;bottom:24px;right:24px;display:flex;flex-direction:column;gap:6px}
#algorithm .alg-legend-item{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text2)}
#algorithm .alg-legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

#algorithm .alg-insight{
  background:rgba(21,128,61,.05);border-left:2px solid var(--green);
  padding:6px 10px;margin:8px 0;font-size:11.5px;color:var(--text2);
  line-height:1.6;font-style:italic;
}
#algorithm .alg-btn.alg-btn-tour{color:var(--amber);border-color:rgba(180,83,9,.3)}
#algorithm .alg-btn.alg-btn-tour:hover{background:rgba(180,83,9,.07)}
#algorithm .alg-progress{
  position:relative;height:26px;background:var(--bg3);
  border:1px solid var(--border);border-radius:5px;
  overflow:hidden;margin-bottom:8px;display:flex;align-items:center;padding:0 10px;
}
#algorithm .alg-progress-bar{
  position:absolute;left:0;top:0;bottom:0;
  background:var(--green-d);border-right:2px solid var(--green);
  transition:width .4s ease;
}
#algorithm .alg-progress-label{
  position:relative;font-family:'Geist Mono',monospace;
  font-size:11px;color:var(--green);font-weight:500;
}
</style>
</head>
<body>
<nav class="nav">
  <span class="nav-logo">geodesic</span>
  <a href="__PREFIX___report.html" class="nav-back">← Back to report</a>
  <span class="nav-prefix">How it works · __PREFIX__</span>
</nav>
<div id="algorithm">
<aside class="alg-panel">
  <div class="alg-panel-hdr">
    <div class="alg-logo">geodesic</div>
    <div class="alg-logo-sub">How it works</div>
    <a class="alg-doc-link" href="https://github.com/genomewalker/geodesic/blob/main/wiki/ALGORITHM.md" target="_blank">Full algorithm documentation &#8599;</a>
  </div>
  <div class="alg-steps" id="alg-steps">
    <div class="alg-step active" data-step="0">
      <div class="alg-step-num">01 · Fingerprinting</div>
      <div class="alg-step-title">Each genome gets a unique fingerprint</div>
      <div class="alg-step-body">
        Two independent OPH signatures (k=21, m=10,000 bins, seeds 42 and 1337). Averaging the two Jaccard estimates halves variance. Each dot on the sphere is a genome.
      </div>
    </div>
    <div class="alg-step" data-step="1">
      <div class="alg-step-num">02 · Projection</div>
      <div class="alg-step-title">Placing every genome on a sphere</div>
      <div class="alg-step-body">
        Nyström spectral embedding via ~512 stratified anchors. Regularised with Laplacian normalisation and Tikhonov loading. The angle between two points encodes their genetic distance.
      </div>
    </div>
    <div class="alg-step" data-step="2">
      <div class="alg-step-num">03 · Indexing</div>
      <div class="alg-step-title">Finding every genome's closest relatives</div>
      <div class="alg-step-body">
        HNSW index for sub-linear kNN search. Isolation score = mean angular distance to k=10 nearest neighbours. The most isolated genome becomes the first representative.
      </div>
    </div>
    <div class="alg-step" data-step="3">
      <div class="alg-step-num">04 · Selection</div>
      <div class="alg-step-title">Choosing representatives to cover all diversity</div>
      <div class="alg-step-body">
        Quality-weighted farthest-point sampling: greedy &theta;-cover that adds the genome farthest from any current representative. Stops when every genome is within the ANI threshold of some representative. Colored zones show coverage.
      </div>
    </div>
    <div class="alg-step" data-step="4">
      <div class="alg-step-num">05 · Refinement</div>
      <div class="alg-step-title">Spreading representatives evenly</div>
      <div class="alg-step-body">
        Union-Find merge collapses over-proximate representatives. Borderline non-reps are re-checked with exact dual-sketch OPH Jaccard. Press Run Thomson to simulate convergence.
      </div>
    </div>
  </div>
  <div class="alg-controls">
    <div class="alg-ctrl-row">
      <div class="alg-step-nav">
        <button class="alg-btn" id="alg-btn-prev">←</button>
        <span class="alg-step-lbl" id="alg-step-indicator">1 / 5</span>
        <button class="alg-btn" id="alg-btn-next">→</button>
      </div>
      <button class="alg-btn alg-btn-tour" id="alg-btn-tour">▶ Tour</button>
    </div>
    <div class="alg-ctrl-row">
      <button class="alg-btn primary" id="alg-btn-action" style="flex:1;display:none">▶ Run</button>
      <button class="alg-btn" id="alg-btn-autoplay" style="display:none">Step</button>
      <button class="alg-btn" id="alg-btn-thomson" style="flex:1;display:none">▶ Run Thomson</button>
      <button class="alg-btn" id="alg-btn-reset">↺</button>
    </div>
    <div class="alg-progress" id="alg-progress" style="display:none">
      <div class="alg-progress-bar" id="alg-progress-bar" style="width:0%"></div>
      <span class="alg-progress-label" id="alg-progress-label">0 / 80 covered</span>
    </div>
    <div class="alg-info-grid">
      <div class="alg-info-cell">
        <div class="alg-info-key">Representatives</div>
        <div class="alg-info-val" id="alg-info-reps">0</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Genomes covered</div>
        <div class="alg-info-val" id="alg-info-cov">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Largest gap</div>
        <div class="alg-info-val" id="alg-info-radius">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Status</div>
        <div class="alg-info-val" id="alg-info-energy">—</div>
      </div>
    </div>
  </div>
</aside>
<div class="alg-canvas-wrap">
  <canvas id="sphere-canvas"></canvas>
  <div class="alg-canvas-lbl" id="alg-canvas-label">UNIT SPHERE S²⁵⁵</div>
  <div class="alg-legend">
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#7090b8"></div>
      <span>Genome</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#15803d"></div>
      <span>Representative</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:rgba(21,128,61,.4)"></div>
      <span>Coverage zone</span>
    </div>
  </div>
</div>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const N_GENOMES = 80;
const COVERAGE_THRESHOLD = 0.38;
const canvas = document.getElementById('sphere-canvas');
const wrap = canvas.parentElement;

const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xf0f0ed, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(0, 0.5, 2.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 1.5;
controls.maxDistance = 6;

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(wrap);

scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(0.998, 32, 24),
  new THREE.MeshBasicMaterial({color:0xe8e8e4, transparent:true, opacity:0.5})
));
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 36),
  new THREE.MeshBasicMaterial({color:0xc8c8c4, wireframe:true, transparent:true, opacity:0.25})
));

function randOnSphere() {
  let x, y, s;
  do { x = Math.random()*2-1; y = Math.random()*2-1; s = x*x+y*y; } while (s >= 1);
  const sq = Math.sqrt(1-s);
  return new THREE.Vector3(2*x*sq, 2*y*sq, 1-2*s).normalize();
}

const centers = Array.from({length:8}, randOnSphere);
const genomePos = Array.from({length:N_GENOMES}, (_, i) => {
  if (Math.random() < 0.65) {
    const c = centers[i % 8];
    return c.clone().lerp(randOnSphere(), 0.18 + Math.random()*0.25).normalize();
  }
  return randOnSphere();
});

const PALETTE = [
  [0.15,0.50,0.24],[0.15,0.38,0.75],[0.75,0.15,0.15],
  [0.70,0.45,0.05],[0.50,0.10,0.75],[0.05,0.55,0.65],
  [0.75,0.30,0.05],[0.60,0.15,0.45],[0.25,0.60,0.10],
  [0.10,0.30,0.60],[0.65,0.10,0.20],[0.40,0.50,0.05],
];
const DEFAULT_COL = [0.44,0.56,0.72];

function paletteHex(ri) {
  const c = PALETTE[ri % PALETTE.length];
  return (Math.round(c[0]*255)<<16)|(Math.round(c[1]*255)<<8)|Math.round(c[2]*255);
}

const genomeGeo = new THREE.BufferGeometry();
const gPos = new Float32Array(N_GENOMES*3);
const gCol = new Float32Array(N_GENOMES*3);
genomePos.forEach((p,i) => {
  gPos[i*3]=p.x; gPos[i*3+1]=p.y; gPos[i*3+2]=p.z;
  gCol[i*3]=DEFAULT_COL[0]; gCol[i*3+1]=DEFAULT_COL[1]; gCol[i*3+2]=DEFAULT_COL[2];
});
genomeGeo.setAttribute('position', new THREE.BufferAttribute(gPos,3));
genomeGeo.setAttribute('color',    new THREE.BufferAttribute(gCol,3));
const genomePoints = new THREE.Points(genomeGeo,
  new THREE.PointsMaterial({size:0.035, sizeAttenuation:true, transparent:true, opacity:0.9, vertexColors:true})
);
scene.add(genomePoints);

const repGroup   = new THREE.Group(); scene.add(repGroup);
const flashGroup = new THREE.Group(); scene.add(flashGroup);
const lineGroup  = new THREE.Group(); scene.add(lineGroup);
const circleGroup= new THREE.Group(); scene.add(circleGroup);

const selected = [];
const repPositions = [];
let thomsonActive = false;
let autoplayTimer = null;
let currentStep = 0;
let thomsonEnergy = 0;
let coverageDone = false;
let tourActive = false;
const tourTimeouts = [];
let tourPoll = null;

function countCovered() {
  if (repPositions.length === 0) return 0;
  let n = 0;
  genomePos.forEach((gp, i) => {
    let minD = Infinity;
    repPositions.forEach(rp => { const d = gp.distanceTo(rp); if (d < minD) minD = d; });
    if (minD < COVERAGE_THRESHOLD) n++;
  });
  return n;
}

function updateVoronoiColors() {
  const col = genomeGeo.attributes.color.array;
  if (repPositions.length === 0) {
    for (let i = 0; i < N_GENOMES; i++) {
      col[i*3]=DEFAULT_COL[0]; col[i*3+1]=DEFAULT_COL[1]; col[i*3+2]=DEFAULT_COL[2];
    }
  } else {
    for (let i = 0; i < N_GENOMES; i++) {
      let ni=0, nd=genomePos[i].distanceTo(repPositions[0]);
      for (let r=1; r<repPositions.length; r++) {
        const d=genomePos[i].distanceTo(repPositions[r]);
        if (d<nd){nd=d;ni=r;}
      }
      const c=PALETTE[ni%PALETTE.length];
      if (selected.includes(i)){
        col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
      } else if (nd < COVERAGE_THRESHOLD) {
        col[i*3]=c[0]*0.45+0.55; col[i*3+1]=c[1]*0.45+0.55; col[i*3+2]=c[2]*0.45+0.55;
      } else {
        // Uncovered: muted blue-grey
        col[i*3]=0.44; col[i*3+1]=0.50; col[i*3+2]=0.58;
      }
    }
  }
  genomeGeo.attributes.color.needsUpdate = true;
}

function fpsStep() {
  if (selected.length === 0) {
    const idx = Math.floor(Math.random()*N_GENOMES);
    selected.push(idx);
    repPositions.push(genomePos[idx].clone());
    animateNewRep(genomePos[idx]);
    updateScene();
    return;
  }
  let best=-1, bestDist=-1;
  for (let i=0; i<N_GENOMES; i++) {
    if (selected.includes(i)) continue;
    let minD=Infinity;
    for (const rp of repPositions) { const d=genomePos[i].distanceTo(rp); if (d<minD) minD=d; }
    if (minD>bestDist){bestDist=minD; best=i;}
  }
  if (best >= 0) {
    selected.push(best);
    repPositions.push(genomePos[best].clone());
    animateNewRep(genomePos[best]);
    updateScene();
    if (bestDist < COVERAGE_THRESHOLD && selected.length > 1) {
      coverageDone = true;
      if (autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
      document.getElementById('alg-btn-action').textContent = '✓ Done';
      document.getElementById('alg-btn-action').disabled = true;
      document.getElementById('alg-info-energy').textContent = 'complete';
    }
  }
}

function animateNewRep(pos) {
  const hex = paletteHex(repPositions.length-1);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.05,10,8),
    new THREE.MeshBasicMaterial({color:hex, transparent:true, opacity:1})
  );
  mesh.position.copy(pos);
  flashGroup.add(mesh);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.05,0.068,24),
    new THREE.MeshBasicMaterial({color:hex, transparent:true, opacity:0.8, side:THREE.DoubleSide})
  );
  ring.position.copy(pos);
  ring.lookAt(camera.position);
  flashGroup.add(ring);
  const t0=performance.now(), dur=900;
  (function tick(){
    const p=Math.min((performance.now()-t0)/dur,1);
    mesh.material.opacity=1-p;
    ring.material.opacity=(1-p)*0.9;
    ring.scale.setScalar(1+p*1.4);
    if(p<1) requestAnimationFrame(tick);
    else{flashGroup.remove(mesh);flashGroup.remove(ring);}
  })();
}

function updateCoverageCircles() {
  while (circleGroup.children.length) circleGroup.remove(circleGroup.children[0]);
  if (repPositions.length===0) return;
  repPositions.forEach((rp,ri) => {
    let maxD=0;
    genomePos.forEach((gp,i) => {
      if (selected.includes(i)) return;
      let ni=0,nd=gp.distanceTo(repPositions[0]);
      for(let r=1;r<repPositions.length;r++){const d=gp.distanceTo(repPositions[r]);if(d<nd){nd=d;ni=r;}}
      if(ni===ri && nd>maxD) maxD=nd;
    });
    if(maxD<0.02) return;
    const angR=2*Math.asin(Math.min(maxD/2,0.9999));
    const up=Math.abs(rp.y)<0.9?new THREE.Vector3(0,1,0):new THREE.Vector3(1,0,0);
    const u=new THREE.Vector3().crossVectors(rp,up).normalize();
    const v=new THREE.Vector3().crossVectors(rp,u).normalize();
    const pts=[];
    for(let t=0;t<=72;t++){
      const a=(t/72)*Math.PI*2;
      pts.push(rp.clone().multiplyScalar(Math.cos(angR))
        .addScaledVector(u,Math.sin(angR)*Math.cos(a))
        .addScaledVector(v,Math.sin(angR)*Math.sin(a)));
    }
    circleGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color:paletteHex(ri),transparent:true,opacity:0.55})
    ));
  });
}

function updateScene() {
  updateVoronoiColors();
  while(repGroup.children.length) repGroup.remove(repGroup.children[0]);
  repPositions.forEach((rp,ri) => {
    const hex=paletteHex(ri);
    const m=new THREE.Mesh(new THREE.SphereGeometry(0.040,14,10),new THREE.MeshBasicMaterial({color:hex}));
    m.position.copy(rp); repGroup.add(m);
    const h=new THREE.Mesh(new THREE.SphereGeometry(0.062,12,10),new THREE.MeshBasicMaterial({color:hex,transparent:true,opacity:0.12}));
    h.position.copy(rp); repGroup.add(h);
  });
  while(lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
  if(repPositions.length>0){
    genomePos.forEach((gp,i) => {
      if(selected.includes(i)) return;
      let ni=0,nearD=gp.distanceTo(repPositions[0]);
      repPositions.forEach((rp,ri)=>{const d=gp.distanceTo(rp);if(d<nearD){nearD=d;ni=ri;}});
      const geo=new THREE.BufferGeometry().setFromPoints([gp.clone().multiplyScalar(1.01),repPositions[ni].clone().multiplyScalar(1.01)]);
      lineGroup.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:paletteHex(ni),transparent:true,opacity:0.22})));
    });
  }
  updateCoverageCircles();
  updateInfo();
}

function updateInfo() {
  document.getElementById('alg-info-reps').textContent = selected.length;
  const cov = countCovered();
  const pct = Math.round(cov/N_GENOMES*100);
  document.getElementById('alg-info-cov').textContent = cov + ' / ' + N_GENOMES;
  // Progress bar
  const pb = document.getElementById('alg-progress-bar');
  const pl = document.getElementById('alg-progress-label');
  if (pb) { pb.style.width = pct + '%'; pl.textContent = cov + ' / ' + N_GENOMES + ' covered (' + pct + '%)'; }
  if (repPositions.length>0) {
    let maxMinD=0;
    genomePos.forEach((gp,i) => {
      if(selected.includes(i)) return;
      let minD=Infinity;
      repPositions.forEach(rp=>{const d=gp.distanceTo(rp);if(d<minD)minD=d;});
      if(minD>maxMinD) maxMinD=minD;
    });
    document.getElementById('alg-info-radius').textContent = (maxMinD*180/Math.PI).toFixed(1)+'°';
  } else {
    document.getElementById('alg-info-radius').textContent = '—';
  }
  if (thomsonActive) {
    document.getElementById('alg-info-energy').textContent = 'U='+thomsonEnergy.toFixed(0);
  }
}

function thomsonStep() {
  const n=repPositions.length;
  if(n<2) return;
  const forces=repPositions.map(()=>new THREE.Vector3());
  let energy=0;
  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      const diff=repPositions[i].clone().sub(repPositions[j]);
      const d2=diff.lengthSq();
      energy+=1/Math.sqrt(d2);
      const f=diff.divideScalar(d2*Math.sqrt(d2)+1e-8);
      forces[i].add(f); forces[j].sub(f);
    }
  }
  thomsonEnergy=energy;
  repPositions.forEach((rp,i)=>{rp.addScaledVector(forces[i],0.0008);rp.normalize();});
  let k=0;
  repGroup.children.forEach(m=>{m.position.copy(repPositions[Math.floor(k/2)]);k++;});
  updateInfo();
}

function reset() {
  selected.length=0; repPositions.length=0;
  thomsonActive=false; coverageDone=false;
  if(autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
  document.getElementById('alg-btn-action').textContent='▶ Run';
  document.getElementById('alg-btn-action').disabled=false;
  document.getElementById('alg-btn-thomson').textContent='▶ Run Thomson';
  document.getElementById('alg-info-energy').textContent='—';
  while(repGroup.children.length)   repGroup.remove(repGroup.children[0]);
  while(lineGroup.children.length)  lineGroup.remove(lineGroup.children[0]);
  while(flashGroup.children.length) flashGroup.remove(flashGroup.children[0]);
  while(circleGroup.children.length)circleGroup.remove(circleGroup.children[0]);
  updateVoronoiColors(); updateInfo();
}

function setStep(s) {
  currentStep=Math.max(0,Math.min(4,s));
  document.querySelectorAll('#algorithm .alg-step').forEach((el,i)=>el.classList.toggle('active',i===currentStep));
  document.getElementById('alg-step-indicator').textContent=(currentStep+1)+' / 5';
  const labels=['FINGERPRINTING','SPHERE PROJECTION','NEIGHBOR INDEX','REPRESENTATIVE SELECTION','THOMSON REFINEMENT'];
  document.getElementById('alg-canvas-label').textContent=labels[currentStep];
  const inFPS=currentStep===3, inThomson=currentStep===4;
  const btnAction=document.getElementById('alg-btn-action');
  btnAction.style.display=inFPS?'':'none';
  if(inFPS&&!coverageDone){btnAction.textContent='▶ Run';btnAction.disabled=false;}
  document.getElementById('alg-btn-autoplay').style.display=inFPS?'':'none';
  document.getElementById('alg-btn-thomson').style.display=inThomson?'':'none';
  const prog=document.getElementById('alg-progress');
  if(prog) prog.style.display=inFPS?'':'none';
}

// ── Guided tour ───────────────────────────────────────────────────────────────
function tourAt(ms, fn) {
  const id=setTimeout(()=>{if(tourActive)fn();},ms);
  tourTimeouts.push(id);
}
function clearTourTimeouts() {
  tourTimeouts.forEach(clearTimeout);
  tourTimeouts.length=0;
  if(tourPoll){clearInterval(tourPoll);tourPoll=null;}
}
function startTour() {
  if(tourActive){stopTour();return;}
  reset();
  tourActive=true;
  document.getElementById('alg-btn-tour').textContent='⏹ Stop';
  setStep(0);
  tourAt(3200,()=>setStep(1));
  tourAt(6400,()=>setStep(2));
  tourAt(9600,()=>{
    setStep(3);
    if(!coverageDone){
      document.getElementById('alg-btn-action').textContent='⏸ Pause';
      autoplayTimer=setInterval(()=>{
        if(coverageDone||selected.length>=N_GENOMES){clearInterval(autoplayTimer);autoplayTimer=null;}
        else fpsStep();
      },150);
    }
  });
  tourPoll=setInterval(()=>{
    if(!tourActive){clearInterval(tourPoll);tourPoll=null;return;}
    if(coverageDone){
      clearInterval(tourPoll);tourPoll=null;
      tourAt(1200,()=>{
        setStep(4);
        thomsonActive=true;
        document.getElementById('alg-btn-thomson').textContent='⏸ Pause Thomson';
        document.getElementById('alg-info-energy').textContent='running…';
        tourAt(8000,()=>{
          thomsonActive=false;updateScene();
          document.getElementById('alg-btn-thomson').textContent='▶ Run Thomson';
          stopTour();
        });
      });
    }
  },300);
}
function stopTour() {
  tourActive=false;clearTourTimeouts();
  if(autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
  document.getElementById('alg-btn-tour').textContent='▶ Tour';
}

// ── Event handlers ────────────────────────────────────────────────────────────
document.getElementById('alg-btn-prev').onclick=()=>setStep(currentStep-1);
document.getElementById('alg-btn-next').onclick=()=>setStep(currentStep+1);
document.getElementById('alg-btn-reset').onclick=reset;
document.getElementById('alg-btn-tour').onclick=startTour;

document.getElementById('alg-btn-action').onclick=function(){
  if(coverageDone) return;
  if(autoplayTimer){
    clearInterval(autoplayTimer);autoplayTimer=null;
    this.textContent='▶ Run';
  } else {
    this.textContent='⏸ Pause';
    autoplayTimer=setInterval(()=>{
      if(coverageDone||selected.length>=N_GENOMES){clearInterval(autoplayTimer);autoplayTimer=null;return;}
      fpsStep();
    },200);
  }
};

document.getElementById('alg-btn-autoplay').onclick=function(){if(!coverageDone)fpsStep();};

document.getElementById('alg-btn-thomson').onclick=function(){
  thomsonActive=!thomsonActive;
  this.textContent=thomsonActive?'⏸ Pause Thomson':'▶ Run Thomson';
  if(!thomsonActive)updateScene();
};

document.querySelectorAll('#algorithm .alg-step').forEach((el,i)=>{
  el.addEventListener('click',()=>setStep(i));
});

setStep(0);

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  if(thomsonActive)thomsonStep();
  renderer.render(scene,camera);
}
animate();

</script>
</body>
</html>
)ALGO";


// ─────────────────────────────────────────────────────────────────────────────
// Main report HTML template  (__DATA__ is replaced with JSON)
// ─────────────────────────────────────────────────────────────────────────────
static const char* REPORT_HTML = R"GEODESIC(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>geodesic · __PREFIX__ · report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=Outfit:wght@300;400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#fff;--bg2:#f8f8f6;--bg3:#f0f0ed;
  --border:#e5e5e2;--border-dark:#ccccc8;
  --text:#0a0a08;--text2:#52524e;--text3:#a8a8a2;
  --ink:#0a0a08;--green:#15803d;--amber:#b45309;--rose:#9f1239;
  --green-d:rgba(21,128,61,.07);--ink-d:rgba(10,10,8,.05);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;font-size:14px;line-height:1.6;min-height:100vh}

/* ── Nav ── */
.nav{
  position:sticky;top:0;z-index:100;
  display:flex;align-items:center;gap:24px;
  padding:0 40px;height:48px;
  background:rgba(255,255,255,.97);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
}
.nav-logo{font-family:'Geist Mono',monospace;font-size:13px;font-weight:500;color:var(--text);letter-spacing:-.02em}
.nav-meta{font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3)}
.nav-links{margin-left:auto;display:flex;gap:2px}
.nav-links a{padding:5px 12px;border-radius:4px;text-decoration:none;font-size:13px;font-weight:500;color:var(--text2);transition:all .15s}
.nav-links a:hover{color:var(--text);background:var(--bg2)}
.nav-links a.active{color:var(--text);background:var(--bg3)}

/* ── Header ── */
.header{padding:64px 40px 40px;border-bottom:1px solid var(--border)}
.header-eyebrow{font-family:'Geist Mono',monospace;font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--text3);margin-bottom:14px}
.header-title{font-family:'Fraunces',serif;font-size:64px;font-weight:400;color:var(--text);line-height:1.0;letter-spacing:-.02em}
.header-title .accent{color:var(--green)}
.header-sub{margin-top:12px;font-family:'Fraunces',serif;font-size:20px;color:var(--text2);font-style:italic}
.header-meta{margin-top:24px;display:flex;gap:0;flex-wrap:wrap}
.meta-badge{padding:5px 14px;border:1px solid var(--border);font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3);margin-right:-1px}
.meta-badge:first-child{border-radius:4px 0 0 4px}
.meta-badge:last-child{border-radius:0 4px 4px 0}
.meta-badge span{color:var(--text);font-weight:500}

/* ── Stats strip ── */
.stats-strip{display:grid;grid-template-columns:repeat(5,1fr);border-bottom:1px solid var(--border)}
.stat-cell{padding:24px 28px;border-right:1px solid var(--border)}
.stat-cell:last-child{border-right:none}
.stat-val{font-family:'Fraunces',serif;font-size:38px;font-weight:400;color:var(--text);line-height:1}
.stat-val.c-ink{color:var(--text)}
.stat-val.c-teal{color:var(--green)}
.stat-val.c-amber{color:var(--amber)}
.stat-val.c-rose{color:var(--rose)}
.stat-lbl{margin-top:6px;font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text3)}
.stat-desc{margin-top:2px;font-family:'Outfit',sans-serif;font-size:11px;color:var(--text3)}

/* ── Section ── */
.section{padding:40px}
.section-title{font-family:'Fraunces',serif;font-size:24px;font-weight:400;color:var(--text);margin-bottom:24px;padding-bottom:12px;border-bottom:1px solid var(--border)}

/* ── Charts ── */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border)}
.chart-card{background:var(--bg);padding:24px}
.chart-label{font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);margin-bottom:16px}
.chart-card canvas{display:block;width:100%}

/* ── Explorer ── */
#explorer{border-top:1px solid var(--border)}
.explorer-toolbar{padding:20px 40px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border);background:var(--bg);position:sticky;top:48px;z-index:50}
.search-wrap{display:flex;align-items:center;gap:8px;border:1px solid var(--border-dark);border-radius:5px;padding:0 12px;background:white;transition:border-color .15s}
.search-wrap:focus-within{border-color:var(--text)}
.search-icon{font-size:13px;color:var(--text3)}
.search-input{border:none;outline:none;background:none;font-family:'Geist Mono',monospace;font-size:12px;color:var(--text);padding:8px 0;width:280px}
.search-input::placeholder{color:var(--text3)}
.count-label{font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3);margin-left:auto}

/* ── Table ── */
.tbl-container{padding:0 40px 32px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
thead th{padding:9px 12px;text-align:left;font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--text3);border-bottom:2px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none}
thead th:hover{color:var(--text)}
thead th.r{text-align:right}
td{padding:7px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--bg2)}
tr:nth-child(even) td{background:rgba(0,0,0,.012)}
td.tx{font-family:'Geist Mono',monospace;font-size:11px;max-width:400px}
td.nm{font-family:'Geist Mono',monospace;font-size:12px;text-align:right;color:var(--text2)}
td.nm.hi{color:var(--text);font-weight:500}
.chip{display:inline-block;padding:1px 5px;border-radius:3px;font-size:9.5px;font-weight:600;letter-spacing:.04em;margin-right:3px;vertical-align:middle}
.chip-p{background:var(--ink-d);color:var(--text2)}
.chip-g{background:var(--green-d);color:var(--green)}
.chip-s{color:var(--text)}
.mini-bar{display:inline-block;height:3px;border-radius:2px;opacity:.7;vertical-align:middle;margin-right:6px}
.method-tag{display:inline-block;padding:1px 6px;border-radius:2px;font-family:'Geist Mono',monospace;font-size:9px;font-weight:500;letter-spacing:.04em}
.mt-singleton{background:var(--bg3);color:var(--text3)}
.mt-tiny{background:var(--green-d);color:var(--green);border:1px solid rgba(21,128,61,.2)}
.mt-geodesic{background:var(--ink-d);color:var(--text);border:1px solid rgba(10,10,8,.12)}

/* ── Pager ── */
.pager{padding:12px 40px 24px;display:flex;align-items:center;justify-content:center;gap:3px}
.pg-btn{background:none;border:1px solid var(--border);border-radius:4px;padding:4px 10px;font-family:'Outfit',sans-serif;font-size:12px;color:var(--text2);cursor:pointer;transition:all .12s}
.pg-btn:hover{border-color:var(--text);color:var(--text)}
.pg-btn.on{background:var(--text);border-color:var(--text);color:white}
.pg-btn:disabled{opacity:.3;cursor:not-allowed}
.pg-sep{color:var(--text3);padding:0 4px;font-size:12px}

/* ── Footer ── */
footer{padding:28px 40px;border-top:1px solid var(--border);display:flex;align-items:center;gap:16px;background:var(--bg2)}
.footer-logo{font-family:'Geist Mono',monospace;font-size:12px;color:var(--text);font-weight:500}
.footer-desc{font-family:'Outfit',sans-serif;font-size:12px;color:var(--text3)}
.footer-links{margin-left:auto;display:flex;gap:20px}
.footer-links a{font-family:'Outfit',sans-serif;font-size:12px;color:var(--text3);text-decoration:none}
.footer-links a:hover{color:var(--text)}

/* ── Utility ── */
.muted{color:var(--text3)!important}
.loading-msg{padding:40px;text-align:center;color:var(--text3);font-size:13px}

/* ── Algorithm CTA card ── */
.alg-cta{padding:32px;border:1px solid var(--border);border-radius:8px;display:flex;align-items:center;gap:32px;background:var(--bg2)}
.alg-cta-desc{flex:1;font-size:13px;color:var(--text2);line-height:1.7}
.alg-cta-btn{display:inline-block;padding:10px 20px;border:1px solid var(--green);border-radius:5px;font-size:13px;font-weight:600;color:var(--green);text-decoration:none;white-space:nowrap;transition:all .15s}
.alg-cta-btn:hover{background:var(--green);color:white}

/* ── Algorithm section ── */
#algorithm{flex:1;display:flex;overflow:hidden;background:var(--bg)}
#algorithm .alg-panel{
  width:320px;min-width:320px;background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;z-index:10;
}
#algorithm .alg-panel-hdr{padding:24px 24px 16px;border-bottom:1px solid var(--border);flex-shrink:0}
#algorithm .alg-logo{font-family:'Geist Mono',monospace;font-size:15px;font-weight:500;color:var(--green);letter-spacing:-.02em}
#algorithm .alg-logo-sub{margin-top:4px;font-size:11px;color:var(--text3);letter-spacing:.08em;text-transform:uppercase}
#algorithm .alg-steps{flex:1;overflow-y:auto;padding:16px 0}
#algorithm .alg-steps::-webkit-scrollbar{width:4px}
#algorithm .alg-steps::-webkit-scrollbar-thumb{background:var(--border-dark);border-radius:2px}
#algorithm .alg-step{padding:14px 24px;border-left:2px solid transparent;cursor:pointer;transition:all .15s;opacity:.5}
#algorithm .alg-step.active{border-left-color:var(--green);opacity:1;background:var(--green-d)}
#algorithm .alg-step:hover{opacity:.8}
#algorithm .alg-step-num{font-family:'Geist Mono',monospace;font-size:10px;color:var(--green);letter-spacing:.12em;margin-bottom:4px}
#algorithm .alg-step-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:6px}
#algorithm .alg-step-body{font-size:12px;color:var(--text2);line-height:1.6;display:none}
#algorithm .alg-step.active .alg-step-body{display:block}
#algorithm .alg-doc-link{display:block;margin-top:10px;font-size:11px;color:var(--text3);text-decoration:none;letter-spacing:.04em;transition:color .15s}
#algorithm .alg-doc-link:hover{color:var(--green)}
#algorithm .alg-controls{padding:16px 24px;border-top:1px solid var(--border);flex-shrink:0}
#algorithm .alg-ctrl-row{display:flex;gap:8px;margin-bottom:8px;align-items:center}
#algorithm .alg-ctrl-row:last-child{margin-bottom:0}
#algorithm .alg-btn{
  background:var(--bg);border:1px solid var(--border);border-radius:5px;
  padding:7px 14px;color:var(--text2);font-family:'Outfit',sans-serif;
  font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;white-space:nowrap;
}
#algorithm .alg-btn:hover{color:var(--text);border-color:var(--border-dark)}
#algorithm .alg-btn.primary{color:var(--green);border-color:rgba(21,128,61,.3);background:var(--green-d)}
#algorithm .alg-btn.primary:hover{background:rgba(21,128,61,.12)}
#algorithm .alg-btn:disabled{opacity:.35;cursor:not-allowed}
#algorithm .alg-step-nav{display:flex;align-items:center;gap:8px;flex:1}
#algorithm .alg-step-lbl{flex:1;text-align:center;font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3)}
#algorithm .alg-info-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-top:8px;
}
#algorithm .alg-info-cell{background:var(--bg);padding:10px 12px}
#algorithm .alg-info-key{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:2px}
#algorithm .alg-info-val{font-family:'Geist Mono',monospace;font-size:13px;font-weight:500;color:var(--text)}
#algorithm .alg-canvas-wrap{flex:1;position:relative;overflow:hidden;background:var(--bg3)}
#algorithm #sphere-canvas{width:100%;height:100%;display:block}
#algorithm .alg-canvas-lbl{
  position:absolute;top:20px;right:24px;
  font-family:'Geist Mono',monospace;font-size:11px;color:var(--text3);
  letter-spacing:.08em;text-transform:uppercase;
}
#algorithm .alg-legend{position:absolute;bottom:24px;right:24px;display:flex;flex-direction:column;gap:6px}
#algorithm .alg-legend-item{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text2)}
#algorithm .alg-legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

#algorithm .alg-insight{
  background:rgba(21,128,61,.05);border-left:2px solid var(--green);
  padding:6px 10px;margin:8px 0;font-size:11.5px;color:var(--text2);
  line-height:1.6;font-style:italic;
}
#algorithm .alg-btn.alg-btn-tour{color:var(--amber);border-color:rgba(180,83,9,.3)}
#algorithm .alg-btn.alg-btn-tour:hover{background:rgba(180,83,9,.07)}
#algorithm .alg-progress{
  position:relative;height:26px;background:var(--bg3);
  border:1px solid var(--border);border-radius:5px;
  overflow:hidden;margin-bottom:8px;display:flex;align-items:center;padding:0 10px;
}
#algorithm .alg-progress-bar{
  position:absolute;left:0;top:0;bottom:0;
  background:var(--green-d);border-right:2px solid var(--green);
  transition:width .4s ease;
}
#algorithm .alg-progress-label{
  position:relative;font-family:'Geist Mono',monospace;
  font-size:11px;color:var(--green);font-weight:500;
}
/* ── Help tooltip ── */
.help-tip{
  display:inline-flex;align-items:center;justify-content:center;
  width:14px;height:14px;border-radius:50%;
  background:var(--bg3);border:1px solid var(--border-dark);
  font-size:9px;color:var(--text3);cursor:help;
  margin-left:5px;vertical-align:middle;position:relative;flex-shrink:0;
}
.help-tip::after{
  content:attr(data-tip);
  position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);
  background:var(--ink);color:#fff;padding:9px 13px;border-radius:6px;
  font-size:11.5px;line-height:1.55;width:260px;white-space:normal;
  z-index:300;pointer-events:none;font-weight:400;font-family:'Outfit',sans-serif;
  opacity:0;transition:opacity .15s;
}
.help-tip::before{
  content:'';position:absolute;bottom:calc(100% + 2px);left:50%;transform:translateX(-50%);
  border:5px solid transparent;border-top-color:var(--ink);
  z-index:300;pointer-events:none;opacity:0;transition:opacity .15s;
}
.help-tip:hover::after,.help-tip:hover::before{opacity:1}

/* ── Algorithm modal ── */
.alg-modal{
  position:fixed;inset:0;z-index:500;
  background:var(--bg);display:flex;flex-direction:column;
  opacity:0;pointer-events:none;transition:opacity .18s ease;
}
.alg-modal.open{opacity:1;pointer-events:all}
.alg-modal-hdr{
  display:flex;align-items:center;gap:12px;
  padding:0 24px;height:48px;
  background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;
}
.alg-modal-title{font-family:'Geist Mono',monospace;font-size:13px;font-weight:500;color:var(--text)}
.alg-modal-sub{font-family:'Outfit',sans-serif;font-size:11px;color:var(--text3)}
.alg-modal-close{
  margin-left:auto;background:none;border:1px solid var(--border);border-radius:4px;
  padding:5px 14px;font-family:'Outfit',sans-serif;font-size:12px;font-weight:500;
  color:var(--text2);cursor:pointer;transition:all .15s;
}
.alg-modal-close:hover{color:var(--text);border-color:var(--border-dark)}
</style>
</head>
<body>

<nav class="nav">
  <span class="nav-logo">geodesic</span>
  <span class="nav-meta">__PREFIX__ · __TIMESTAMP__</span>
  <div class="nav-links">
    <a href="#overview">Overview</a>
    <a href="#explorer">Explorer</a>
    <a href="#" id="nav-alg-btn">Algorithm</a>
  </div>
</nav>

<section id="overview">
  <div class="header">
    <div class="header-eyebrow">geodesic run report</div>
    <div class="header-title"><span class="accent counter" data-n="__N_GENOMES__">0</span><br>genomes processed</div>
    <div class="header-sub">
      <span class="counter" data-n="__N_TAXA__">0</span> species ·
      <span class="counter" data-n="__N_REPS__">0</span> representatives selected
    </div>
    <div class="header-meta">
      <div class="meta-badge">prefix <span>__PREFIX__</span></div>
      <div class="meta-badge">run <span>__TIMESTAMP__</span></div>
      <div class="meta-badge">runtime <span>__RUNTIME__</span></div>
      <div class="meta-badge">singletons <span>__N_SINGLETONS__</span></div>
    </div>
  </div>

  <div class="stats-strip">
    <div class="stat-cell">
      <div class="stat-val c-ink"><span class="counter" data-n="__N_REPS__" data-fmt="int">0</span></div>
      <div class="stat-lbl">Representatives</div>
      <div class="stat-desc">selected across all species</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val c-teal"><span id="m-cov">—</span></div>
      <div class="stat-lbl">Mean Coverage ANI <span class="help-tip" data-tip="Average ANI between each genome and its nearest representative. Higher = representatives cover the diversity more tightly. 99% means no genome is more than 1% diverged from some representative.">?</span></div>
      <div class="stat-desc">genome to nearest representative</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val c-amber"><span id="m-div">—</span></div>
      <div class="stat-lbl">Mean Diversity ANI <span class="help-tip" data-tip="Average pairwise ANI among the selected representatives. Lower = representatives are more genetically spread out. High diversity ANI means reps cluster together and coverage may overlap.">?</span></div>
      <div class="stat-desc">pairwise among representatives</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val"><span id="m-red">—</span></div>
      <div class="stat-lbl">Mean Reduction</div>
      <div class="stat-desc">fraction of genomes retained</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val c-rose"><span class="counter" data-n="__N_FAILED__">0</span></div>
      <div class="stat-lbl">Failed Taxa</div>
      <div class="stat-desc">processing errors</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Distribution Overview</div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-label">Taxon Size Distribution</div>
        <canvas id="chart-size" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label">Coverage ANI Distribution</div>
        <canvas id="chart-cov" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label">Reduction Ratio Distribution</div>
        <canvas id="chart-rr" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-label">Processing Method</div>
        <canvas id="chart-method" height="160"></canvas>
      </div>
    </div>
  </div>
</section>

<section id="explorer">
  <div class="explorer-toolbar">
    <div class="search-wrap">
      <span class="search-icon">⌕</span>
      <input class="search-input" id="q" type="text" placeholder="search taxonomy…" autocomplete="off" spellcheck="false">
    </div>
    <span class="count-label" id="cnt">— taxa</span>
  </div>
  <div class="tbl-container">
    <table>
      <thead>
        <tr>
          <th data-col="tx">Species</th>
          <th data-col="ng" class="r">Genomes</th>
          <th data-col="nr" class="r">Reps</th>
          <th data-col="rr" class="r">Reduction</th>
          <th data-col="cm" class="r">Coverage ANI <span class="help-tip" data-tip="Mean ANI from each genome to its nearest representative. Measures how well the species diversity is covered.">?</span></th>
          <th data-col="dm" class="r">Diversity ANI <span class="help-tip" data-tip="Mean pairwise ANI among the representatives. Lower = more genetically diverse representative set.">?</span></th>
          <th data-col="mt">Method</th>
          <th data-col="rt" class="r">Runtime</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr><td colspan="8" class="loading-msg">Loading…</td></tr>
      </tbody>
    </table>
  </div>
  <div class="pager" id="pager"></div>
</section>


<div id="alg-modal" class="alg-modal">
<div class="alg-modal-hdr">
  <span class="alg-modal-title">geodesic</span>
  <span class="alg-modal-sub">Algorithm · How it works</span>
  <button class="alg-modal-close" id="alg-modal-close">✕ Close</button>
</div>
<div id="algorithm">
<aside class="alg-panel">
  <div class="alg-panel-hdr">
    <div class="alg-logo">geodesic</div>
    <div class="alg-logo-sub">How it works</div>
    <a class="alg-doc-link" href="https://github.com/genomewalker/geodesic/blob/main/wiki/ALGORITHM.md" target="_blank">Full algorithm documentation &#8599;</a>
  </div>
  <div class="alg-steps" id="alg-steps">
    <div class="alg-step active" data-step="0">
      <div class="alg-step-num">01 · Fingerprinting</div>
      <div class="alg-step-title">Each genome gets a unique fingerprint</div>
      <div class="alg-step-body">
        Two independent OPH signatures (k=21, m=10,000 bins, seeds 42 and 1337). Averaging the two Jaccard estimates halves variance. Each dot on the sphere is a genome.
      </div>
    </div>
    <div class="alg-step" data-step="1">
      <div class="alg-step-num">02 · Projection</div>
      <div class="alg-step-title">Placing every genome on a sphere</div>
      <div class="alg-step-body">
        Nyström spectral embedding via ~512 stratified anchors. Regularised with Laplacian normalisation and Tikhonov loading. The angle between two points encodes their genetic distance.
      </div>
    </div>
    <div class="alg-step" data-step="2">
      <div class="alg-step-num">03 · Indexing</div>
      <div class="alg-step-title">Finding every genome's closest relatives</div>
      <div class="alg-step-body">
        HNSW index for sub-linear kNN search. Isolation score = mean angular distance to k=10 nearest neighbours. The most isolated genome becomes the first representative.
      </div>
    </div>
    <div class="alg-step" data-step="3">
      <div class="alg-step-num">04 · Selection</div>
      <div class="alg-step-title">Choosing representatives to cover all diversity</div>
      <div class="alg-step-body">
        Quality-weighted farthest-point sampling: greedy &theta;-cover that adds the genome farthest from any current representative. Stops when every genome is within the ANI threshold of some representative. Colored zones show coverage.
      </div>
    </div>
    <div class="alg-step" data-step="4">
      <div class="alg-step-num">05 · Refinement</div>
      <div class="alg-step-title">Spreading representatives evenly</div>
      <div class="alg-step-body">
        Union-Find merge collapses over-proximate representatives. Borderline non-reps are re-checked with exact dual-sketch OPH Jaccard. Press Run Thomson to simulate convergence.
      </div>
    </div>
  </div>
  <div class="alg-controls">
    <div class="alg-ctrl-row">
      <div class="alg-step-nav">
        <button class="alg-btn" id="alg-btn-prev">←</button>
        <span class="alg-step-lbl" id="alg-step-indicator">1 / 5</span>
        <button class="alg-btn" id="alg-btn-next">→</button>
      </div>
      <button class="alg-btn alg-btn-tour" id="alg-btn-tour">▶ Tour</button>
    </div>
    <div class="alg-ctrl-row">
      <button class="alg-btn primary" id="alg-btn-action" style="flex:1;display:none">▶ Run</button>
      <button class="alg-btn" id="alg-btn-autoplay" style="display:none">Step</button>
      <button class="alg-btn" id="alg-btn-thomson" style="flex:1;display:none">▶ Run Thomson</button>
      <button class="alg-btn" id="alg-btn-reset">↺</button>
    </div>
    <div class="alg-progress" id="alg-progress" style="display:none">
      <div class="alg-progress-bar" id="alg-progress-bar" style="width:0%"></div>
      <span class="alg-progress-label" id="alg-progress-label">0 / 80 covered</span>
    </div>
    <div class="alg-info-grid">
      <div class="alg-info-cell">
        <div class="alg-info-key">Representatives</div>
        <div class="alg-info-val" id="alg-info-reps">0</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Genomes covered</div>
        <div class="alg-info-val" id="alg-info-cov">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Largest gap</div>
        <div class="alg-info-val" id="alg-info-radius">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Status</div>
        <div class="alg-info-val" id="alg-info-energy">—</div>
      </div>
    </div>
  </div>
</aside>
<div class="alg-canvas-wrap">
  <canvas id="sphere-canvas"></canvas>
  <div class="alg-canvas-lbl" id="alg-canvas-label">UNIT SPHERE S²⁵⁵</div>
  <div class="alg-legend">
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#7090b8"></div>
      <span>Genome</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#15803d"></div>
      <span>Representative</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:rgba(21,128,61,.4)"></div>
      <span>Coverage zone</span>
    </div>
  </div>
</div>
</div>
</div>
<footer>
  <div class="footer-logo">geodesic</div>
  <div class="footer-desc">spherical genome embeddings for diverse representative selection</div>
  <div class="footer-links">
    <a href="https://github.com/genomewalker/geodesic" target="_blank">GitHub ↗</a>
  </div>
</footer>

<script>
const D = __DATA__;

// ── Counters ──────────────────────────────────────────────────────────────
function animCounters() {
  const dur = 1500;
  const ease = t => 1 - Math.pow(1-t, 3);
  document.querySelectorAll('.counter').forEach(el => {
    const v = parseInt(el.dataset.n) || 0;
    const t0 = performance.now();
    const tick = () => {
      const p = Math.min((performance.now()-t0)/dur, 1);
      el.textContent = Math.floor(ease(p)*v).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = v.toLocaleString();
    };
    requestAnimationFrame(tick);
  });
}

// ── Inline metrics ─────────────────────────────────────────────────────────
function initMetrics() {
  const s = D.summary;
  document.getElementById('m-cov').textContent = s.mean_coverage ? s.mean_coverage.toFixed(2)+'%' : '—';
  document.getElementById('m-div').textContent = s.mean_diversity ? s.mean_diversity.toFixed(2)+'%' : '—';
  const rr = s.n_reps && s.n_genomes ? ((1 - s.n_reps/s.n_genomes)*100).toFixed(1)+'%' : '—';
  document.getElementById('m-red').textContent = rr;
}

// ── Chart drawing ──────────────────────────────────────────────────────────
function drawBars(canvas, labels, vals, color, opts) {
  opts = opts||{};
  const dpr = window.devicePixelRatio||1;
  const w = canvas.offsetWidth||canvas.parentElement.clientWidth, h = parseInt(canvas.getAttribute('height')||160);
  canvas.width = w*dpr; canvas.height = h*dpr;
  canvas.style.height = h+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {t:8, r:10, b:30, l:opts.logY ? 36 : 40};
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
  const maxV = Math.max(...vals, 1);
  const bw = cw / vals.length;

  // Grid
  ctx.strokeStyle = '#e5e5e2'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ch*(1 - i/4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+cw, y); ctx.stroke();
    const v = Math.round(maxV*i/4);
    ctx.fillStyle = '#a8a8a2'; ctx.font = '10px Geist Mono,monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v >= 10000 ? (v/1000).toFixed(0)+'k' : v >= 1000 ? (v/1000).toFixed(1)+'k' : v, pad.l-5, y+3.5);
  }

  vals.forEach((v, i) => {
    const bh = (v/maxV)*ch;
    const x = pad.l + i*bw, y = pad.t + ch - bh;
    ctx.fillStyle = color;
    if (ctx.roundRect) {
      ctx.beginPath(); ctx.roundRect(x+2, y, bw-4, bh, [3,3,0,0]); ctx.fill();
    } else {
      ctx.fillRect(x+2, y, bw-4, bh);
    }
    const step = Math.ceil(vals.length / 10);
    if (i % step === 0 || i === vals.length-1) {
      ctx.fillStyle = '#a8a8a2'; ctx.font = '9px Geist Mono,monospace';
      ctx.textAlign = 'center';
      const lbl = labels[i];
      ctx.fillText(lbl.length > 6 ? lbl.slice(0,5)+'…' : lbl, x+bw/2, pad.t+ch+18);
    }
  });
}

function histCounts(vals, edges) {
  const n = edges.length-1;
  const counts = Array(n).fill(0);
  const labels = Array.from({length:n}, (_,i) => edges[i].toFixed(1));
  vals.forEach(v => {
    for (let i = 0; i < n; i++) {
      if (v >= edges[i] && v < edges[i+1]) { counts[i]++; break; }
      if (i === n-1 && v <= edges[i+1]) counts[i]++;
    }
  });
  return {labels, counts};
}

function initCharts() {
  const t = D.taxa;

  // 1. Taxon size bins
  const sizeBuckets = {
    '1':0,'2':0,'3-5':0,'6-10':0,'11-20':0,'21-50':0,
    '51-100':0,'101-500':0,'501-1k':0,'1k+':0
  };
  t.ng.forEach(n => {
    const k = n===1?'1':n===2?'2':n<=5?'3-5':n<=10?'6-10':n<=20?'11-20':n<=50?'21-50':
              n<=100?'51-100':n<=500?'101-500':n<=1000?'501-1k':'1k+';
    sizeBuckets[k]++;
  });
  const sKeys = Object.keys(sizeBuckets);
  drawBars(document.getElementById('chart-size'), sKeys, sKeys.map(k=>sizeBuckets[k]), '#0a0a08');

  // 2. Coverage ANI (only non-singleton taxa)
  const covVals = t.cm.filter((_,i) => t.mt[i]!=='singleton' && t.cm[i]>0);
  const covEdges = [90,91,92,93,94,95,96,97,98,99,99.5,100,100.01];
  const cov = histCounts(covVals, covEdges);
  drawBars(document.getElementById('chart-cov'), cov.labels, cov.counts, '#15803d');

  // 3. Reduction ratio (non-singleton taxa with >1 genome)
  const rrVals = t.rr.filter((_,i) => t.ng[i]>1);
  const rrEdges = Array.from({length:21},(_,i)=>i*0.05);
  const rr = histCounts(rrVals, rrEdges);
  drawBars(document.getElementById('chart-rr'), rr.labels, rr.counts, '#854d0e');

  // 4. Method breakdown
  const mtCounts = {};
  t.mt.forEach(m => { mtCounts[m]=(mtCounts[m]||0)+1; });
  const mtKeys = Object.keys(mtCounts).sort((a,b)=>mtCounts[b]-mtCounts[a]);
  drawBars(document.getElementById('chart-method'), mtKeys, mtKeys.map(k=>mtCounts[k]), '#881337');
}

// ── Table ──────────────────────────────────────────────────────────────────
const PG = 50;
let rows = [], filtered = [], page = 0, sortCol = 'ng', sortAsc = false;

function buildRows() {
  const t = D.taxa;
  rows = t.tx.map((tx,i) => ({tx,ng:t.ng[i],nr:t.nr[i],rr:t.rr[i],cm:t.cm[i],dm:t.dm[i],mt:t.mt[i],rt:t.rt[i]}));
}

function parseSpecies(tx) {
  const parts = tx.split(';');
  const get = k => { const p = parts.find(x=>x.startsWith(k+'__')); return p ? p.slice(k.length+2) : ''; };
  return { p:get('p'), g:get('g'), s:get('s') };
}

function methodTag(m) {
  if (m==='singleton') return '<span class="method-tag mt-singleton">singleton</span>';
  if (m.includes('tiny')) return '<span class="method-tag mt-tiny">tiny</span>';
  return '<span class="method-tag mt-geodesic">geodesic</span>';
}

function fmtRt(rt) {
  if (!rt || rt<=0) return '<span class="muted">—</span>';
  return rt < 60 ? rt.toFixed(1)+'s' : (rt/60).toFixed(1)+'m';
}

function renderTable() {
  const start = page*PG;
  const slice = filtered.slice(start, start+PG);
  document.getElementById('tbody').innerHTML = slice.map(r => {
    const {p,g,s} = parseSpecies(r.tx);
    const pChip = p ? `<span class="chip chip-p">${p.slice(0,14)}</span>` : '';
    const gChip = g ? `<span class="chip chip-g">${g}</span>` : '';
    const sName = s || g || r.tx.split(';').pop();
    const barW = Math.max(1, Math.round(r.rr*60));
    const rrPct = r.ng>1 ? `<span class="mini-bar" style="width:${barW}px;background:var(--amber)"></span>${(r.rr*100).toFixed(1)}%` : '<span class="muted">—</span>';
    const cov = r.cm>0 ? r.cm.toFixed(2)+'%' : '<span class="muted">—</span>';
    const div = r.dm>0 ? r.dm.toFixed(2)+'%' : '<span class="muted">—</span>';
    return `<tr>
      <td class="tx">${pChip}${gChip}<span class="chip chip-s">${sName}</span></td>
      <td class="nm hi">${r.ng.toLocaleString()}</td>
      <td class="nm">${r.nr.toLocaleString()}</td>
      <td class="nm">${rrPct}</td>
      <td class="nm">${cov}</td>
      <td class="nm">${div}</td>
      <td>${methodTag(r.mt)}</td>
      <td class="nm muted">${fmtRt(r.rt)}</td>
    </tr>`;
  }).join('');
}

function renderPager() {
  const n = Math.ceil(filtered.length/PG);
  if (n<=1) { document.getElementById('pager').innerHTML=''; return; }
  const visible = new Set([0,n-1,...Array.from({length:5},(_,i)=>page-2+i).filter(x=>x>=0&&x<n)]);
  let html = `<button class="pg-btn" onclick="go(${Math.max(0,page-1)})" ${page?'':'disabled'}>‹</button>`;
  let prev=-1;
  [...visible].sort((a,b)=>a-b).forEach(p => {
    if (prev>=0 && p>prev+1) html += '<span class="pg-sep">…</span>';
    html += `<button class="pg-btn${p===page?' on':''}" onclick="go(${p})">${p+1}</button>`;
    prev=p;
  });
  html += `<button class="pg-btn" onclick="go(${Math.min(n-1,page+1)})" ${page===n-1?'disabled':''}>›</button>`;
  document.getElementById('pager').innerHTML = html;
}

window.go = p => {
  page=p; renderTable(); renderPager();
  document.getElementById('explorer').scrollIntoView({behavior:'smooth',block:'start'});
};

function applySort() {
  filtered.sort((a,b) => {
    const va=a[sortCol], vb=b[sortCol];
    const c = typeof va==='string' ? va.localeCompare(vb) : (va||0)-(vb||0);
    return sortAsc ? c : -c;
  });
}

function applyFilter(q) {
  const lq = q.toLowerCase();
  filtered = lq ? rows.filter(r=>r.tx.toLowerCase().includes(lq)) : [...rows];
  document.getElementById('cnt').textContent = filtered.length.toLocaleString()+' taxa';
  page=0; applySort(); renderTable(); renderPager();
}

// ── Nav highlight ──────────────────────────────────────────────────────────
function initNav() {
  const links = document.querySelectorAll('.nav-links a');
  const secs = document.querySelectorAll('section[id]');
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(a => a.classList.toggle('active', a.getAttribute('href')==='#'+e.target.id));
      }
    });
  }, {threshold:0.3});
  secs.forEach(s => io.observe(s));
}

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  initMetrics();
  animCounters();
  initNav();
  buildRows();

  // Column sort
  document.querySelectorAll('thead th[data-col]').forEach(th => {
    th.style.cursor='pointer';
    th.addEventListener('click', () => {
      const c = th.dataset.col;
      if (sortCol===c) sortAsc=!sortAsc; else { sortCol=c; sortAsc=c==='tx'; }
      document.querySelectorAll('thead th').forEach(h => {
        h.textContent = h.textContent.replace(/ [↑↓]$/,'');
        if (h.dataset.col===sortCol) h.textContent += sortAsc?' ↑':' ↓';
      });
      applyFilter(document.getElementById('q').value);
    });
  });

  // Search
  let timer;
  document.getElementById('q').addEventListener('input', e => {
    clearTimeout(timer); timer = setTimeout(()=>applyFilter(e.target.value), 180);
  });

  applyFilter('');

  // Charts (defer to allow layout)
  setTimeout(initCharts, 150);
});

// ── Algorithm modal ───────────────────────────────────────────────────────────
const algModal = document.getElementById('alg-modal');
function openAlgModal()  { algModal.classList.add('open'); document.body.style.overflow='hidden'; }
function closeAlgModal() { algModal.classList.remove('open'); document.body.style.overflow=''; }
document.getElementById('nav-alg-btn').addEventListener('click', e => { e.preventDefault(); openAlgModal(); });
document.getElementById('alg-modal-close').addEventListener('click', closeAlgModal);
document.addEventListener('keydown', e => { if (e.key === 'Escape' && algModal.classList.contains('open')) closeAlgModal(); });
algModal.addEventListener('click', e => { if (e.target === algModal) closeAlgModal(); });

</script>
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const N_GENOMES = 80;
const COVERAGE_THRESHOLD = 0.38;
const canvas = document.getElementById('sphere-canvas');
const wrap = canvas.parentElement;

const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0xf0f0ed, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(0, 0.5, 2.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 1.5;
controls.maxDistance = 6;

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(wrap);

scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(0.998, 32, 24),
  new THREE.MeshBasicMaterial({color:0xe8e8e4, transparent:true, opacity:0.5})
));
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 36),
  new THREE.MeshBasicMaterial({color:0xc8c8c4, wireframe:true, transparent:true, opacity:0.25})
));

function randOnSphere() {
  let x, y, s;
  do { x = Math.random()*2-1; y = Math.random()*2-1; s = x*x+y*y; } while (s >= 1);
  const sq = Math.sqrt(1-s);
  return new THREE.Vector3(2*x*sq, 2*y*sq, 1-2*s).normalize();
}

const centers = Array.from({length:8}, randOnSphere);
const genomePos = Array.from({length:N_GENOMES}, (_, i) => {
  if (Math.random() < 0.65) {
    const c = centers[i % 8];
    return c.clone().lerp(randOnSphere(), 0.18 + Math.random()*0.25).normalize();
  }
  return randOnSphere();
});

const PALETTE = [
  [0.15,0.50,0.24],[0.15,0.38,0.75],[0.75,0.15,0.15],
  [0.70,0.45,0.05],[0.50,0.10,0.75],[0.05,0.55,0.65],
  [0.75,0.30,0.05],[0.60,0.15,0.45],[0.25,0.60,0.10],
  [0.10,0.30,0.60],[0.65,0.10,0.20],[0.40,0.50,0.05],
];
const DEFAULT_COL = [0.44,0.56,0.72];

function paletteHex(ri) {
  const c = PALETTE[ri % PALETTE.length];
  return (Math.round(c[0]*255)<<16)|(Math.round(c[1]*255)<<8)|Math.round(c[2]*255);
}

const genomeGeo = new THREE.BufferGeometry();
const gPos = new Float32Array(N_GENOMES*3);
const gCol = new Float32Array(N_GENOMES*3);
genomePos.forEach((p,i) => {
  gPos[i*3]=p.x; gPos[i*3+1]=p.y; gPos[i*3+2]=p.z;
  gCol[i*3]=DEFAULT_COL[0]; gCol[i*3+1]=DEFAULT_COL[1]; gCol[i*3+2]=DEFAULT_COL[2];
});
genomeGeo.setAttribute('position', new THREE.BufferAttribute(gPos,3));
genomeGeo.setAttribute('color',    new THREE.BufferAttribute(gCol,3));
const genomePoints = new THREE.Points(genomeGeo,
  new THREE.PointsMaterial({size:0.035, sizeAttenuation:true, transparent:true, opacity:0.9, vertexColors:true})
);
scene.add(genomePoints);

const repGroup   = new THREE.Group(); scene.add(repGroup);
const flashGroup = new THREE.Group(); scene.add(flashGroup);
const lineGroup  = new THREE.Group(); scene.add(lineGroup);
const circleGroup= new THREE.Group(); scene.add(circleGroup);

const selected = [];
const repPositions = [];
let thomsonActive = false;
let autoplayTimer = null;
let currentStep = 0;
let thomsonEnergy = 0;
let coverageDone = false;
let tourActive = false;
const tourTimeouts = [];
let tourPoll = null;

function countCovered() {
  if (repPositions.length === 0) return 0;
  let n = 0;
  genomePos.forEach((gp, i) => {
    let minD = Infinity;
    repPositions.forEach(rp => { const d = gp.distanceTo(rp); if (d < minD) minD = d; });
    if (minD < COVERAGE_THRESHOLD) n++;
  });
  return n;
}

function updateVoronoiColors() {
  const col = genomeGeo.attributes.color.array;
  if (repPositions.length === 0) {
    for (let i = 0; i < N_GENOMES; i++) {
      col[i*3]=DEFAULT_COL[0]; col[i*3+1]=DEFAULT_COL[1]; col[i*3+2]=DEFAULT_COL[2];
    }
  } else {
    for (let i = 0; i < N_GENOMES; i++) {
      let ni=0, nd=genomePos[i].distanceTo(repPositions[0]);
      for (let r=1; r<repPositions.length; r++) {
        const d=genomePos[i].distanceTo(repPositions[r]);
        if (d<nd){nd=d;ni=r;}
      }
      const c=PALETTE[ni%PALETTE.length];
      if (selected.includes(i)){
        col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
      } else if (nd < COVERAGE_THRESHOLD) {
        col[i*3]=c[0]*0.45+0.55; col[i*3+1]=c[1]*0.45+0.55; col[i*3+2]=c[2]*0.45+0.55;
      } else {
        // Uncovered: muted blue-grey
        col[i*3]=0.44; col[i*3+1]=0.50; col[i*3+2]=0.58;
      }
    }
  }
  genomeGeo.attributes.color.needsUpdate = true;
}

function fpsStep() {
  if (selected.length === 0) {
    const idx = Math.floor(Math.random()*N_GENOMES);
    selected.push(idx);
    repPositions.push(genomePos[idx].clone());
    animateNewRep(genomePos[idx]);
    updateScene();
    return;
  }
  let best=-1, bestDist=-1;
  for (let i=0; i<N_GENOMES; i++) {
    if (selected.includes(i)) continue;
    let minD=Infinity;
    for (const rp of repPositions) { const d=genomePos[i].distanceTo(rp); if (d<minD) minD=d; }
    if (minD>bestDist){bestDist=minD; best=i;}
  }
  if (best >= 0) {
    selected.push(best);
    repPositions.push(genomePos[best].clone());
    animateNewRep(genomePos[best]);
    updateScene();
    if (bestDist < COVERAGE_THRESHOLD && selected.length > 1) {
      coverageDone = true;
      if (autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
      document.getElementById('alg-btn-action').textContent = '✓ Done';
      document.getElementById('alg-btn-action').disabled = true;
      document.getElementById('alg-info-energy').textContent = 'complete';
    }
  }
}

function animateNewRep(pos) {
  const hex = paletteHex(repPositions.length-1);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.05,10,8),
    new THREE.MeshBasicMaterial({color:hex, transparent:true, opacity:1})
  );
  mesh.position.copy(pos);
  flashGroup.add(mesh);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.05,0.068,24),
    new THREE.MeshBasicMaterial({color:hex, transparent:true, opacity:0.8, side:THREE.DoubleSide})
  );
  ring.position.copy(pos);
  ring.lookAt(camera.position);
  flashGroup.add(ring);
  const t0=performance.now(), dur=900;
  (function tick(){
    const p=Math.min((performance.now()-t0)/dur,1);
    mesh.material.opacity=1-p;
    ring.material.opacity=(1-p)*0.9;
    ring.scale.setScalar(1+p*1.4);
    if(p<1) requestAnimationFrame(tick);
    else{flashGroup.remove(mesh);flashGroup.remove(ring);}
  })();
}

function updateCoverageCircles() {
  while (circleGroup.children.length) circleGroup.remove(circleGroup.children[0]);
  if (repPositions.length===0) return;
  repPositions.forEach((rp,ri) => {
    let maxD=0;
    genomePos.forEach((gp,i) => {
      if (selected.includes(i)) return;
      let ni=0,nd=gp.distanceTo(repPositions[0]);
      for(let r=1;r<repPositions.length;r++){const d=gp.distanceTo(repPositions[r]);if(d<nd){nd=d;ni=r;}}
      if(ni===ri && nd>maxD) maxD=nd;
    });
    if(maxD<0.02) return;
    const angR=2*Math.asin(Math.min(maxD/2,0.9999));
    const up=Math.abs(rp.y)<0.9?new THREE.Vector3(0,1,0):new THREE.Vector3(1,0,0);
    const u=new THREE.Vector3().crossVectors(rp,up).normalize();
    const v=new THREE.Vector3().crossVectors(rp,u).normalize();
    const pts=[];
    for(let t=0;t<=72;t++){
      const a=(t/72)*Math.PI*2;
      pts.push(rp.clone().multiplyScalar(Math.cos(angR))
        .addScaledVector(u,Math.sin(angR)*Math.cos(a))
        .addScaledVector(v,Math.sin(angR)*Math.sin(a)));
    }
    circleGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color:paletteHex(ri),transparent:true,opacity:0.55})
    ));
  });
}

function updateScene() {
  updateVoronoiColors();
  while(repGroup.children.length) repGroup.remove(repGroup.children[0]);
  repPositions.forEach((rp,ri) => {
    const hex=paletteHex(ri);
    const m=new THREE.Mesh(new THREE.SphereGeometry(0.040,14,10),new THREE.MeshBasicMaterial({color:hex}));
    m.position.copy(rp); repGroup.add(m);
    const h=new THREE.Mesh(new THREE.SphereGeometry(0.062,12,10),new THREE.MeshBasicMaterial({color:hex,transparent:true,opacity:0.12}));
    h.position.copy(rp); repGroup.add(h);
  });
  while(lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
  if(repPositions.length>0){
    genomePos.forEach((gp,i) => {
      if(selected.includes(i)) return;
      let ni=0,nearD=gp.distanceTo(repPositions[0]);
      repPositions.forEach((rp,ri)=>{const d=gp.distanceTo(rp);if(d<nearD){nearD=d;ni=ri;}});
      const geo=new THREE.BufferGeometry().setFromPoints([gp.clone().multiplyScalar(1.01),repPositions[ni].clone().multiplyScalar(1.01)]);
      lineGroup.add(new THREE.Line(geo,new THREE.LineBasicMaterial({color:paletteHex(ni),transparent:true,opacity:0.22})));
    });
  }
  updateCoverageCircles();
  updateInfo();
}

function updateInfo() {
  document.getElementById('alg-info-reps').textContent = selected.length;
  const cov = countCovered();
  const pct = Math.round(cov/N_GENOMES*100);
  document.getElementById('alg-info-cov').textContent = cov + ' / ' + N_GENOMES;
  // Progress bar
  const pb = document.getElementById('alg-progress-bar');
  const pl = document.getElementById('alg-progress-label');
  if (pb) { pb.style.width = pct + '%'; pl.textContent = cov + ' / ' + N_GENOMES + ' covered (' + pct + '%)'; }
  if (repPositions.length>0) {
    let maxMinD=0;
    genomePos.forEach((gp,i) => {
      if(selected.includes(i)) return;
      let minD=Infinity;
      repPositions.forEach(rp=>{const d=gp.distanceTo(rp);if(d<minD)minD=d;});
      if(minD>maxMinD) maxMinD=minD;
    });
    document.getElementById('alg-info-radius').textContent = (maxMinD*180/Math.PI).toFixed(1)+'°';
  } else {
    document.getElementById('alg-info-radius').textContent = '—';
  }
  if (thomsonActive) {
    document.getElementById('alg-info-energy').textContent = 'U='+thomsonEnergy.toFixed(0);
  }
}

function thomsonStep() {
  const n=repPositions.length;
  if(n<2) return;
  const forces=repPositions.map(()=>new THREE.Vector3());
  let energy=0;
  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      const diff=repPositions[i].clone().sub(repPositions[j]);
      const d2=diff.lengthSq();
      energy+=1/Math.sqrt(d2);
      const f=diff.divideScalar(d2*Math.sqrt(d2)+1e-8);
      forces[i].add(f); forces[j].sub(f);
    }
  }
  thomsonEnergy=energy;
  repPositions.forEach((rp,i)=>{rp.addScaledVector(forces[i],0.0008);rp.normalize();});
  let k=0;
  repGroup.children.forEach(m=>{m.position.copy(repPositions[Math.floor(k/2)]);k++;});
  updateInfo();
}

function reset() {
  selected.length=0; repPositions.length=0;
  thomsonActive=false; coverageDone=false;
  if(autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
  document.getElementById('alg-btn-action').textContent='▶ Run';
  document.getElementById('alg-btn-action').disabled=false;
  document.getElementById('alg-btn-thomson').textContent='▶ Run Thomson';
  document.getElementById('alg-info-energy').textContent='—';
  while(repGroup.children.length)   repGroup.remove(repGroup.children[0]);
  while(lineGroup.children.length)  lineGroup.remove(lineGroup.children[0]);
  while(flashGroup.children.length) flashGroup.remove(flashGroup.children[0]);
  while(circleGroup.children.length)circleGroup.remove(circleGroup.children[0]);
  updateVoronoiColors(); updateInfo();
}

function setStep(s) {
  currentStep=Math.max(0,Math.min(4,s));
  document.querySelectorAll('#algorithm .alg-step').forEach((el,i)=>el.classList.toggle('active',i===currentStep));
  document.getElementById('alg-step-indicator').textContent=(currentStep+1)+' / 5';
  const labels=['FINGERPRINTING','SPHERE PROJECTION','NEIGHBOR INDEX','REPRESENTATIVE SELECTION','THOMSON REFINEMENT'];
  document.getElementById('alg-canvas-label').textContent=labels[currentStep];
  const inFPS=currentStep===3, inThomson=currentStep===4;
  const btnAction=document.getElementById('alg-btn-action');
  btnAction.style.display=inFPS?'':'none';
  if(inFPS&&!coverageDone){btnAction.textContent='▶ Run';btnAction.disabled=false;}
  document.getElementById('alg-btn-autoplay').style.display=inFPS?'':'none';
  document.getElementById('alg-btn-thomson').style.display=inThomson?'':'none';
  const prog=document.getElementById('alg-progress');
  if(prog) prog.style.display=inFPS?'':'none';
}

// ── Guided tour ───────────────────────────────────────────────────────────────
function tourAt(ms, fn) {
  const id=setTimeout(()=>{if(tourActive)fn();},ms);
  tourTimeouts.push(id);
}
function clearTourTimeouts() {
  tourTimeouts.forEach(clearTimeout);
  tourTimeouts.length=0;
  if(tourPoll){clearInterval(tourPoll);tourPoll=null;}
}
function startTour() {
  if(tourActive){stopTour();return;}
  reset();
  tourActive=true;
  document.getElementById('alg-btn-tour').textContent='⏹ Stop';
  setStep(0);
  tourAt(3200,()=>setStep(1));
  tourAt(6400,()=>setStep(2));
  tourAt(9600,()=>{
    setStep(3);
    if(!coverageDone){
      document.getElementById('alg-btn-action').textContent='⏸ Pause';
      autoplayTimer=setInterval(()=>{
        if(coverageDone||selected.length>=N_GENOMES){clearInterval(autoplayTimer);autoplayTimer=null;}
        else fpsStep();
      },150);
    }
  });
  tourPoll=setInterval(()=>{
    if(!tourActive){clearInterval(tourPoll);tourPoll=null;return;}
    if(coverageDone){
      clearInterval(tourPoll);tourPoll=null;
      tourAt(1200,()=>{
        setStep(4);
        thomsonActive=true;
        document.getElementById('alg-btn-thomson').textContent='⏸ Pause Thomson';
        document.getElementById('alg-info-energy').textContent='running…';
        tourAt(8000,()=>{
          thomsonActive=false;updateScene();
          document.getElementById('alg-btn-thomson').textContent='▶ Run Thomson';
          stopTour();
        });
      });
    }
  },300);
}
function stopTour() {
  tourActive=false;clearTourTimeouts();
  if(autoplayTimer){clearInterval(autoplayTimer);autoplayTimer=null;}
  document.getElementById('alg-btn-tour').textContent='▶ Tour';
}

// ── Event handlers ────────────────────────────────────────────────────────────
document.getElementById('alg-btn-prev').onclick=()=>setStep(currentStep-1);
document.getElementById('alg-btn-next').onclick=()=>setStep(currentStep+1);
document.getElementById('alg-btn-reset').onclick=reset;
document.getElementById('alg-btn-tour').onclick=startTour;

document.getElementById('alg-btn-action').onclick=function(){
  if(coverageDone) return;
  if(autoplayTimer){
    clearInterval(autoplayTimer);autoplayTimer=null;
    this.textContent='▶ Run';
  } else {
    this.textContent='⏸ Pause';
    autoplayTimer=setInterval(()=>{
      if(coverageDone||selected.length>=N_GENOMES){clearInterval(autoplayTimer);autoplayTimer=null;return;}
      fpsStep();
    },200);
  }
};

document.getElementById('alg-btn-autoplay').onclick=function(){if(!coverageDone)fpsStep();};

document.getElementById('alg-btn-thomson').onclick=function(){
  thomsonActive=!thomsonActive;
  this.textContent=thomsonActive?'⏸ Pause Thomson':'▶ Run Thomson';
  if(!thomsonActive)updateScene();
};

document.querySelectorAll('#algorithm .alg-step').forEach((el,i)=>{
  el.addEventListener('click',()=>setStep(i));
});

setStep(0);

function animate(){
  requestAnimationFrame(animate);
  controls.update();
  if(thomsonActive)thomsonStep();
  renderer.render(scene,camera);
}
animate();

</script>
</body>
</html>
)GEODESIC";

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// ReportWriter implementation
// ─────────────────────────────────────────────────────────────────────────────

ReportWriter::ReportWriter(std::filesystem::path output_dir, std::string prefix, std::string timestamp)
    : dir_(std::move(output_dir)), prefix_(std::move(prefix)), ts_(std::move(timestamp)) {}

std::string ReportWriter::build_json(db::DBManager& db) const {
    // ── Summary ──────────────────────────────────────────────────────────────
    int64_t total_genomes = 0, total_taxa = 0, total_reps = 0, n_singletons = 0, n_failed = 0;
    double mean_cov = 0, mean_div = 0, total_rt = 0;
    {
        auto r = db.query(
            "SELECT "
            "  SUM(r.n_genomes), COUNT(*), SUM(r.n_genomes_derep), "
            "  SUM(CASE WHEN r.method='singleton' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN r.method='failed'    THEN 1 ELSE 0 END), "
            "  AVG(CASE WHEN r.method NOT IN ('singleton','fixed') THEN d.coverage_mean_ani  ELSE NULL END), "
            "  AVG(CASE WHEN r.method NOT IN ('singleton','fixed') THEN d.diversity_mean_ani ELSE NULL END), "
            "  SUM(COALESCE(d.runtime_seconds,0)) "
            "FROM results r LEFT JOIN diversity_stats d ON r.taxonomy = d.taxonomy");
        auto chunk = r->Fetch();
        if (chunk && chunk->size() > 0) {
            auto get_i64 = [&](int c) -> int64_t {
                auto v = chunk->GetValue(c, 0);
                return v.IsNull() ? 0 : v.GetValue<int64_t>();
            };
            auto get_d = [&](int c) -> double {
                auto v = chunk->GetValue(c, 0);
                return v.IsNull() ? 0.0 : v.GetValue<double>();
            };
            total_genomes = get_i64(0);
            total_taxa    = get_i64(1);
            total_reps    = get_i64(2);
            n_singletons  = get_i64(3);
            n_failed      = get_i64(4);
            mean_cov      = get_d(5);
            mean_div      = get_d(6);
            total_rt      = get_d(7);
        }
    }

    // ── Taxa data (columnar) ─────────────────────────────────────────────────
    auto taxa_res = db.query(
        "SELECT "
        "  r.taxonomy, r.n_genomes, r.n_genomes_derep, r.method, "
        "  COALESCE(d.reduction_ratio,   0.0) AS rr, "
        "  COALESCE(d.coverage_mean_ani, 100.0) AS cm, "
        "  COALESCE(d.diversity_mean_ani, 0.0) AS dm, "
        "  COALESCE(d.diversity_ani_range, 0.0) AS da, "
        "  COALESCE(d.runtime_seconds,   0.0) AS rt "
        "FROM results r LEFT JOIN diversity_stats d ON r.taxonomy = d.taxonomy "
        "ORDER BY r.n_genomes DESC");

    std::ostringstream tx_s, ng_s, nr_s, rr_s, cm_s, dm_s, mt_s, rt_s;
    tx_s << '['; ng_s << '['; nr_s << '['; rr_s << '[';
    cm_s << '['; dm_s << '['; mt_s << '['; rt_s << '[';
    bool first = true;
    while (auto chunk = taxa_res->Fetch()) {
        for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
            if (!first) { tx_s<<','; ng_s<<','; nr_s<<','; rr_s<<','; cm_s<<','; dm_s<<','; mt_s<<','; rt_s<<','; }
            first = false;
            tx_s << '"' << esc_json(chunk->GetValue(0,row).GetValue<std::string>()) << '"';
            ng_s << chunk->GetValue(1,row).GetValue<int32_t>();
            nr_s << chunk->GetValue(2,row).GetValue<int32_t>();
            mt_s << '"' << esc_json(chunk->GetValue(3,row).GetValue<std::string>()) << '"';
            rr_s << fmt_d(chunk->GetValue(4,row).IsNull() ? 0.0 : chunk->GetValue(4,row).GetValue<double>(), 4);
            cm_s << fmt_d(chunk->GetValue(5,row).IsNull() ? 0.0 : chunk->GetValue(5,row).GetValue<double>(), 4);
            dm_s << fmt_d(chunk->GetValue(6,row).IsNull() ? 0.0 : chunk->GetValue(6,row).GetValue<double>(), 4);
            rt_s << fmt_d(chunk->GetValue(8,row).IsNull() ? 0.0 : chunk->GetValue(8,row).GetValue<double>(), 2);
        }
    }
    tx_s<<']'; ng_s<<']'; nr_s<<']'; rr_s<<']'; cm_s<<']'; dm_s<<']'; mt_s<<']'; rt_s<<']';

    // ── Assemble JSON ────────────────────────────────────────────────────────
    std::ostringstream j;
    j << "{"
      << "\"meta\":{\"prefix\":\"" << esc_json(prefix_) << "\","
      <<            "\"timestamp\":\"" << esc_json(ts_) << "\"},"
      << "\"summary\":{"
      <<   "\"n_genomes\":"   << total_genomes << ","
      <<   "\"n_taxa\":"      << total_taxa    << ","
      <<   "\"n_reps\":"      << total_reps    << ","
      <<   "\"n_singletons\":" << n_singletons  << ","
      <<   "\"n_failed\":"    << n_failed      << ","
      <<   "\"mean_coverage\":" << fmt_d(mean_cov, 4) << ","
      <<   "\"mean_diversity\":" << fmt_d(mean_div, 4) << ","
      <<   "\"total_runtime\":" << fmt_d(total_rt, 1)
      << "},"
      << "\"taxa\":{"
      <<   "\"tx\":" << tx_s.str() << ","
      <<   "\"ng\":" << ng_s.str() << ","
      <<   "\"nr\":" << nr_s.str() << ","
      <<   "\"rr\":" << rr_s.str() << ","
      <<   "\"cm\":" << cm_s.str() << ","
      <<   "\"dm\":" << dm_s.str() << ","
      <<   "\"mt\":" << mt_s.str() << ","
      <<   "\"rt\":" << rt_s.str()
      << "}"
      << "}";
    return j.str();
}

static std::string fmt_runtime(double sec) {
    if (sec < 60)   return std::to_string(static_cast<int>(sec)) + "s";
    if (sec < 3600) return std::to_string(static_cast<int>(sec/60)) + "m " + std::to_string(static_cast<int>(std::fmod(sec,60))) + "s";
    int h = static_cast<int>(sec/3600), m = static_cast<int>(std::fmod(sec,3600)/60);
    return std::to_string(h) + "h " + std::to_string(m) + "m";
}

void ReportWriter::write(db::DBManager& db) const {
    // ── Build JSON data ──────────────────────────────────────────────────────
    spdlog::info("Generating HTML report...");
    std::string json;
    try {
        json = build_json(db);
    } catch (const std::exception& e) {
        spdlog::warn("Report generation failed: {}", e.what());
        return;
    }

    // Extract scalar summary values for HTML placeholder substitution
    int64_t n_genomes=0, n_taxa=0, n_reps=0, n_singletons=0, n_failed=0;
    double mean_cov=0, total_rt=0;
    {
        auto r = db.query(
            "SELECT SUM(n_genomes), COUNT(*), SUM(n_genomes_derep), "
            "  SUM(CASE WHEN method='singleton' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN method='failed'    THEN 1 ELSE 0 END) "
            "FROM results");
        auto chunk = r->Fetch();
        if (chunk && chunk->size() > 0) {
            auto gi = [&](int c) -> int64_t { auto v=chunk->GetValue(c,0); return v.IsNull()?0:v.GetValue<int64_t>(); };
            n_genomes=gi(0); n_taxa=gi(1); n_reps=gi(2); n_singletons=gi(3); n_failed=gi(4);
        }
        auto r2 = db.query("SELECT SUM(runtime_seconds) FROM diversity_stats");
        auto c2 = r2->Fetch();
        if (c2 && c2->size()>0 && !c2->GetValue(0,0).IsNull()) total_rt = c2->GetValue(0,0).GetValue<double>();
    }

    auto replace_all = [](std::string s, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
        return s;
    };

    // ── Main report ──────────────────────────────────────────────────────────
    std::string html = REPORT_HTML;
    html = replace_all(html, "__DATA__",        json);
    html = replace_all(html, "__PREFIX__",      prefix_);
    html = replace_all(html, "__TIMESTAMP__",   ts_);
    html = replace_all(html, "__N_GENOMES__",   std::to_string(n_genomes));
    html = replace_all(html, "__N_TAXA__",      std::to_string(n_taxa));
    html = replace_all(html, "__N_REPS__",      std::to_string(n_reps));
    html = replace_all(html, "__N_SINGLETONS__",std::to_string(n_singletons));
    html = replace_all(html, "__N_FAILED__",    std::to_string(n_failed));
    html = replace_all(html, "__RUNTIME__",     fmt_runtime(total_rt));

    auto report_path = dir_ / (prefix_ + "_report.html");
    std::ofstream rf(report_path);
    if (!rf) throw std::runtime_error("Cannot write report: " + report_path.string());
    rf << html;
    rf.close();
    spdlog::info("Report written to {}", report_path.string());

    // ── Algorithm visualization page ─────────────────────────────────────────
    std::string alg_html = ALGORITHM_HTML;
    alg_html = replace_all(alg_html, "__PREFIX__", prefix_);
    auto alg_path = dir_ / (prefix_ + "_algorithm.html");
    std::ofstream af(alg_path);
    if (!af) throw std::runtime_error("Cannot write algorithm page: " + alg_path.string());
    af << alg_html;
    af.close();
    spdlog::info("Algorithm page written to {}", alg_path.string());
}

} // namespace derep
