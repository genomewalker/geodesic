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
<link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--s1:#0c1020;--s2:#111828;--b1:#1c2840;--b2:#243254;
  --t1:#c4d0e8;--t2:#7a8fb0;--t3:#3d5070;
  --teal:#00d4a3;--teal-d:rgba(0,212,163,.1);--teal-b:rgba(0,212,163,.18);
  --blue:#4a9eff;--blue-d:rgba(74,158,255,.1);
  --amber:#f0a030;--amber-d:rgba(240,160,48,.1);
  --rose:#e05070;--rose-d:rgba(224,80,112,.1);
  --purple:#9b7cf4;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--t1);font-family:'Outfit',sans-serif;font-size:14px;line-height:1.6;min-height:100vh}

/* ── Nav ── */
.nav{
  position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:20px;
  padding:0 32px;height:50px;
  background:rgba(7,9,15,.92);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--b1);
}
.nav-logo{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:500;color:var(--teal);letter-spacing:-.02em}
.nav-sep{width:1px;height:14px;background:var(--b2)}
.nav-run{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t3)}
.nav-links{margin-left:auto;display:flex;gap:2px}
.nav-links a{
  padding:6px 12px;border-radius:6px;text-decoration:none;
  color:var(--t2);font-size:12px;font-weight:500;transition:all .15s;letter-spacing:.02em;
}
.nav-links a:hover{color:var(--t1);background:var(--s2)}
.nav-links a.active{color:var(--teal)}

/* ── Hero ── */
.hero{padding:64px 48px 40px;max-width:960px}
.hero-tag{
  font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--teal);margin-bottom:14px;
}
.hero h1{
  font-family:'Crimson Pro',serif;font-size:60px;font-weight:600;
  line-height:1.05;color:var(--t1);letter-spacing:-.02em;
}
.hero h1 .hi{color:var(--teal)}
.hero-sub{
  margin-top:12px;font-family:'Crimson Pro',serif;font-size:20px;
  color:var(--t2);font-style:italic;
}
.hero-meta{
  margin-top:20px;display:flex;gap:24px;flex-wrap:wrap;
}
.hero-badge{
  display:flex;align-items:center;gap:6px;
  font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--t3);
}
.hero-badge span{color:var(--t2)}

/* ── Metric strip ── */
.metric-strip{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:1px;background:var(--b1);
  border-top:1px solid var(--b1);border-bottom:1px solid var(--b1);
  margin:0 0 40px;
}
.metric-cell{
  background:var(--s1);padding:20px 28px;
  transition:background .15s;cursor:default;
}
.metric-cell:hover{background:var(--s2)}
.metric-val{
  font-family:'Crimson Pro',serif;font-size:38px;font-weight:600;
  line-height:1;color:var(--t1);letter-spacing:-.01em;
}
.metric-val.c-teal{color:var(--teal)}
.metric-val.c-blue{color:var(--blue)}
.metric-val.c-amber{color:var(--amber)}
.metric-val.c-rose{color:var(--rose)}
.metric-val.c-purple{color:var(--purple)}
.metric-lbl{
  margin-top:6px;font-size:10px;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;color:var(--t3);
}
.metric-desc{margin-top:2px;font-size:11px;color:var(--t3)}

/* ── Section wrapper ── */
.sec{padding:0 32px 48px}
.sec-title{
  font-family:'Crimson Pro',serif;font-size:28px;font-weight:600;
  color:var(--t1);margin-bottom:20px;display:flex;align-items:baseline;gap:12px;
}
.sec-title small{font-family:'Outfit',sans-serif;font-size:12px;color:var(--t3);font-weight:400}

/* ── Charts ── */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:0}
.chart-card{
  background:var(--s1);border:1px solid var(--b1);border-radius:10px;
  padding:20px 22px;
}
.chart-title{
  font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--t2);margin-bottom:16px;
}
.chart-card canvas{display:block;width:100%}

/* ── Explorer ── */
.explorer-head{
  display:flex;align-items:center;gap:16px;margin-bottom:16px;
  flex-wrap:wrap;
}
.search-box{
  background:var(--s1);border:1px solid var(--b1);border-radius:8px;
  display:flex;align-items:center;gap:8px;padding:0 12px;
  transition:border-color .15s;
}
.search-box:focus-within{border-color:var(--teal)}
.search-icon{color:var(--t3);font-size:14px;flex-shrink:0}
.search-input{
  background:none;border:none;outline:none;
  font-family:'JetBrains Mono',monospace;font-size:12px;
  color:var(--t1);padding:8px 0;width:300px;
}
.search-input::placeholder{color:var(--t3)}
.search-count{
  margin-left:auto;font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--t3);white-space:nowrap;
}
.sort-btns{display:flex;gap:6px;flex-wrap:wrap}
.sort-btn{
  background:var(--s1);border:1px solid var(--b1);border-radius:6px;
  padding:5px 10px;font-size:11px;color:var(--t2);cursor:pointer;
  transition:all .15s;font-family:'Outfit',sans-serif;white-space:nowrap;
}
.sort-btn:hover{color:var(--t1);border-color:var(--b2)}
.sort-btn.active{color:var(--teal);border-color:rgba(0,212,163,.35);background:var(--teal-d)}

/* ── Table ── */
.tbl-wrap{
  border:1px solid var(--b1);border-radius:10px;overflow:hidden;
  margin-bottom:16px;
}
table{width:100%;border-collapse:collapse}
thead th{
  background:var(--s1);padding:10px 14px;text-align:left;
  font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--t2);border-bottom:1px solid var(--b1);
  white-space:nowrap;user-select:none;
}
thead th.r{text-align:right}
td{
  padding:8px 14px;border-bottom:1px solid rgba(28,40,64,.6);
  font-size:12.5px;vertical-align:middle;
}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(17,24,40,.5)}
td.tx{
  font-family:'JetBrains Mono',monospace;font-size:11.5px;
  max-width:380px;
}
td.nm{
  font-family:'JetBrains Mono',monospace;font-size:12px;
  text-align:right;color:var(--t2);white-space:nowrap;
}
td.nm.hi{color:var(--t1)}
.chip{
  display:inline-block;padding:1px 5px;border-radius:3px;
  font-size:9.5px;font-weight:600;letter-spacing:.04em;
  margin-right:3px;vertical-align:middle;
}
.chip-p{background:#0e1a30;color:#4a7aaa}
.chip-g{background:#081e18;color:#1e8060}
.chip-s{color:var(--t1)}
.mini-bar{
  display:inline-block;height:3px;border-radius:2px;
  opacity:.7;vertical-align:middle;margin-right:6px;
}
.method-tag{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-size:9px;font-weight:600;letter-spacing:.05em;font-family:'JetBrains Mono',monospace;
}
.mt-singleton{background:#1a1a1a;color:#555}
.mt-tiny{background:var(--teal-d);color:var(--teal)}
.mt-geodesic{background:var(--blue-d);color:var(--blue)}

/* ── Pager ── */
.pager{display:flex;align-items:center;justify-content:center;gap:4px;padding:4px 0 20px}
.pg-btn{
  background:var(--s1);border:1px solid var(--b1);border-radius:6px;
  padding:5px 10px;color:var(--t2);cursor:pointer;font-size:12px;
  transition:all .15s;font-family:'Outfit',sans-serif;
}
.pg-btn:hover{color:var(--t1);border-color:var(--b2)}
.pg-btn.on{color:var(--teal);border-color:rgba(0,212,163,.35);background:var(--teal-d)}
.pg-btn:disabled{opacity:.3;cursor:not-allowed}
.pg-sep{color:var(--t3);padding:0 4px;font-size:12px}

/* ── Footer ── */
footer{
  padding:36px 32px;border-top:1px solid var(--b1);
  display:flex;align-items:center;gap:16px;flex-wrap:wrap;
}
.footer-logo{font-family:'JetBrains Mono',monospace;font-size:14px;color:var(--teal)}
.footer-desc{font-size:12px;color:var(--t3)}
.footer-links{margin-left:auto;display:flex;gap:16px}
.footer-links a{font-size:12px;color:var(--t3);text-decoration:none;transition:color .15s}
.footer-links a:hover{color:var(--t1)}

/* ── Utility ── */
.muted{color:var(--t3)!important}
.loading-msg{padding:40px;text-align:center;color:var(--t3);font-size:13px}

/* ── Algorithm section ── */
#algorithm{height:100vh;display:flex;overflow:hidden;background:#000005}
#algorithm .alg-panel{
  width:340px;min-width:340px;background:#080d18;border-right:1px solid #182038;
  display:flex;flex-direction:column;overflow:hidden;z-index:10;
}
#algorithm .alg-panel-hdr{padding:24px 24px 16px;border-bottom:1px solid #182038;flex-shrink:0}
#algorithm .alg-logo{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:500;color:var(--teal);letter-spacing:-.02em}
#algorithm .alg-logo-sub{margin-top:4px;font-size:11px;color:#3d5070;letter-spacing:.08em;text-transform:uppercase}
#algorithm .alg-steps{flex:1;overflow-y:auto;padding:16px 0}
#algorithm .alg-steps::-webkit-scrollbar{width:4px}
#algorithm .alg-steps::-webkit-scrollbar-thumb{background:#1e2d4a;border-radius:2px}
#algorithm .alg-step{padding:14px 24px;border-left:2px solid transparent;cursor:pointer;transition:all .15s;opacity:.45}
#algorithm .alg-step.active{border-left-color:var(--teal);opacity:1;background:rgba(0,212,163,.04)}
#algorithm .alg-step:hover{opacity:.8}
#algorithm .alg-step-num{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--teal);letter-spacing:.12em;margin-bottom:4px}
#algorithm .alg-step-title{font-size:13px;font-weight:600;color:var(--t1);margin-bottom:6px}
#algorithm .alg-step-body{font-size:12px;color:var(--t2);line-height:1.6;display:none}
#algorithm .alg-step.active .alg-step-body{display:block}
#algorithm .alg-eq{
  font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--amber);
  background:rgba(240,160,48,.08);border:1px solid rgba(240,160,48,.15);
  border-radius:4px;padding:6px 10px;margin-top:8px;line-height:1.7;
}
#algorithm .alg-controls{padding:16px 24px;border-top:1px solid #182038;flex-shrink:0}
#algorithm .alg-ctrl-row{display:flex;gap:8px;margin-bottom:8px;align-items:center}
#algorithm .alg-ctrl-row:last-child{margin-bottom:0}
#algorithm .alg-btn{
  background:#0c1220;border:1px solid #182038;border-radius:6px;
  padding:7px 14px;color:var(--t2);font-family:'Outfit',sans-serif;
  font-size:12px;font-weight:500;cursor:pointer;transition:all .15s;white-space:nowrap;
}
#algorithm .alg-btn:hover{color:var(--t1);border-color:#1e2d4a}
#algorithm .alg-btn.primary{color:var(--teal);border-color:rgba(0,212,163,.3);background:rgba(0,212,163,.1)}
#algorithm .alg-btn.primary:hover{background:rgba(0,212,163,.18)}
#algorithm .alg-btn:disabled{opacity:.35;cursor:not-allowed}
#algorithm .alg-step-nav{display:flex;align-items:center;gap:8px;flex:1}
#algorithm .alg-step-lbl{flex:1;text-align:center;font-family:'JetBrains Mono',monospace;font-size:11px;color:#3d5070}
#algorithm .alg-info-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:#182038;border:1px solid #182038;border-radius:8px;overflow:hidden;margin-top:12px;
}
#algorithm .alg-info-cell{background:#080d18;padding:10px 12px}
#algorithm .alg-info-key{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#3d5070;margin-bottom:2px}
#algorithm .alg-info-val{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;color:var(--teal)}
#algorithm .alg-canvas-wrap{flex:1;position:relative;overflow:hidden}
#algorithm #sphere-canvas{width:100%;height:100%;display:block}
#algorithm .alg-canvas-lbl{
  position:absolute;top:20px;right:24px;
  font-family:'JetBrains Mono',monospace;font-size:11px;color:#3d5070;
  letter-spacing:.08em;text-transform:uppercase;
}
#algorithm .alg-legend{position:absolute;bottom:24px;right:24px;display:flex;flex-direction:column;gap:6px}
#algorithm .alg-legend-item{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--t2)}
#algorithm .alg-legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
</style>
</head>
<body>

<nav class="nav">
  <span class="nav-logo">geodesic</span>
  <span class="nav-sep"></span>
  <span class="nav-run">__PREFIX__ · __TIMESTAMP__</span>
  <div class="nav-links">
    <a href="#overview">Overview</a>
    <a href="#explorer">Explorer</a>
    <a href="#algorithm">Algorithm</a>
  </div>
</nav>

<section id="overview">
  <div class="hero">
    <div class="hero-tag">geodesic run report</div>
    <h1><span class="hi counter" data-n="__N_GENOMES__">0</span><br>genomes processed</h1>
    <p class="hero-sub">
      <span class="counter" data-n="__N_TAXA__">0</span> species ·
      <span class="counter" data-n="__N_REPS__">0</span> representatives selected
    </p>
    <div class="hero-meta">
      <div class="hero-badge">prefix <span>__PREFIX__</span></div>
      <div class="hero-badge">run <span>__TIMESTAMP__</span></div>
      <div class="hero-badge">runtime <span>__RUNTIME__</span></div>
      <div class="hero-badge">singletons <span>__N_SINGLETONS__</span></div>
    </div>
  </div>

  <div class="metric-strip">
    <div class="metric-cell">
      <div class="metric-val c-teal"><span class="counter" data-n="__N_REPS__" data-fmt="int">0</span></div>
      <div class="metric-lbl">Representatives</div>
      <div class="metric-desc">selected across all species</div>
    </div>
    <div class="metric-cell">
      <div class="metric-val c-blue"><span id="m-cov">—</span></div>
      <div class="metric-lbl">Mean Coverage ANI</div>
      <div class="metric-desc">genome to nearest representative</div>
    </div>
    <div class="metric-cell">
      <div class="metric-val c-amber"><span id="m-div">—</span></div>
      <div class="metric-lbl">Mean Diversity ANI</div>
      <div class="metric-desc">pairwise among representatives</div>
    </div>
    <div class="metric-cell">
      <div class="metric-val c-purple"><span id="m-red">—</span></div>
      <div class="metric-lbl">Mean Reduction</div>
      <div class="metric-desc">fraction of genomes retained</div>
    </div>
    <div class="metric-cell">
      <div class="metric-val c-rose"><span class="counter" data-n="__N_FAILED__">0</span></div>
      <div class="metric-lbl">Failed Taxa</div>
      <div class="metric-desc">processing errors</div>
    </div>
  </div>

  <div class="sec">
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title">Taxon Size Distribution</div>
        <canvas id="chart-size" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Coverage ANI Distribution</div>
        <canvas id="chart-cov" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Reduction Ratio Distribution</div>
        <canvas id="chart-rr" height="160"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Processing Method</div>
        <canvas id="chart-method" height="160"></canvas>
      </div>
    </div>
  </div>
</section>

<section id="explorer">
  <div class="sec">
    <div class="sec-title">
      Taxa Explorer
      <small>click column headers to sort · search filters all fields</small>
    </div>
    <div class="explorer-head">
      <div class="search-box">
        <span class="search-icon">⌕</span>
        <input class="search-input" id="q" type="text" placeholder="search taxonomy…" autocomplete="off" spellcheck="false">
      </div>
      <span class="search-count" id="cnt">— taxa</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th data-col="tx">Species</th>
            <th data-col="ng" class="r">Genomes</th>
            <th data-col="nr" class="r">Reps</th>
            <th data-col="rr" class="r">Reduction</th>
            <th data-col="cm" class="r">Coverage ANI</th>
            <th data-col="dm" class="r">Diversity ANI</th>
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
  </div>
</section>

<section id="algorithm">
<aside class="alg-panel">
  <div class="alg-panel-hdr">
    <div class="alg-logo">geodesic</div>
    <div class="alg-logo-sub">Algorithm Visualization</div>
  </div>
  <div class="alg-steps" id="alg-steps">
    <div class="alg-step active" data-step="0">
      <div class="alg-step-num">01 / OPH SKETCHING</div>
      <div class="alg-step-title">Genome Embedding</div>
      <div class="alg-step-body">
        Each genome is processed with One-Permutation Hashing (OPH): all k-mers (k=21) are hashed into m=10,000 bins. The minimum hash value per bin forms the genome's signature.
        <div class="alg-eq">P[sig_A[t] = sig_B[t]] = J(A,B)</div>
        This is an unbiased estimator of the Jaccard similarity between the k-mer sets of any two genomes.
      </div>
    </div>
    <div class="alg-step" data-step="1">
      <div class="alg-step-num">02 / COUNTSKETCH</div>
      <div class="alg-step-title">Sphere Projection</div>
      <div class="alg-step-body">
        The OPH signature (10,000 bins) is compressed to a 256-dimensional unit vector via CountSketch: each bin maps to a random dimension with a ±1 sign flip. Normalizing places the genome on the unit sphere S²⁵⁵.
        <div class="alg-eq">E[u·v] = J(A,B)<br>ANI = (2J / (1+J))^(1/k) × 100</div>
        The dot product is an unbiased Jaccard estimator — no calibration needed.
      </div>
    </div>
    <div class="alg-step" data-step="2">
      <div class="alg-step-num">03 / HNSW INDEX</div>
      <div class="alg-step-title">Fast Nearest Neighbors</div>
      <div class="alg-step-body">
        All genome vectors are inserted into a Hierarchical Navigable Small World graph. This enables sub-linear approximate nearest-neighbor queries on the sphere — finding the k closest genomes to any query point in O(log n) time instead of O(n).
        <div class="alg-eq">isolation(g) = mean distance to<br>k nearest neighbors</div>
        Isolation scores guide representative selection: genomes in sparse regions are more likely to be selected.
      </div>
    </div>
    <div class="alg-step" data-step="3">
      <div class="alg-step-num">04 / FARTHEST POINT SAMPLING</div>
      <div class="alg-step-title">Select Representatives</div>
      <div class="alg-step-body">
        FPS greedily places each new representative as far as possible from all already-selected ones. This is a 2-approximation to the k-center problem: the maximum coverage radius is at most twice optimal.
        <div class="alg-eq">next_rep = argmax_g min_r d(g, r)</div>
        Click <strong style="color:var(--teal)">Add Representative</strong> to step through the selection, or use Auto-play.
      </div>
    </div>
    <div class="alg-step" data-step="4">
      <div class="alg-step-num">05 / THOMSON MERGE</div>
      <div class="alg-step-title">Electrostatic Equilibrium</div>
      <div class="alg-step-body">
        After FPS, representatives within <code style="color:var(--amber)">min_rep_distance</code> of each other are coalesced via Union-Find. In Thomson mode, remaining reps drift under mutual Coulomb repulsion to maximize their minimum pairwise separation.
        <div class="alg-eq">U = Σ_{i≠j} 1/‖r_i − r_j‖</div>
        The equilibrium configuration maximally tiles the sphere — the continuous Thomson problem.
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
    </div>
    <div class="alg-ctrl-row">
      <button class="alg-btn primary" id="alg-btn-action">▶ Add Representative</button>
      <button class="alg-btn" id="alg-btn-reset">Reset</button>
    </div>
    <div class="alg-ctrl-row">
      <button class="alg-btn" id="alg-btn-autoplay" style="flex:1">Auto-play FPS</button>
      <button class="alg-btn" id="alg-btn-thomson" style="flex:1">Thomson Mode</button>
    </div>
    <div class="alg-info-grid">
      <div class="alg-info-cell">
        <div class="alg-info-key">Representatives</div>
        <div class="alg-info-val" id="alg-info-reps">0 / 80</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Coverage radius</div>
        <div class="alg-info-val" id="alg-info-radius">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Thomson energy</div>
        <div class="alg-info-val" id="alg-info-energy">—</div>
      </div>
      <div class="alg-info-cell">
        <div class="alg-info-key">Angular dist</div>
        <div class="alg-info-val" id="alg-info-dist">—</div>
      </div>
    </div>
  </div>
</aside>

<div class="alg-canvas-wrap">
  <canvas id="sphere-canvas"></canvas>
  <div class="alg-canvas-lbl" id="alg-canvas-label">UNIT SPHERE S²⁵⁵</div>
  <div class="alg-legend">
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#2a5090"></div>
      <span>Genome embedding</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#00d4a3"></div>
      <span>Representative</span>
    </div>
    <div class="alg-legend-item">
      <div class="alg-legend-dot" style="background:#f0a030"></div>
      <span>Coverage boundary</span>
    </div>
  </div>
</div>
</section>

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
  ctx.strokeStyle = '#1c2840'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ch*(1 - i/4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+cw, y); ctx.stroke();
    const v = Math.round(maxV*i/4);
    ctx.fillStyle = '#3d5070'; ctx.font = '10px JetBrains Mono,monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v >= 10000 ? (v/1000).toFixed(0)+'k' : v >= 1000 ? (v/1000).toFixed(1)+'k' : v, pad.l-5, y+3.5);
  }

  vals.forEach((v, i) => {
    const bh = (v/maxV)*ch;
    const x = pad.l + i*bw, y = pad.t + ch - bh;
    const gr = ctx.createLinearGradient(0, y, 0, y+bh);
    gr.addColorStop(0, color); gr.addColorStop(1, color+'44');
    ctx.fillStyle = gr;
    if (ctx.roundRect) {
      ctx.beginPath(); ctx.roundRect(x+2, y, bw-4, bh, [3,3,0,0]); ctx.fill();
    } else {
      ctx.fillRect(x+2, y, bw-4, bh);
    }
    const step = Math.ceil(vals.length / 10);
    if (i % step === 0 || i === vals.length-1) {
      ctx.fillStyle = '#3d5070'; ctx.font = '9px JetBrains Mono,monospace';
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
  drawBars(document.getElementById('chart-size'), sKeys, sKeys.map(k=>sizeBuckets[k]), '#4a9eff');

  // 2. Coverage ANI (only non-singleton taxa)
  const covVals = t.cm.filter((_,i) => t.mt[i]!=='singleton' && t.cm[i]>0);
  const covEdges = [90,91,92,93,94,95,96,97,98,99,99.5,100,100.01];
  const cov = histCounts(covVals, covEdges);
  drawBars(document.getElementById('chart-cov'), cov.labels, cov.counts, '#00d4a3');

  // 3. Reduction ratio (non-singleton taxa with >1 genome)
  const rrVals = t.rr.filter((_,i) => t.ng[i]>1);
  const rrEdges = Array.from({length:21},(_,i)=>i*0.05);
  const rr = histCounts(rrVals, rrEdges);
  drawBars(document.getElementById('chart-rr'), rr.labels, rr.counts, '#f0a030');

  // 4. Method breakdown
  const mtCounts = {};
  t.mt.forEach(m => { mtCounts[m]=(mtCounts[m]||0)+1; });
  const mtKeys = Object.keys(mtCounts).sort((a,b)=>mtCounts[b]-mtCounts[a]);
  drawBars(document.getElementById('chart-method'), mtKeys, mtKeys.map(k=>mtCounts[k]), '#9b7cf4');
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
</script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function(){
'use strict';

const N_GENOMES = 80;
const canvas = document.getElementById('sphere-canvas');
const wrap = canvas.parentElement;

const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000005, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(0, 0.5, 2.8);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
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

{
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(1200 * 3);
  for (let i = 0; i < 1200; i++) {
    const phi = Math.random() * Math.PI * 2;
    const theta = Math.acos(Math.random() * 2 - 1);
    const r = 60 + Math.random() * 80;
    pos[i*3]   = r * Math.sin(theta) * Math.cos(phi);
    pos[i*3+1] = r * Math.sin(theta) * Math.sin(phi);
    pos[i*3+2] = r * Math.cos(theta);
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({color:0x334466, size:0.15, sizeAttenuation:true})));
}

const sphereObj = new THREE.Mesh(
  new THREE.SphereGeometry(1, 48, 36),
  new THREE.MeshBasicMaterial({color:0x152040, wireframe:true, transparent:true, opacity:0.12})
);
scene.add(sphereObj);

scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(0.998, 32, 24),
  new THREE.MeshBasicMaterial({color:0x030610, transparent:true, opacity:0.6})
));

function randOnSphere() {
  let x, y, s;
  do { x = Math.random()*2-1; y = Math.random()*2-1; s = x*x+y*y; } while (s >= 1);
  const sq = Math.sqrt(1 - s);
  return new THREE.Vector3(2*x*sq, 2*y*sq, 1-2*s).normalize();
}

const CLUSTER_CENTERS = 8;
const centers = Array.from({length:CLUSTER_CENTERS}, randOnSphere);
const genomePos = Array.from({length:N_GENOMES}, (_, i) => {
  if (Math.random() < 0.65) {
    const c = centers[i % CLUSTER_CENTERS];
    const v = randOnSphere();
    return c.clone().lerp(v, 0.18 + Math.random()*0.25).normalize();
  }
  return randOnSphere();
});

const genomeGeo = new THREE.BufferGeometry();
const gPos = new Float32Array(N_GENOMES * 3);
genomePos.forEach((p,i) => { gPos[i*3]=p.x; gPos[i*3+1]=p.y; gPos[i*3+2]=p.z; });
genomeGeo.setAttribute('position', new THREE.BufferAttribute(gPos, 3));
const genomePoints = new THREE.Points(genomeGeo,
  new THREE.PointsMaterial({color:0x1a4878, size:0.03, sizeAttenuation:true, transparent:true, opacity:0.9})
);
scene.add(genomePoints);

const repGroup = new THREE.Group();
scene.add(repGroup);
const flashGroup = new THREE.Group();
scene.add(flashGroup);
const lineGroup = new THREE.Group();
scene.add(lineGroup);

const selected = [];
const repPositions = [];
let thomsonActive = false;
let autoplayTimer = null;
let currentStep = 0;
let thomsonEnergy = 0;

function fpsStep() {
  if (selected.length === 0) {
    const idx = Math.floor(Math.random() * N_GENOMES);
    selected.push(idx);
    repPositions.push(genomePos[idx].clone());
    animateNewRep(genomePos[idx]);
    updateScene();
    return;
  }
  let best = -1, bestDist = -1;
  for (let i = 0; i < N_GENOMES; i++) {
    if (selected.includes(i)) continue;
    let minD = Infinity;
    for (const rp of repPositions) {
      const d = genomePos[i].distanceTo(rp);
      if (d < minD) minD = d;
    }
    if (minD > bestDist) { bestDist = minD; best = i; }
  }
  if (best >= 0) {
    selected.push(best);
    repPositions.push(genomePos[best].clone());
    animateNewRep(genomePos[best]);
    updateScene();
  }
}

function animateNewRep(pos) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.04, 10, 8),
    new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:1})
  );
  mesh.position.copy(pos);
  flashGroup.add(mesh);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.04, 0.055, 24),
    new THREE.MeshBasicMaterial({color:0x00d4a3, transparent:true, opacity:0.9, side:THREE.DoubleSide})
  );
  ring.position.copy(pos);
  ring.lookAt(camera.position);
  flashGroup.add(ring);
  const t0 = performance.now();
  const dur = 900;
  function tick() {
    const p = Math.min((performance.now()-t0)/dur, 1);
    mesh.material.opacity = 1 - p;
    ring.material.opacity = (1-p)*0.9;
    ring.scale.setScalar(1 + p * 1.4);
    if (p < 1) requestAnimationFrame(tick);
    else { flashGroup.remove(mesh); flashGroup.remove(ring); }
  }
  requestAnimationFrame(tick);
}

function updateScene() {
  genomePoints.material.color.setHex(0x1a4878);
  while (repGroup.children.length) repGroup.remove(repGroup.children[0]);
  repPositions.forEach(rp => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.038, 12, 10),
      new THREE.MeshBasicMaterial({color:0x00d4a3})
    );
    m.position.copy(rp);
    repGroup.add(m);
    const glow = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 12, 10),
      new THREE.MeshBasicMaterial({color:0x00d4a3, transparent:true, opacity:0.12})
    );
    glow.position.copy(rp);
    repGroup.add(glow);
  });
  while (lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
  if (repPositions.length > 0) {
    genomePos.forEach((gp, i) => {
      if (selected.includes(i)) return;
      let nearest = repPositions[0], nearD = gp.distanceTo(repPositions[0]);
      repPositions.forEach(rp => { const d = gp.distanceTo(rp); if (d < nearD) { nearD = d; nearest = rp; } });
      const geo = new THREE.BufferGeometry().setFromPoints([gp.clone().multiplyScalar(1.01), nearest.clone().multiplyScalar(1.01)]);
      lineGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({color:0x1a3a60, transparent:true, opacity:0.45})));
    });
  }
  updateInfo();
}

function updateInfo() {
  document.getElementById('alg-info-reps').textContent = selected.length + ' / ' + N_GENOMES;
  if (repPositions.length > 0) {
    let maxMinD = 0;
    genomePos.forEach((gp, i) => {
      if (selected.includes(i)) return;
      let minD = Infinity;
      repPositions.forEach(rp => { const d = gp.distanceTo(rp); if (d < minD) minD = d; });
      if (minD > maxMinD) maxMinD = minD;
    });
    document.getElementById('alg-info-radius').textContent = maxMinD.toFixed(3) + ' rad';
    document.getElementById('alg-info-dist').textContent = (maxMinD * 180 / Math.PI).toFixed(1) + '°';
  } else {
    document.getElementById('alg-info-radius').textContent = '—';
    document.getElementById('alg-info-dist').textContent = '—';
  }
  if (thomsonActive) {
    document.getElementById('alg-info-energy').textContent = thomsonEnergy.toFixed(1);
  }
}

function thomsonStep() {
  const n = repPositions.length;
  if (n < 2) return;
  const forces = repPositions.map(() => new THREE.Vector3());
  let energy = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i+1; j < n; j++) {
      const diff = repPositions[i].clone().sub(repPositions[j]);
      const d2 = diff.lengthSq();
      energy += 1 / Math.sqrt(d2);
      const f = diff.divideScalar(d2 * Math.sqrt(d2) + 1e-8);
      forces[i].add(f);
      forces[j].sub(f);
    }
  }
  thomsonEnergy = energy;
  repPositions.forEach((rp, i) => {
    rp.addScaledVector(forces[i], 0.0008);
    rp.normalize();
  });
  let k = 0;
  repGroup.children.forEach(m => {
    m.position.copy(repPositions[Math.floor(k/2)]);
    k++;
  });
  updateScene();
}

function reset() {
  selected.length = 0;
  repPositions.length = 0;
  thomsonActive = false;
  clearTimeout(autoplayTimer);
  autoplayTimer = null;
  document.getElementById('alg-btn-autoplay').textContent = 'Auto-play FPS';
  document.getElementById('alg-btn-thomson').textContent = 'Thomson Mode';
  document.getElementById('alg-info-energy').textContent = '—';
  while (repGroup.children.length) repGroup.remove(repGroup.children[0]);
  while (lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
  while (flashGroup.children.length) flashGroup.remove(flashGroup.children[0]);
  updateInfo();
}

function setStep(s) {
  currentStep = Math.max(0, Math.min(4, s));
  document.querySelectorAll('#algorithm .alg-step').forEach((el, i) => el.classList.toggle('active', i === currentStep));
  document.getElementById('alg-step-indicator').textContent = (currentStep+1) + ' / 5';
  const labels = ['OPH SKETCH','SPHERE PROJECTION','HNSW INDEX','FARTHEST POINT SAMPLING','THOMSON MERGE'];
  document.getElementById('alg-canvas-label').textContent = labels[currentStep];
  const inFPS = currentStep === 3;
  const inThomson = currentStep === 4;
  document.getElementById('alg-btn-action').style.display = (inFPS || inThomson) ? '' : 'none';
  document.getElementById('alg-btn-action').textContent = inThomson ? '⚡ Thomson Step' : '▶ Add Representative';
  document.getElementById('alg-btn-autoplay').style.display = inFPS ? '' : 'none';
  document.getElementById('alg-btn-thomson').style.display = inThomson ? '' : 'none';
}

document.getElementById('alg-btn-prev').onclick = () => setStep(currentStep - 1);
document.getElementById('alg-btn-next').onclick = () => setStep(currentStep + 1);
document.getElementById('alg-btn-reset').onclick = reset;

document.getElementById('alg-btn-action').onclick = () => {
  if (currentStep === 3) fpsStep();
  else if (currentStep === 4) { thomsonActive = false; thomsonStep(); }
};

document.getElementById('alg-btn-autoplay').onclick = function() {
  if (autoplayTimer) {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
    this.textContent = 'Auto-play FPS';
  } else {
    this.textContent = '⏸ Pause';
    autoplayTimer = setInterval(() => {
      if (selected.length >= N_GENOMES) {
        clearInterval(autoplayTimer); autoplayTimer = null;
        document.getElementById('alg-btn-autoplay').textContent = 'Auto-play FPS';
        return;
      }
      fpsStep();
    }, 220);
  }
};

document.getElementById('alg-btn-thomson').onclick = function() {
  thomsonActive = !thomsonActive;
  this.textContent = thomsonActive ? '⏸ Pause Thomson' : 'Thomson Mode';
};

document.querySelectorAll('#algorithm .alg-step').forEach((el, i) => {
  el.addEventListener('click', () => setStep(i));
});

setStep(0);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (thomsonActive) thomsonStep();
  renderer.render(scene, camera);
}
animate();

})();
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
}

} // namespace derep
