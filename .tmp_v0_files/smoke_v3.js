'use strict';
const fs = require('fs');
const HTML = '/Users/wangziyi/code/codex-cache-balance-run/results/golden-sweep-20260911/chart-report-golden-v3.html';
const html = fs.readFileSync(HTML, 'utf8');

// ---- 从 v3 HTML 提取 report-data 与主内嵌 JS ----
const dataJson = html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)(?![^>]*\bid=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const js = scripts[scripts.length - 1];
fs.writeFileSync('/tmp/v3_extracted.js', js);
fs.writeFileSync('/tmp/v3_data.json', dataJson);

let fails = 0, total = 0;
function ok(cond, msg) { total++; if (cond) console.log('  [PASS] ' + msg); else { fails++; console.log('  [FAIL] ' + msg); } }
function near(v, w, t) { return v !== undefined && v !== null && Math.abs(v - w) <= (t === undefined ? 5e-3 : t); }

// ---- fake DOM / Chart ----
const els = {};
function mkCtx(id) {
  return { __id: id, save(){}, restore(){}, fillText(){}, beginPath(){}, arc(){}, stroke(){}, fill(){},
    moveTo(){}, lineTo(){}, setLineDash(){}, fillRect(){},
    fillStyle:null, strokeStyle:null, lineWidth:null, font:null, textAlign:null };
}
function el(id) {
  if (!els[id]) {
    if (id === 'report-data') els[id] = { textContent: dataJson };
    else { const ctx = mkCtx(id); els[id] = { getContext(){ return ctx; }, __id: id, innerHTML: '' }; }
  }
  return els[id];
}
global.document = { getElementById: el };
const CHARTS = {};
global.Chart = function Chart(elc, cfg) { CHARTS[elc.__id] = cfg; };
global.Chart.defaults = { font: {}, color: null };

eval(js);

const DATA = JSON.parse(dataJson);

// ---- 图表实例 ----
console.log('== 图表实例 ==');
const IDS = ['c11','c8','c9','c7','c10','c2','c3','c5','c6'];
ok(Object.keys(CHARTS).length === IDS.length, 'Chart 实例共 ' + IDS.length + ' 个，实际 ' + Object.keys(CHARTS).length);
IDS.forEach(id => ok(!!CHARTS[id], '图 ' + id + ' 已构造'));
ok(CHARTS.c11.type === 'line', 'c11 = line（P1′ 相变重判）');
ok(CHARTS.c8.type === 'bar', 'c8 = bar（X3 双系列柱）');
ok(CHARTS.c10.type === 'scatter', 'c10 = scatter（X4 一致性散点）');
ok(!CHARTS.c9.type && CHARTS.c9.data.datasets[0].type === 'bar' && CHARTS.c9.data.datasets[1].type === 'line', 'c9 = 混合（柱 TTFT + 线归因）');
ok(!CHARTS.c7.type && CHARTS.c7.data.datasets[0].type === 'bar', 'c7 = 混合（直方图柱 + 固定档点）');

// ---- 核心反转数字（用户验收指定：29.88/64.98/18.76/0.85/5.06/2.6397）----
console.log('== 核心反转数字 ==');
const V = DATA.v1arms.v;
ok(near(V[0].newPct, 29.8829, 1e-3), 'v1 hi 新 SLO 违约 = 29.88%（三臂最少），实际 ' + V[0].newPct);
ok(near(V[2].newPct, 64.9784, 1e-3), 'v1 zero 新 SLO 违约 = 64.98%（三臂最多），实际 ' + V[2].newPct);
ok(near(V[0].oldPct, 18.7554, 1e-3), 'v1 hi 旧 delta>100 = 18.76%（旧判最有害），实际 ' + V[0].oldPct);
ok(near(DATA.phase.newHi, 0.85, 5.1e-3), '相变新口径 cap500 = 0.85%，实际 ' + DATA.phase.newHi);
ok(near(DATA.phase.newLo, 5.06, 5.1e-3), '相变新口径 cap250 = 5.06%，实际 ' + DATA.phase.newLo);
ok(DATA.mcal.p95 === 2.6397, 'M_cal P95 = 2.6397，实际 ' + DATA.mcal.p95);
ok(DATA.mcal.p99 === 2.6528 && near(DATA.mcal.mp95, 1.0398, 1e-9), 'M_cal P99 = 2.6528 · M′_cal P95 = 1.0398');
ok(DATA.mcal.n === 8100, 'pooled n = 8100');

// ---- X3（c8/c9）反转面板 ----
console.log('== X3 三臂反转（c8/c9）==');
const c8 = CHARTS.c8;
ok(c8.data.datasets.length === 2, 'c8 双系列（旧 vs 新）');
ok(c8.data.datasets[0].data.every((v,i) => near(v, [18.7554,5.5453,0][i], 1e-3)), 'c8 旧口径 = [18.76, 5.55, 0]，实际 ' + JSON.stringify(c8.data.datasets[0].data));
ok(c8.data.datasets[1].data.every((v,i) => near(v, [29.8829,59.809,64.9784][i], 1e-3)), 'c8 新口径 = [29.88, 59.81, 64.98]，实际 ' + JSON.stringify(c8.data.datasets[1].data));
ok(c8.data.datasets[0].borderColor === '#ff7a59' && c8.data.datasets[1].borderColor === '#3ddc97', 'c8 旧 warn / 新 hot 配色');
ok(c8.plugins.some(p => p.id === 'barValueLabels'), 'c8 柱顶数值插件');
const c9 = CHARTS.c9;
ok(c9.data.datasets[0].data.every((v,i) => near(v, [681.6,958.39,874.51][i], 0.05)), 'c9 TTFT p95 = [682, 958, 874]ms，实际 ' + JSON.stringify(c9.data.datasets[0].data));
ok(c9.data.datasets[1].data.every((v,i) => near(v, [12.7665,3.7338,0][i], 1e-3)), 'c9 归因违约 = [12.77, 3.73, 0]%，实际 ' + JSON.stringify(c9.data.datasets[1].data));
ok(c9.data.datasets[1].yAxisID === 'y1' && c9.options.scales.y1.max === 16, 'c9 归因线走右轴 0-16%');
ok(c9.plugins.some(p => p.id === 'ttftBarLabels'), 'c9 柱顶 ms + 线点标签插件');

// ---- X2（c7）M 校准 ----
console.log('== X2 M 校准（c7）==');
const c7 = CHARTS.c7;
const H = DATA.hist;
ok(H.counts.reduce((a,b) => a+b, 0) === 8100, '直方图计数和 = 8100');
ok(H.centers.length === H.counts.length && H.bw === 0.05, '直方图 bin 宽 0.05 · centers/counts 对齐（' + H.centers.length + ' bin）');
ok(c7.data.datasets[0].data.length === H.counts.length, 'c7 柱数 = bin 数');
ok(c7.options.scales.x.type === 'linear', 'c7 x 轴 linear（r 值域）');
const ml = c7.options.plugins.mCalLines.lines;
ok(ml.length === 2 && ml[0].x === 2.6397 && near(ml[1].x, 2.6528, 1e-9), 'c7 P95=2.6397 / P99=2.6528 标线');
ok(JSON.stringify(DATA.fixM.x) === '[1.5,2,2.6397,3]', 'c7 固定档 x = [1.5, 2.0, M_cal, 3.0]，实际 ' + JSON.stringify(DATA.fixM.x));
ok(near(DATA.fixM.y[0], 50.04, 0.06) && near(DATA.fixM.y[1], 50.04, 0.06), '固定档 1.5/2.0 黄金点违约 ≈50.04%');
ok(near(DATA.fixM.y[3], 0, 0.01), '固定档 3.0 黄金点违约 = 0%（全漏）');
ok(DATA.fixM.y[2] > 4 && DATA.fixM.y[2] < 6, 'M_cal 违约 ≈5%（P95 tautology），实际 ' + DATA.fixM.y[2]);

// ---- X4（c10）一致性散点 ----
console.log('== X4 一致性散点（c10）==');
const c10 = CHARTS.c10;
ok(c10.data.datasets[0].data.length === 7, 'c10 golden 7 档点');
ok(c10.data.datasets[0].data[4].x > 33 && near(c10.data.datasets[0].data[4].y, 0.8519, 1e-3), 'cap500 点 = (33.30, 0.85) 旧高新低');
ok(c10.data.datasets[1].data.length === 3 && near(c10.data.datasets[1].data[0].y, 29.8829, 1e-3), 'c10 v1 三臂 ▲ · hi y=29.88');
ok(c10.plugins.some(p => p.id === 'diagRef') && c10.plugins.some(p => p.id === 'armLabels'), 'c10 对角参考线 + 臂标签插件');
ok(els.capscale10.innerHTML.indexOf('v1 三臂') >= 0, 'X4 色标 DOM 渲染');

// ---- P1'（c11）相变重判 ----
console.log('== P1′ 相变重判（c11）==');
const c11 = CHARTS.c11;
ok(c11.data.datasets[0].data.every((v,i) => near(v, [0,0,0,0,33.2963,33.3086,33.2963][i], 1e-3)), 'c11 旧口径阶梯 = [0×4, 33.3×3]，实际 ' + JSON.stringify(c11.data.datasets[0].data));
ok(c11.data.datasets[1].data.every((v,i) => near(v, DATA.newPct[i], 1e-9)) && near(c11.data.datasets[1].data[3], 5.0617, 1e-3) && near(c11.data.datasets[1].data[4], 0.8519, 1e-3), 'c11 新口径违约 250→500 = 5.06→0.85%');
ok(c11.data.datasets[0].stepped === 'middle', 'c11 旧口径阶梯线（stepped）');
ok(c11.options.plugins.phaseLine.text.indexOf('阈值伪影') >= 0, 'c11 相变线标注「阈值伪影」');
ok(c11.data.datasets[2].data.every((v,i) => near(v, DATA.bgMean[i], 1e-9)), 'c11 bg 命中线（右轴）= v2 bgMean');

// ---- X1（c6）双口径并列 ----
console.log('== X1 双口径（c6）==');
const c6 = CHARTS.c6;
ok(c6.data.datasets.length === 2, 'c6 双系列（旧/新并列）');
ok(c6.data.datasets[0].data.every((v,i) => near(v, [0,20.85,33.3][i], 0.02)), 'c6 旧口径 = [0, 20.85, 33.3]，实际 ' + JSON.stringify(c6.data.datasets[0].data));
ok(near(c6.data.datasets[1].data[1], 29.8829, 1e-3) && near(c6.data.datasets[1].data[2], 0.7284, 1e-3) && near(c6.data.datasets[1].data[0], 5.0617, 1e-3), 'c6 新口径 = [5.06, 29.88, 0.73]，实际 ' + JSON.stringify(c6.data.datasets[1].data));
ok(c6.data.datasets[0].borderColor === '#ff7a59' && c6.data.datasets[1].borderColor === '#3ddc97', 'c6 旧 warn / 新 hot 配色');

// ---- v2 复用图（c2/c3/c5）回归 ----
console.log('== v2 复用图回归 ==');
ok(near(CHARTS.c3.options.plugins.h1Band.lo, 0.57295062, 1e-6) && near(CHARTS.c3.options.plugins.h1Band.hi, 0.58452538, 1e-6), 'c3 H1 带 0.57295–0.58453（同 v2）');
ok(CHARTS.c5.options.plugins.evictBand.lo === 200 && CHARTS.c5.options.plugins.evictBand.hi === 300, 'c5 健康带 200-300s（同 v2）');
ok(CHARTS.c2.data.datasets.length === 12, 'c2 数据集 12 个（band 4 + mean 2 + seed 6），实际 ' + CHARTS.c2.data.datasets.length);

// ---- HTML 文本验收 ----
console.log('== HTML 文本 ==');
[
  ['2.6397', 'M_cal 主口径'], ['29.88%', 'hi 新违约'], ['64.98%', 'zero 新违约'],
  ['18.76%', 'hi 旧有害'], ['0.85%', '相变新 cap500'], ['5.06%', '相变新 cap250'],
  ['12.77%', 'hi 归因违约'], ['682', 'hi TTFT p95'], ['958', 'finite TTFT'], ['874', 'zero TTFT'],
  ['50.04%', '固定档误判 50.04%'], ['1.0398', 'M′_cal 敏感性'],
  ['旧相变 = 阈值伪影', 'P1′ 结论标注'], ['方向完全判反', 'KPI 判反实锤'],
  ['HARMFUL', 'verdict 词汇'], ['compensated', 'compensated 标签'],
  ['SLO 违约率 &gt; 基线 seed 分布 P95', 'verdict 规则'],
  ['56700', 'join golden 全量'], ['16230', 'join v1 每臂'],
  ['P95 定义的 tautology', 'tautology 声明'], ['跨 regime 不做比值', '跨 regime 纪律'],
  ['反事实是近似', '归因近似声明'], ['12.92%', '比例式对照口径'],
  ['slo-analysis.json', '数据源 slo-analysis'], ['RUN-slo.md', '口径说明文档'], ['/tmp/slo-golden', 'staging 溯源'],
  ['run 20260911_101402 @ host111', 'run 溯源'],
  ['[SLO]', '来源缩写 SLO'], ['[v1-hi]', '来源缩写 v1-hi'], ['[摸测]', '来源缩写 摸测'],
  ['三方对照 — 线上 V8 ∞ · 旧版 os30 · 黄金点 250', 'X1 面板标题'],
  ['有害判定（旧 delta&gt;100ms / 新 SLO 违约）', 'X1 双口径行'],
  ['class="x1grid"', 'X1 双栏栅格类'], ['class="subgrid"', 'X3 双栏栅格类'], ['minmax(0,', 'minmax(0,) 防溢出'],
  ['class="tblwrap"', '宽表滚动容器'],
].forEach(([kw, name]) => ok(html.indexOf(kw) >= 0, '文本含「' + name + '」：' + kw));
ok(html.indexOf('@@') < 0, '无未替换占位符 @@');
ok(html.indexOf('<canvas id="c4"') < 0 && html.indexOf('Pareto') < 0, 'v2 的 P4 Pareto（旧口径有害轴）已按 v3 面板结构移除');
ok(/v2 报告保留不动|复用 v2 排版体系/.test(html), '页脚注明复用 v2 排版 · v2 保留');

console.log(fails === 0 ? '\n[SMOKE ALL PASS] ' + total + ' 项断言全过' : '\n[SMOKE FAIL x' + fails + '] / ' + total);
process.exit(fails === 0 ? 0 : 1);
