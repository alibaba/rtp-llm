'use strict';
const fs = require('fs');
const HTML = '/Users/wangziyi/code/codex-cache-balance-run/results/golden-sweep-20260911/chart-report-golden-v4.html';
const html = fs.readFileSync(HTML, 'utf8');

// ---- 从 v4 HTML 提取 report-data 与主内嵌 JS ----
const dataJson = html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)(?![^>]*\bid=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const js = scripts[scripts.length - 1];
fs.writeFileSync('/tmp/v4_extracted.js', js);
fs.writeFileSync('/tmp/v4_data.json', dataJson);

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
const IDS = ['c1','c2','c3','c4','c5','c6'];
ok(Object.keys(CHARTS).length === IDS.length, 'Chart 实例共 ' + IDS.length + ' 个，实际 ' + Object.keys(CHARTS).length);
IDS.forEach(id => ok(!!CHARTS[id], '图 ' + id + ' 已构造'));
ok(!CHARTS.c1.type && CHARTS.c1.data.datasets[0].type === 'bar', 'c1 = 混合（S1 直方图柱 + 固定档点）');
ok(CHARTS.c2.type === 'bar', 'c2 = bar（S2 违约率单系列柱）');
ok(!CHARTS.c3.type && CHARTS.c3.data.datasets[0].type === 'bar' && CHARTS.c3.data.datasets[1].type === 'line', 'c3 = 混合（柱 TTFT + 线归因）');
ok(CHARTS.c4.type === 'line', 'c4 = line（S3 矩阵违约聚合）');
ok(CHARTS.c5.type === 'line' && CHARTS.c6.type === 'line', 'c5/c6 = line（S5 场景健康）');

// ---- 核心新指标数字（用户验收指定：29.88/59.81/64.98/682/2.6397/0.85）----
console.log('== 核心新指标数字 ==');
const V = DATA.v1arms.v;
ok(near(V[0].newPct, 29.8829, 1e-3), 'hi SLO 违约 = 29.88%（三臂最少），实际 ' + V[0].newPct);
ok(near(V[1].newPct, 59.809, 1e-3), 'finite SLO 违约 = 59.81%，实际 ' + V[1].newPct);
ok(near(V[2].newPct, 64.9784, 1e-3), 'zero SLO 违约 = 64.98%（三臂最多），实际 ' + V[2].newPct);
ok(near(V[0].ttft, 681.6, 0.05), 'hi TTFT p95 = 682ms（三臂最低），实际 ' + V[0].ttft);
ok(DATA.mcal.p95 === 2.6397, 'M_cal P95 = 2.6397，实际 ' + DATA.mcal.p95);
ok(DATA.mcal.p99 === 2.6528 && near(DATA.mcal.mp95, 1.0398, 1e-9), 'M_cal P99 = 2.6528 · M′_cal P95 = 1.0398');
ok(DATA.mcal.n === 8100, 'pooled n = 8100');
ok(near(DATA.violMean[4], 0.8519, 1e-3), 'cap500 违约 = 0.85%，实际 ' + DATA.violMean[4]);
ok(near(DATA.violMean[3], 5.0617, 1e-3), 'cap250 违约 = 5.06%（校准点），实际 ' + DATA.violMean[3]);
ok(near(DATA.baseP95Pct, 5.4333, 1e-3), '基线 seed P95 门限 = 5.43%，实际 ' + DATA.baseP95Pct);

// ---- S2（c2/c3）高压三臂 ----
console.log('== S2 高压三臂（c2/c3）==');
const c2 = CHARTS.c2;
ok(c2.data.datasets.length === 1, 'c2 单系列（纯 SLO 违约，无历史对照系列）');
ok(c2.data.datasets[0].data.every((v,i) => near(v, [29.8829,59.809,64.9784][i], 1e-3)), 'c2 违约率 = [29.88, 59.81, 64.98]，实际 ' + JSON.stringify(c2.data.datasets[0].data));
ok(JSON.stringify(c2.data.datasets[0].borderColor) === '["#3ddc97","#ffb454","#ff7a59"]', 'c2 档内优劣配色 绿→琥珀→珊瑚');
ok(c2.options.scales.y.max === 76 && c2.options.scales.y.ticks.stepSize === 20, 'c2 y 轴 0-76 step 20');
ok(c2.plugins.some(p => p.id === 'barValueLabels'), 'c2 柱顶数值插件');
const c3 = CHARTS.c3;
ok(c3.data.datasets[0].data.every((v,i) => near(v, [681.6,958.39,874.51][i], 0.05)), 'c3 TTFT p95 = [682, 958, 875]ms，实际 ' + JSON.stringify(c3.data.datasets[0].data));
ok(c3.data.datasets[1].data.every((v,i) => near(v, [12.7665,3.7338,0][i], 1e-3)), 'c3 归因违约 = [12.77, 3.73, 0]%，实际 ' + JSON.stringify(c3.data.datasets[1].data));
ok(c3.data.datasets[1].yAxisID === 'y1' && c3.options.scales.y1.max === 16, 'c3 归因线走右轴 0-16%');
ok(c3.plugins.some(p => p.id === 'ttftBarLabels'), 'c3 柱顶 ms + 线点标签插件');

// ---- S1（c1）M 校准 ----
console.log('== S1 M 校准（c1）==');
const c1 = CHARTS.c1;
const H = DATA.hist;
ok(H.counts.reduce((a,b) => a+b, 0) === 8100, '直方图计数和 = 8100');
ok(H.centers.length === H.counts.length && H.bw === 0.05, '直方图 bin 宽 0.05 · centers/counts 对齐（' + H.centers.length + ' bin）');
ok(c1.data.datasets[0].data.length === H.counts.length && c1.data.datasets[0].data.every(p => typeof p === 'object' && 'x' in p && 'y' in p), 'c1 柱为 {x,y} 点对象（linear 轴必需）');
ok(c1.options.scales.x.type === 'linear', 'c1 x 轴 linear（r 值域）');
const ml = c1.options.plugins.mCalLines.lines;
ok(ml.length === 2 && ml[0].x === 2.6397 && near(ml[1].x, 2.6528, 1e-9), 'c1 P95=2.6397 / P99=2.6528 标线');
ok(JSON.stringify(DATA.fixM.x) === '[1.5,2,2.6397,3]', 'c1 固定档 x = [1.5, 2.0, M_cal, 3.0]，实际 ' + JSON.stringify(DATA.fixM.x));
ok(near(DATA.fixM.y[0], 50.04, 0.06) && near(DATA.fixM.y[1], 50.04, 0.06), '固定档 1.5/2.0 黄金点违约 ≈50.04%（M 必须校准注记）');
ok(near(DATA.fixM.y[3], 0, 0.01), '固定档 3.0 黄金点违约 = 0%（全漏）');
ok(DATA.fixM.y[2] > 4 && DATA.fixM.y[2] < 6, 'M_cal 违约 ≈5%（P95 tautology），实际 ' + DATA.fixM.y[2]);

// ---- S3（c4）矩阵违约聚合 ----
console.log('== S3 矩阵违约（c4）==');
const c4 = CHARTS.c4;
ok(DATA.violSeed.length === 3 && DATA.violSeed.every(a => a.length === 7), 'violSeed = 3 seed × 7 档');
ok(DATA.violSeed.every((a,s) => a.every((v,i) => near(DATA.violMean[i], (DATA.violSeed[0][i]+DATA.violSeed[1][i]+DATA.violSeed[2][i])/3, 0.02))), 'violMean ≈ 三 seed 均值');
ok(c4.data.datasets.length === 6, 'c4 数据集 6 个（band 2 + mean 1 + seed 3），实际 ' + c4.data.datasets.length);
ok(c4.data.datasets[0]._band && c4.data.datasets[1]._band, 'c4 前两个数据集 = seed 极差带');
ok(c4.data.datasets[2].data.every((v,i) => near(v, DATA.violMean[i], 1e-9)), 'c4 均值线 = violMean');
ok(near(c4.options.plugins.hLine.y, 5.4333, 1e-3), 'c4 基线门限线 y = 5.43%');
ok(c4.plugins.some(p => p.id === 'hLine'), 'c4 hLine 插件');
ok(c4.options.scales.y.max === 8 && c4.options.scales.y.ticks.stepSize === 2, 'c4 y 轴 0-8 step 2');

// ---- S5（c5/c6）场景健康 ----
console.log('== S5 场景健康（c5/c6）==');
ok(near(CHARTS.c5.options.plugins.h1Band.lo, 0.57295062, 1e-6) && near(CHARTS.c5.options.plugins.h1Band.hi, 0.58452538, 1e-6), 'c5 H1 带 0.57295–0.58453');
ok(CHARTS.c6.options.plugins.evictBand.lo === 200 && CHARTS.c6.options.plugins.evictBand.hi === 300, 'c6 健康带 200-300s');
ok(DATA.ttftMean.every(m => m > 0.576 && m < 0.580), 'TTFT 全档平坦 0.576–0.580s 区间，实际 ' + JSON.stringify(DATA.ttftMean));

// ---- DATA 净化（report-data 无历史字段）----
console.log('== DATA 净化 ==');
ok(!('oldPct' in DATA) && !('phase' in DATA) && !('tri' in DATA), 'DATA 无 oldPct/phase/tri 字段');
ok(DATA.v1arms.v.every(d => !('oldPct' in d)), 'v1arms 无 oldPct 字段');
ok(!/old_harmful|delta_gt100/.test(dataJson), 'report-data JSON 无历史口径键名');

// ---- HTML 文本验收 ----
console.log('== HTML 文本 ==');
[
  ['2.6397', 'M_cal 主口径'], ['29.88%', 'hi 违约'], ['59.81%', 'finite 违约'], ['64.98%', 'zero 违约'],
  ['0.85%', 'cap500 违约'], ['5.06%', 'cap250 校准点'], ['0.73%', 'cap1e8 违约'],
  ['12.77%', 'hi 归因违约'], ['3.73%', 'finite 归因违约'],
  ['682', 'hi TTFT p95'], ['958', 'finite TTFT'], ['875', 'zero TTFT'],
  ['50.04%', '固定档误判 50.04%'], ['1.0398', 'M′_cal 敏感性'], ['5.43%', '基线 seed P95 门限'],
  ['1.00096', 'hi 吞吐比'], ['0.99868', 'hi hot 命中比'],
  ['SLO_i = prefill_ms_i(选中候选) × M', 'S1 公式主口径'], ['违约_i = TTFT_i(客户端实测) &gt; SLO_i', 'S1 违约判定式'],
  ['综合最优', 'S2 结论'], ['HARMFUL', 'verdict 词汇'], ['compensated', 'compensated 标签'],
  ['SLO 违约率 &gt; 基线 seed 分布 P95', 'verdict/门禁规则'],
  ['56700', 'join golden 全量'], ['16230', 'join 高压每臂'],
  ['P95 定义的 tautology', 'tautology 声明'], ['跨 regime 不做比值', '跨 regime 纪律'],
  ['反事实是近似', '归因近似声明'], ['12.92%', '比例式对照口径'],
  ['有害判定（SLO 违约率 · M_cal P95）', 'X1 有害行单口径'],
  ['slo-analysis.json', '数据源 slo-analysis'], ['RUN-slo.md', '口径说明文档'], ['/tmp/slo-golden', 'staging 溯源'],
  ['run 20260911_101402 @ host111', 'run 溯源'],
  ['[SLO]', '来源缩写 SLO'], ['[v1-hi]', '来源缩写 v1-hi'], ['[摸测]', '来源缩写 摸测'],
  ['三方对照 — 线上 V8 ∞ · 旧版 os30 · 黄金点 250', 'X1 面板标题'],
  ['class="subgrid"', '双栏栅格类'], ['minmax(0,', 'minmax(0,) 防溢出'],
  ['class="tblwrap"', '宽表滚动容器'], ['class="formula"', 'S1 公式块'], ['class="deflist"', '定义列表'],
].forEach(([kw, name]) => ok(html.indexOf(kw) >= 0, '文本含「' + name + '」：' + kw));
ok(html.indexOf('@@') < 0, '无未替换占位符 @@');
[/delta&gt;100/, /旧口径/, /新口径/, /反转/, /判反/, /伪影/, /18\.76/, /新旧/, /相变/, /20\.85/, /一致性散点/].forEach(re =>
  ok(!re.test(html), '禁词零命中：' + re.source));
ok(/既有版本报告保留不动|复用既有排版体系/.test(html), '页脚注明复用既有排版 · 既有版本保留');

console.log(fails === 0 ? '\n[SMOKE ALL PASS] ' + total + ' 项断言全过' : '\n[SMOKE FAIL x' + fails + '] / ' + total);
process.exit(fails === 0 ? 0 : 1);
