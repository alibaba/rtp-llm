
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('report-data').textContent);

  var MONO = "'IBM Plex Mono','PingFang SC',monospace";
  var C = {
    hot: '#3ddc97', cool: '#5db8d9', warn: '#ff7a59', amber: '#ffb454', gold: '#e8c468',
    ink: '#e4f0e9', dim: '#8ea599', faint: '#5b6e64', grid: 'rgba(212,240,229,0.07)',
    axis: '#283930'
  };
  var CAP_COLORS = ['#1d5c4a', '#2fa87e', '#4fd39e', '#e8c468', '#ffb454', '#ff9160', '#ff7a59'];
  var LABELS = DATA.labels;
  var ARM_NAMES = { hi: 'hi cap=1e8', finite: 'finite cap=500', zero: 'zero cap=0' };

  Chart.defaults.font.family = MONO;
  Chart.defaults.font.size = 10.5;
  Chart.defaults.color = C.dim;

  function fmtCap(c) { return c === 100000000 ? '∞' : String(c); }

  /* ---------- 公共配置（v2 复用） ---------- */
  function tooltipOpts() {
    return {
      backgroundColor: '#0f1512', borderColor: C.axis, borderWidth: 1,
      titleColor: C.ink, bodyColor: C.dim, padding: 10,
      titleFont: { family: MONO, size: 11 }, bodyFont: { family: MONO, size: 11 },
      displayColors: true, boxWidth: 8, boxHeight: 8, boxPadding: 4
    };
  }
  function xCatScale() {
    return {
      grid: { color: C.grid, drawTicks: false }, border: { color: C.axis },
      ticks: { color: C.dim, font: { family: MONO, size: 10.5 }, maxRotation: 0, autoSkip: false },
      title: { display: true, text: 'cap 档位（maxExtraTtftMs · 0 = arm 关闭 · ∞ = 1e8）', color: C.faint, font: { family: MONO, size: 10 } }
    };
  }
  function yScale(opts) {
    return {
      min: opts.min, max: opts.max,
      grid: { color: C.grid }, border: { display: false },
      ticks: { color: opts.color || C.dim, font: { family: MONO, size: 10 }, stepSize: opts.stepSize, callback: opts.fmt },
      title: { display: !!opts.title, text: opts.title || '', color: opts.color || C.faint, font: { family: MONO, size: 10 } }
    };
  }
  function baseOpts(extra) {
    var o = {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      interaction: { mode: 'nearest', intersect: true },
      plugins: { legend: { display: false }, tooltip: tooltipOpts() }
    };
    if (extra) { for (var k in extra) { o[k] = extra[k]; } }
    return o;
  }
  function band(top, bottom, color, axisId) {
    return [
      { data: top, yAxisID: axisId, borderColor: 'transparent', backgroundColor: color,
        borderWidth: 0, pointRadius: 0, fill: '+1', _band: true, label: 'band' },
      { data: bottom, yAxisID: axisId, borderColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, _band: true, label: 'band' }
    ];
  }
  var seedStyles = [
    { pointStyle: 'circle', dash: [] },
    { pointStyle: 'triangle', dash: [2, 3] },
    { pointStyle: 'rect', dash: [5, 3] }
  ];
  function seedLine(label, arr, color, axisId, si) {
    return {
      label: label, data: arr, yAxisID: axisId,
      borderColor: color, borderWidth: 1.2, borderDash: seedStyles[si].dash,
      pointStyle: seedStyles[si].pointStyle, pointRadius: 2.6, pointBackgroundColor: color,
      pointBorderColor: color, tension: 0.25
    };
  }

  /* ---------- 自定义插件 ---------- */
  var phaseLine = {
    id: 'phaseLine',
    afterDatasetsDraw: function (chart, args, opts) {
      var xs = chart.scales.x, area = chart.chartArea, ctx = chart.ctx;
      var x0 = xs.getPixelForValue(opts.between[0]);
      var x1 = xs.getPixelForValue(opts.between[1]);
      var xm = (x0 + x1) / 2;
      ctx.save();
      ctx.strokeStyle = C.amber; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(xm, area.top + 18); ctx.lineTo(xm, area.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '600 10px ' + MONO; ctx.fillStyle = C.amber; ctx.textAlign = 'center';
      ctx.fillText(opts.text || '相变 250 → 500', xm, area.top + 12);
      ctx.restore();
    }
  };
  var h1Band = {
    id: 'h1Band',
    beforeDatasetsDraw: function (chart, args, opts) {
      var ys = chart.scales.y, area = chart.chartArea, ctx = chart.ctx;
      var yT = ys.getPixelForValue(opts.hi), yB = ys.getPixelForValue(opts.lo);
      ctx.save();
      ctx.fillStyle = 'rgba(232,196,104,0.07)';
      ctx.fillRect(area.left, yT, area.right - area.left, yB - yT);
      ctx.strokeStyle = 'rgba(232,196,104,0.45)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(area.left, yT); ctx.lineTo(area.right, yT);
      ctx.moveTo(area.left, yB); ctx.lineTo(area.right, yB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '500 10px ' + MONO; ctx.fillStyle = 'rgba(232,196,104,0.85)';
      ctx.textAlign = 'center';
      ctx.fillText('H1 门限带 ±1%（' + opts.lo.toFixed(4) + ' – ' + opts.hi.toFixed(4) + ' s）', (area.left + area.right) / 2, yT + 14);
      ctx.restore();
    }
  };
  var evictBand = {
    id: 'evictBand',
    beforeDatasetsDraw: function (chart, args, opts) {
      var ys = chart.scales.y, area = chart.chartArea, ctx = chart.ctx;
      var yT = ys.getPixelForValue(opts.hi), yB = ys.getPixelForValue(opts.lo);
      ctx.save();
      ctx.fillStyle = 'rgba(61,220,151,0.09)';
      ctx.fillRect(area.left, yT, area.right - area.left, yB - yT);
      ctx.strokeStyle = 'rgba(61,220,151,0.42)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(area.left, yT); ctx.lineTo(area.right, yT);
      ctx.moveTo(area.left, yB); ctx.lineTo(area.right, yB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '600 10px ' + MONO; ctx.fillStyle = 'rgba(61,220,151,0.9)';
      ctx.textAlign = 'right';
      ctx.fillText('健康带 ' + opts.lo + '–' + opts.hi + 's', area.right - 6, (yT + yB) / 2 + 3);
      ctx.restore();
    }
  };
  /* X2：P95/P99 标线 + 固定档违约率点标签 */
  var mCalLines = {
    id: 'mCalLines',
    afterDatasetsDraw: function (chart, args, opts) {
      var xs = chart.scales.x, area = chart.chartArea, ctx = chart.ctx;
      ctx.save();
      opts.lines.forEach(function (L) {
        var px = xs.getPixelForValue(L.x);
        ctx.strokeStyle = L.color; ctx.lineWidth = 1.4; ctx.setLineDash(L.dash || []);
        ctx.beginPath(); ctx.moveTo(px, area.top + 14); ctx.lineTo(px, area.bottom); ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '600 10px ' + MONO; ctx.fillStyle = L.color; ctx.textAlign = L.align || 'left';
        ctx.fillText(L.text, px + (L.dx === undefined ? 4 : L.dx), area.top + 10);
      });
      var y1 = chart.scales.y1, meta = chart.getDatasetMeta(1);
      ctx.font = '700 10px ' + MONO; ctx.textAlign = 'center';
      meta.data.forEach(function (el, i) {
        var pt = chart.data.datasets[1].data[i];
        var v = (pt && typeof pt === 'object') ? pt.y : pt;
        if (!el || v === null || v === undefined) { return; }
        ctx.fillStyle = C.gold;
        ctx.fillText(v.toFixed(2) + '%', el.x, y1.getPixelForValue(v) - 8);
      });
      ctx.restore();
    }
  };
  /* X3/X1：柱顶数值标签（多数据集版） */
  var barValueLabels = {
    id: 'barValueLabels',
    afterDatasetsDraw: function (chart, args, opts) {
      var ctx = chart.ctx, ys = chart.scales.y;
      ctx.save();
      ctx.font = '700 10.5px ' + MONO;
      ctx.textAlign = 'center';
      chart.data.datasets.forEach(function (ds, di) {
        if (ds.type === 'line') { return; }
        var meta = chart.getDatasetMeta(di);
        var colors = Array.isArray(ds.borderColor) ? ds.borderColor : [ds.borderColor];
        ds.data.forEach(function (raw, i) {
          var el = meta.data[i];
          var v = (raw && typeof raw === 'object') ? raw.y : raw;
          if (!el || v === null || v === undefined) { return; }
          ctx.fillStyle = colors.length > 1 ? colors[i] : colors[0];
          ctx.fillText(v.toFixed(opts.dec === undefined ? 2 : opts.dec) + '%', el.x, ys.getPixelForValue(v) - 7);
        });
      });
      ctx.restore();
    }
  };
  /* X3 右：TTFT 柱顶 ms 标签 + 归因线点标签 */
  var ttftBarLabels = {
    id: 'ttftBarLabels',
    afterDatasetsDraw: function (chart, args, opts) {
      var ctx = chart.ctx, ys = chart.scales.y, y1 = chart.scales.y1;
      ctx.save();
      ctx.textAlign = 'center';
      var m0 = chart.getDatasetMeta(0);
      chart.data.datasets[0].data.forEach(function (v, i) {
        ctx.font = '700 11px ' + MONO;
        ctx.fillStyle = chart.data.datasets[0].borderColor[i];
        ctx.fillText(v.toFixed(0) + 'ms', m0.data[i].x, ys.getPixelForValue(v) - 8);
      });
      var m1 = chart.getDatasetMeta(1);
      chart.data.datasets[1].data.forEach(function (v, i) {
        ctx.font = '700 10px ' + MONO;
        ctx.fillStyle = C.gold;
        ctx.fillText(v.toFixed(2) + '%', m1.data[i].x, y1.getPixelForValue(v) - 10);
      });
      ctx.restore();
    }
  };
  /* X4：一致性对角参考线 + v1 臂标签 */
  var diagRef = {
    id: 'diagRef',
    beforeDatasetsDraw: function (chart, args, opts) {
      var xs = chart.scales.x, ys = chart.scales.y, ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = 'rgba(142,165,153,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(xs.getPixelForValue(0), ys.getPixelForValue(0));
      ctx.lineTo(xs.getPixelForValue(opts.to), ys.getPixelForValue(opts.to));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '500 9.5px ' + MONO; ctx.fillStyle = C.faint; ctx.textAlign = 'left';
      ctx.fillText('两口径一致参考线 y = x', xs.getPixelForValue(opts.to) - 118, ys.getPixelForValue(opts.to) + 14);
      ctx.restore();
    }
  };
  var armLabels = {
    id: 'armLabels',
    afterDatasetsDraw: function (chart, args, opts) {
      var xs = chart.scales.x, ys = chart.scales.y, ctx = chart.ctx;
      var meta = chart.getDatasetMeta(1);
      ctx.save();
      ctx.font = '700 10px ' + MONO; ctx.textAlign = 'center'; ctx.fillStyle = C.ink;
      meta.data.forEach(function (el, i) {
        var p = chart.data.datasets[1].data[i];
        ctx.fillText(opts.names[i], el.x, ys.getPixelForValue(p.y) - 13);
      });
      ctx.restore();
    }
  };

  /* ================= P1' 相变重判（c11） ================= */
  new Chart(document.getElementById('c11').getContext('2d'), {
    type: 'line',
    data: {
      labels: LABELS,
      datasets: [
        {
          label: '旧口径有害占比', data: DATA.oldPct, yAxisID: 'y', stepped: 'middle',
          borderColor: C.warn, backgroundColor: 'rgba(255,122,89,0.10)',
          borderWidth: 2.4, pointRadius: 4.5, pointHoverRadius: 7,
          pointBackgroundColor: C.warn, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4
        },
        {
          label: '新口径 SLO 违约率', data: DATA.newPct, yAxisID: 'y',
          borderColor: C.hot, backgroundColor: 'rgba(61,220,151,0.10)',
          borderWidth: 2.4, pointRadius: 4.5, pointHoverRadius: 7,
          pointBackgroundColor: C.hot, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4,
          tension: 0.25
        },
        {
          label: 'bg 命中率', data: DATA.bgMean, yAxisID: 'y1',
          borderColor: C.cool, borderWidth: 1.6, borderDash: [5, 3], pointRadius: 3,
          pointBackgroundColor: C.cool, pointBorderColor: '#0b0f0d', pointBorderWidth: 1, tension: 0.25
        }
      ]
    },
    options: baseOpts({
      animation: { duration: 1000, easing: 'easeOutQuart' },
      scales: {
        x: xCatScale(),
        y: yScale({ min: 0, max: 40, color: C.warn, title: '占比（%）· 旧有害 / 新违约同轴对照', stepSize: 10, fmt: function (v) { return v + '%'; } }),
        y1: { position: 'right', min: 0, max: 0.75, grid: { drawOnChartArea: false }, border: { display: false },
              ticks: { color: C.cool, font: { family: MONO, size: 10 }, callback: function (v) { return v.toFixed(2); } },
              title: { display: true, text: 'bg 命中率', color: C.cool, font: { family: MONO, size: 10 } } }
      },
      plugins: {
        legend: { display: false },
        tooltip: Object.assign(tooltipOpts(), {
          mode: 'index', intersect: false,
          callbacks: {
            title: function (items) { return 'cap ' + items[0].label; },
            label: function (item) {
              if (item.dataset.yAxisID === 'y1') { return ' bg 命中率  ' + item.parsed.y.toFixed(3); }
              return (item.datasetIndex === 0 ? ' 旧有害占比  ' : ' 新 SLO 违约率  ') + item.parsed.y.toFixed(2) + '%';
            }
          }
        }),
        phaseLine: { between: [3, 4], text: '旧相变 = 阈值伪影（旧 0→33.3% · 新 5.06→0.85% ↓ · hot ×1.17）' }
      }
    }),
    plugins: [phaseLine]
  });

  /* ================= X3 左（c8）：三臂旧 vs 新 双系列柱 ================= */
  var V1 = DATA.v1arms;
  var armLabelsX = V1.arms.map(function (a) { return ARM_NAMES[a]; });
  new Chart(document.getElementById('c8').getContext('2d'), {
    type: 'bar',
    data: {
      labels: armLabelsX,
      datasets: [
        {
          label: '旧 delta>100ms（%）', data: V1.v.map(function (d) { return d.oldPct; }),
          backgroundColor: 'rgba(255,122,89,0.82)', borderColor: C.warn, borderWidth: 1.4,
          barPercentage: 0.72, categoryPercentage: 0.66, borderRadius: 2
        },
        {
          label: '新 SLO 违约 M_cal P95（%）', data: V1.v.map(function (d) { return d.newPct; }),
          backgroundColor: 'rgba(61,220,151,0.8)', borderColor: C.hot, borderWidth: 1.4,
          barPercentage: 0.72, categoryPercentage: 0.66, borderRadius: 2
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: { grid: { display: false }, border: { color: C.axis },
             ticks: { color: C.dim, font: { family: MONO, size: 10 }, maxRotation: 0, autoSkip: false } },
        y: yScale({ min: 0, max: 76, color: C.dim, title: '占比（%）', stepSize: 20, fmt: function (v) { return v + '%'; } })
      },
      plugins: {
        legend: { display: false },
        barValueLabels: { dec: 2 },
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) { return 'v1 ' + items[0].label + ' · n=16230'; },
            label: function (item) { return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(2) + '%'; }
          }
        })
      }
    }),
    plugins: [barValueLabels]
  });

  /* ================= X3 右（c9）：TTFT p95 柱 × 归因违约线 ================= */
  new Chart(document.getElementById('c9').getContext('2d'), {
    data: {
      labels: armLabelsX,
      datasets: [
        {
          type: 'bar', label: '客户端 TTFT p95（ms）', data: V1.v.map(function (d) { return d.ttft; }),
          backgroundColor: ['rgba(61,220,151,0.85)', 'rgba(255,122,89,0.85)', 'rgba(255,180,84,0.85)'],
          borderColor: [C.hot, C.warn, C.amber], borderWidth: 1.4,
          barPercentage: 0.56, categoryPercentage: 0.7, borderRadius: 2, yAxisID: 'y'
        },
        {
          type: 'line', label: '亲和归因违约率（%）', data: V1.v.map(function (d) { return d.attrPct; }),
          borderColor: C.gold, borderWidth: 2, pointRadius: 4.5,
          pointBackgroundColor: C.gold, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4,
          tension: 0.2, yAxisID: 'y1'
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: { grid: { display: false }, border: { color: C.axis },
             ticks: { color: C.dim, font: { family: MONO, size: 10 }, maxRotation: 0, autoSkip: false } },
        y: yScale({ min: 0, max: 1150, color: C.dim, title: 'TTFT p95（ms）', stepSize: 250, fmt: function (v) { return String(v); } }),
        y1: { position: 'right', min: 0, max: 16, grid: { drawOnChartArea: false }, border: { display: false },
              ticks: { color: C.gold, font: { family: MONO, size: 10 }, stepSize: 4, callback: function (v) { return v + '%'; } },
              title: { display: true, text: '归因违约率（%）', color: C.gold, font: { family: MONO, size: 10 } } }
      },
      plugins: {
        legend: { display: false },
        ttftBarLabels: {},
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) { return 'v1 ' + items[0].label; },
            label: function (item) {
              return item.datasetIndex === 0
                ? ' TTFT p95  ' + item.parsed.y.toFixed(1) + ' ms'
                : ' 归因违约  ' + item.parsed.y.toFixed(2) + '%（反事实近似口径）';
            }
          }
        })
      }
    }),
    plugins: [ttftBarLabels]
  });

  /* ================= X2（c7）：r 分布直方图 + 固定档违约率 ================= */
  var H = DATA.hist, MCAL = DATA.mcal;
  var fixPts = DATA.fixM.x.map(function (x, i) { return { x: x, y: DATA.fixM.y[i] }; });
  new Chart(document.getElementById('c7').getContext('2d'), {
    data: {
      datasets: [
        {
          type: 'bar', label: 'r 分布（请求数/bin）',
          data: H.counts.map(function (c, i) { return { x: H.centers[i], y: c }; }), yAxisID: 'y',
          backgroundColor: 'rgba(61,220,151,0.5)', borderColor: 'rgba(61,220,151,0.8)', borderWidth: 0.6,
          barPercentage: 1.0, categoryPercentage: 1.0, borderRadius: 0
        },
        {
          type: 'line', label: '固定档 M 违约率（%）', data: fixPts, yAxisID: 'y1',
          borderColor: C.gold, borderWidth: 1.8, borderDash: [4, 3],
          pointStyle: 'rectRot', pointRadius: 5.5, pointHoverRadius: 8,
          pointBackgroundColor: C.gold, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.2,
          showLine: false
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: {
          type: 'linear', min: H.centers[0] - H.bw / 2, max: H.centers[H.centers.length - 1] + H.bw / 2,
          grid: { color: C.grid, drawTicks: false }, border: { color: C.axis },
          ticks: { color: C.dim, font: { family: MONO, size: 10 }, stepSize: 0.25, callback: function (v) { return v.toFixed(2); } },
          title: { display: true, text: 'r = 客户端 TTFT / prefill_ms（cap250 三 seed pooled n=8100 · bin 0.05）', color: C.faint, font: { family: MONO, size: 10 } }
        },
        y: yScale({ min: 0, color: C.hot, title: '请求数 / bin', fmt: function (v) { return String(v); } }),
        y1: { position: 'right', min: 0, max: 100, grid: { drawOnChartArea: false }, border: { display: false },
              ticks: { color: C.gold, font: { family: MONO, size: 10 }, stepSize: 25, callback: function (v) { return v + '%'; } },
              title: { display: true, text: '固定档 M 在黄金点的违约率（%）', color: C.gold, font: { family: MONO, size: 10 } } }
      },
      plugins: {
        legend: { display: false },
        mCalLines: { lines: [
          { x: MCAL.p95, color: C.warn, text: 'P95 = ' + MCAL.p95.toFixed(4) + '（M_cal）', dx: -118, align: 'left' },
          { x: MCAL.p99, color: C.amber, dash: [2, 3], text: 'P99 ' + MCAL.p99.toFixed(4), dx: 5 }
        ] },
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) {
              var p = items[0];
              return p.datasetIndex === 0
                ? 'r ∈ [' + (H.centers[p.dataIndex] - H.bw / 2).toFixed(2) + ', ' + (H.centers[p.dataIndex] + H.bw / 2).toFixed(2) + ')'
                : 'M = ' + p.parsed.x.toFixed(4);
            },
            label: function (item) {
              return item.datasetIndex === 0
                ? ' 请求数 ' + item.parsed.y
                : ' 黄金点违约率 ' + item.parsed.y.toFixed(2) + '%';
            }
          }
        })
      }
    }),
    plugins: [mCalLines]
  });

  /* ================= X4（c10）：新旧口径一致性散点 ================= */
  var gPts = DATA.caps.map(function (c, i) {
    return { x: DATA.oldPct[i], y: DATA.newPct[i], cap: c };
  });
  var vPts = V1.v.map(function (d, i) {
    return { x: d.oldPct, y: d.newPct, arm: V1.arms[i], ttft: d.ttft };
  });
  new Chart(document.getElementById('c10').getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'golden（cap 聚合）', data: gPts,
          backgroundColor: gPts.map(function (p) { return CAP_COLORS[DATA.caps.indexOf(p.cap)]; }),
          pointRadius: 6, pointHoverRadius: 9, pointBorderColor: '#0b0f0d', pointBorderWidth: 1
        },
        {
          label: 'v1 高压三臂', data: vPts,
          pointStyle: 'triangle', pointRadius: 9, pointHoverRadius: 12,
          backgroundColor: 'rgba(228,240,233,0.9)', borderColor: C.ink, borderWidth: 1.4
        }
      ]
    },
    options: baseOpts({
      animation: { duration: 900, easing: 'easeOutQuart' },
      scales: {
        x: {
          min: -2, max: 39,
          grid: { color: C.grid }, border: { color: C.axis },
          ticks: { color: C.warn, font: { family: MONO, size: 10 }, stepSize: 5, callback: function (v) { return v + '%'; } },
          title: { display: true, text: '旧口径：delta>100ms 有害占比（%）', color: C.warn, font: { family: MONO, size: 10 } }
        },
        y: {
          min: -3, max: 72,
          grid: { color: C.grid }, border: { display: false },
          ticks: { color: C.hot, font: { family: MONO, size: 10 }, stepSize: 10, callback: function (v) { return v + '%'; } },
          title: { display: true, text: '新口径：SLO 违约率 M_cal P95（%）', color: C.hot, font: { family: MONO, size: 10 } }
        }
      },
      plugins: {
        legend: { display: false },
        diagRef: { to: 38 },
        armLabels: { names: ['hi', 'finite', 'zero'] },
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) {
              var p = items[0].raw;
              return p.arm ? 'v1 ' + ARM_NAMES[p.arm] : 'golden cap ' + fmtCap(p.cap) + '（3 seed 聚合）';
            },
            label: function (item) {
              var p = item.raw;
              var s = ' 旧 ' + p.x.toFixed(2) + '% · 新 ' + p.y.toFixed(2) + '%';
              if (p.arm) { s += ' · TTFT p95 ' + p.ttft.toFixed(0) + 'ms'; }
              return s;
            }
          }
        })
      }
    }),
    plugins: [diagRef, armLabels]
  });

  /* ================= P2 seed 间分布（c2 · v2 原样） ================= */
  var hotBandTop = DATA.hotMean.map(function (m, i) { return m + DATA.hotStd[i]; });
  var hotBandBot = DATA.hotMean.map(function (m, i) { return m - DATA.hotStd[i]; });
  var evBandTop = DATA.evMean.map(function (m, i) { return m + DATA.evStd[i]; });
  var evBandBot = DATA.evMean.map(function (m, i) { return m - DATA.evStd[i]; });
  var p2ds = []
    .concat(band(evBandTop, evBandBot, 'rgba(93,184,217,0.14)', 'y1'))
    .concat(band(hotBandTop, hotBandBot, 'rgba(61,220,151,0.14)', 'y'))
    .concat([{
      label: 'ev_age mean', data: DATA.evMean, yAxisID: 'y1',
      borderColor: C.cool, borderWidth: 2.4, pointRadius: 4,
      pointBackgroundColor: C.cool, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.25
    }])
    .concat([{
      label: 'hot_hit mean', data: DATA.hotMean, yAxisID: 'y',
      borderColor: C.hot, borderWidth: 2.4, pointRadius: 4,
      pointBackgroundColor: C.hot, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.25
    }]);
  DATA.evSeed.forEach(function (arr, si) { p2ds.push(seedLine('s' + (si + 1) + ' ev', arr, 'rgba(93,184,217,0.55)', 'y1', si)); });
  DATA.hotSeed.forEach(function (arr, si) { p2ds.push(seedLine('s' + (si + 1) + ' hot', arr, 'rgba(61,220,151,0.55)', 'y', si)); });
  new Chart(document.getElementById('c2').getContext('2d'), {
    type: 'line',
    data: { labels: LABELS, datasets: p2ds },
    options: baseOpts({
      scales: {
        x: xCatScale(),
        y: yScale({ min: 0.45, max: 0.75, color: C.hot, title: 'hot_hit', fmt: function (v) { return v.toFixed(2); } }),
        y1: { position: 'right', min: 0, max: 600, grid: { drawOnChartArea: false }, border: { display: false },
              ticks: { color: C.cool, font: { family: MONO, size: 10 }, stepSize: 100 },
              title: { display: true, text: 'ev_age（s）', color: C.cool, font: { family: MONO, size: 10 } } }
      },
      plugins: {
        legend: { display: false },
        tooltip: Object.assign(tooltipOpts(), {
          filter: function (item) { return !item.dataset._band; },
          callbacks: {
            title: function (items) { return 'cap ' + (items[0] ? items[0].label : ''); },
            label: function (item) {
              return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(item.dataset.yAxisID === 'y' ? 3 : 1);
            }
          }
        })
      }
    })
  });

  /* ================= P3 TTFT 平坦线（c3 · v2 原样） ================= */
  var tBandTop = DATA.ttftMean.map(function (m, i) { return m + DATA.ttftStd[i]; });
  var tBandBot = DATA.ttftMean.map(function (m, i) { return m - DATA.ttftStd[i]; });
  var tSeeds = [];
  DATA.labels.forEach(function (lab, ci) {
    DATA.ttftSeed.forEach(function (arr) { tSeeds.push({ x: lab, y: arr[ci] }); });
  });
  new Chart(document.getElementById('c3').getContext('2d'), {
    type: 'line',
    data: {
      labels: LABELS,
      datasets: [
        { data: tBandTop, borderColor: 'transparent', backgroundColor: 'rgba(232,196,104,0.13)',
          borderWidth: 0, pointRadius: 0, fill: '+1', _band: true, label: 'band' },
        { data: tBandBot, borderColor: 'transparent', borderWidth: 0, pointRadius: 0, fill: false, _band: true, label: 'band' },
        {
          label: 'p95 mean', data: DATA.ttftMean,
          borderColor: C.gold, borderWidth: 2.4, pointRadius: 4.5,
          pointBackgroundColor: C.gold, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.2
        },
        {
          label: 'seed', data: tSeeds, showLine: false,
          pointRadius: 3, pointHoverRadius: 5, pointBackgroundColor: 'rgba(232,196,104,0.75)',
          pointBorderColor: 'transparent', clip: 6
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: xCatScale(),
        y: yScale({ min: 0.571, max: 0.585, color: C.gold, title: 'TTFT p95（s）', fmt: function (v) { return v.toFixed(3); } })
      },
      plugins: {
        legend: { display: false },
        h1Band: { lo: DATA.h1Band.lo, hi: DATA.h1Band.hi },
        tooltip: Object.assign(tooltipOpts(), {
          filter: function (item) { return !item.dataset._band; },
          callbacks: {
            title: function (items) { return 'cap ' + (items[0] ? items[0].label : ''); },
            label: function (item) { return ' ttft_p95  ' + item.parsed.y.toFixed(4) + ' s'; }
          }
        })
      }
    }),
    plugins: [h1Band]
  });

  /* ================= P5 驱逐健康带（c5 · v2 原样） ================= */
  var eBandTop = DATA.evMean.map(function (m, i) { return m + DATA.evStd[i]; });
  var eBandBot = DATA.evMean.map(function (m, i) { return m - DATA.evStd[i]; });
  var p5ds = band(eBandTop, eBandBot, 'rgba(93,184,217,0.14)', 'y')
    .concat([{
      label: 'ev_age mean', data: DATA.evMean,
      borderColor: C.cool, borderWidth: 2.4, pointRadius: 4.5,
      pointBackgroundColor: C.cool, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.25
    }]);
  DATA.evSeed.forEach(function (arr, si) { p5ds.push(seedLine('s' + (si + 1), arr, 'rgba(93,184,217,0.55)', 'y', si)); });
  new Chart(document.getElementById('c5').getContext('2d'), {
    type: 'line',
    data: { labels: LABELS, datasets: p5ds },
    options: baseOpts({
      scales: {
        x: xCatScale(),
        y: yScale({ min: 100, max: 600, color: C.cool, title: '驱逐年龄（s）· prefill-0', fmt: function (v) { return String(v); } })
      },
      plugins: {
        legend: { display: false },
        evictBand: { lo: DATA.evBand.lo, hi: DATA.evBand.hi },
        tooltip: Object.assign(tooltipOpts(), {
          filter: function (item) { return !item.dataset._band; },
          callbacks: {
            title: function (items) { return 'cap ' + (items[0] ? items[0].label : ''); },
            label: function (item) { return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(1) + ' s'; }
          }
        })
      }
    }),
    plugins: [evictBand]
  });

  /* ================= X1（c6）：有害判定双口径并列柱 ================= */
  var tri = DATA.tri;
  new Chart(document.getElementById('c6').getContext('2d'), {
    type: 'bar',
    data: {
      labels: [
        ['黄金点 cap=250', '摸测 3QPS 低压'],
        ['线上 V8 cap=∞', 'v1-hi 154QPS 高压'],
        ['线上 V8 cap=∞', '摸测 3QPS 低压']
      ],
      datasets: [
        {
          label: '旧 delta>100ms（%）',
          data: [tri.goldenHarmPct, tri.v1HarmPct, tri.sweepInfHarmPct],
          backgroundColor: 'rgba(255,122,89,0.82)', borderColor: C.warn, borderWidth: 1.4,
          barPercentage: 0.78, categoryPercentage: 0.68, borderRadius: 2
        },
        {
          label: '新 SLO 违约 M_cal P95（%）',
          data: [tri.goldenNewSloPct, tri.v1NewSloPct, tri.sweepInfNewSloPct],
          backgroundColor: 'rgba(61,220,151,0.8)', borderColor: C.hot, borderWidth: 1.4,
          barPercentage: 0.78, categoryPercentage: 0.68, borderRadius: 2
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: { grid: { display: false }, border: { color: C.axis },
             ticks: { color: C.dim, font: { family: MONO, size: 10 }, maxRotation: 0, autoSkip: false } },
        y: yScale({ min: 0, max: 40, color: C.warn, title: '占比（%）', stepSize: 10, fmt: function (v) { return v + '%'; } })
      },
      plugins: {
        legend: { display: false },
        barValueLabels: { dec: 2 },
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) { var l = items[0].label; return Array.isArray(l) ? l.join(' · ') : l; },
            label: function (item) { return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(2) + '%'; }
          }
        })
      }
    }),
    plugins: [barValueLabels]
  });

  /* ================= X4 色标 ================= */
  var cs = document.getElementById('capscale10');
  var html = '';
  DATA.caps.forEach(function (c, i) {
    html += '<span class="ci"><span class="dot" style="background:' + CAP_COLORS[i] + '"></span>cap ' + fmtCap(c) + '</span>';
  });
  html += '<span class="ci"><span class="tri" style="background:#e4f0e9"></span>v1 三臂（hi / finite / zero）</span>';
  html += '<span class="ci" style="color:#8ea599">虚线 = 两口径一致参考线 y=x · 点远离对角带 = 系统性背离</span>';
  cs.innerHTML = html;
})();
