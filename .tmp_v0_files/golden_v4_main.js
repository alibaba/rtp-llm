/* v4 主内嵌脚本 — 纯 SLO 违约指标体系（S1/S2/S3/S5 六图） */
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('report-data').textContent);

  var MONO = "'IBM Plex Mono','PingFang SC',monospace";
  var C = {
    hot: '#3ddc97', cool: '#5db8d9', warn: '#ff7a59', amber: '#ffb454', gold: '#e8c468',
    ink: '#e4f0e9', dim: '#8ea599', faint: '#5b6e64', grid: 'rgba(212,240,229,0.07)',
    axis: '#283930'
  };
  var LABELS = DATA.labels;
  var ARM_NAMES = { hi: 'hi cap=1e8', finite: 'finite cap=500', zero: 'zero cap=0' };

  Chart.defaults.font.family = MONO;
  Chart.defaults.font.size = 10.5;
  Chart.defaults.color = C.dim;

  /* ---------- 公共配置（沿用既有排版体系） ---------- */
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
  function xArmScale() {
    return {
      grid: { display: false }, border: { color: C.axis },
      ticks: { color: C.dim, font: { family: MONO, size: 10 }, maxRotation: 0, autoSkip: false }
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
  /* S1：P95/P99 标线 + 固定档违约率点标签 */
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
  /* S2：柱顶数值标签（多数据集版） */
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
  /* S2 右：TTFT 柱顶 ms 标签 + 归因线点标签 */
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
        ctx.fillText(Math.floor(v + 0.5) + 'ms', m0.data[i].x, ys.getPixelForValue(v) - 8);
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
  /* S3：基线 seed P95 水平门限线 */
  var hLine = {
    id: 'hLine',
    afterDatasetsDraw: function (chart, args, opts) {
      var ys = chart.scales.y, area = chart.chartArea, ctx = chart.ctx;
      var py = ys.getPixelForValue(opts.y);
      ctx.save();
      ctx.strokeStyle = opts.color; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(area.left, py); ctx.lineTo(area.right, py); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = '600 10px ' + MONO; ctx.fillStyle = opts.color; ctx.textAlign = 'right';
      ctx.fillText(opts.text, area.right - 6, py - 5);
      ctx.restore();
    }
  };
  /* S5：H1 门限带 / 驱逐健康带 */
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

  /* ================= S1（c1）：r 分布直方图 + 固定档违约率 ================= */
  var H = DATA.hist, MCAL = DATA.mcal;
  var fixPts = DATA.fixM.x.map(function (x, i) { return { x: x, y: DATA.fixM.y[i] }; });
  new Chart(document.getElementById('c1').getContext('2d'), {
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

  /* ================= S2 左（c2）：高压三臂 SLO 违约率柱 ================= */
  var V1 = DATA.v1arms;
  var armCatsX = V1.arms.map(function (a) { return ARM_NAMES[a]; });
  new Chart(document.getElementById('c2').getContext('2d'), {
    type: 'bar',
    data: {
      labels: armCatsX,
      datasets: [
        {
          label: 'SLO 违约率 M_cal P95（%）', data: V1.v.map(function (d) { return d.newPct; }),
          backgroundColor: ['rgba(61,220,151,0.8)', 'rgba(255,180,84,0.8)', 'rgba(255,122,89,0.8)'],
          borderColor: [C.hot, C.amber, C.warn], borderWidth: 1.4,
          barPercentage: 0.56, categoryPercentage: 0.7, borderRadius: 2
        }
      ]
    },
    options: baseOpts({
      scales: {
        x: xArmScale(),
        y: yScale({ min: 0, max: 76, color: C.dim, title: 'SLO 违约率（%）', stepSize: 20, fmt: function (v) { return v + '%'; } })
      },
      plugins: {
        legend: { display: false },
        barValueLabels: { dec: 2 },
        tooltip: Object.assign(tooltipOpts(), {
          callbacks: {
            title: function (items) { return '高压 ' + items[0].label + ' · n=16230'; },
            label: function (item) { return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(2) + '%'; }
          }
        })
      }
    }),
    plugins: [barValueLabels]
  });

  /* ================= S2 右（c3）：TTFT p95 柱 × 归因违约线 ================= */
  new Chart(document.getElementById('c3').getContext('2d'), {
    data: {
      labels: armCatsX,
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
        x: xArmScale(),
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
            title: function (items) { return '高压 ' + items[0].label; },
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

  /* ================= S3（c4）：golden 矩阵违约率按 cap 聚合 ================= */
  var violTop = DATA.violMean.map(function (m, i) { return Math.max.apply(null, DATA.violSeed.map(function (a) { return a[i]; })); });
  var violBot = DATA.violMean.map(function (m, i) { return Math.min.apply(null, DATA.violSeed.map(function (a) { return a[i]; })); });
  var c4ds = band(violTop, violBot, 'rgba(61,220,151,0.14)', 'y')
    .concat([{
      label: '档均值（M_cal P95）', data: DATA.violMean,
      borderColor: C.hot, borderWidth: 2.4, pointRadius: 4.5,
      pointBackgroundColor: C.hot, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.25
    }]);
  DATA.violSeed.forEach(function (arr, si) { c4ds.push(seedLine('s' + (si + 1), arr, 'rgba(61,220,151,0.5)', 'y', si)); });
  new Chart(document.getElementById('c4').getContext('2d'), {
    type: 'line',
    data: { labels: LABELS, datasets: c4ds },
    options: baseOpts({
      scales: {
        x: xCatScale(),
        y: yScale({ min: 0, max: 8, color: C.hot, title: 'SLO 违约率（%）· M_cal P95', stepSize: 2, fmt: function (v) { return v + '%'; } })
      },
      plugins: {
        legend: { display: false },
        hLine: { y: DATA.baseP95Pct, color: C.gold, text: '基线 seed P95 门限 ' + DATA.baseP95Pct.toFixed(2) + '%' },
        tooltip: Object.assign(tooltipOpts(), {
          filter: function (item) { return !item.dataset._band; },
          callbacks: {
            title: function (items) { return 'cap ' + (items[0] ? items[0].label : ''); },
            label: function (item) { return ' ' + item.dataset.label + '  ' + item.parsed.y.toFixed(2) + '%'; }
          }
        })
      }
    }),
    plugins: [hLine]
  });

  /* ================= S5 左（c5）：TTFT 平坦线（H1 带） ================= */
  var tBandTop = DATA.ttftMean.map(function (m, i) { return m + DATA.ttftStd[i]; });
  var tBandBot = DATA.ttftMean.map(function (m, i) { return m - DATA.ttftStd[i]; });
  var tSeeds = [];
  DATA.labels.forEach(function (lab, ci) {
    DATA.ttftSeed.forEach(function (arr) { tSeeds.push({ x: lab, y: arr[ci] }); });
  });
  new Chart(document.getElementById('c5').getContext('2d'), {
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

  /* ================= S5 右（c6）：驱逐健康带 ================= */
  var eBandTop = DATA.evMean.map(function (m, i) { return m + DATA.evStd[i]; });
  var eBandBot = DATA.evMean.map(function (m, i) { return m - DATA.evStd[i]; });
  var c6ds = band(eBandTop, eBandBot, 'rgba(93,184,217,0.14)', 'y')
    .concat([{
      label: 'ev_age mean', data: DATA.evMean,
      borderColor: C.cool, borderWidth: 2.4, pointRadius: 4.5,
      pointBackgroundColor: C.cool, pointBorderColor: '#0b0f0d', pointBorderWidth: 1.4, tension: 0.25
    }]);
  DATA.evSeed.forEach(function (arr, si) { c6ds.push(seedLine('s' + (si + 1), arr, 'rgba(93,184,217,0.55)', 'y', si)); });
  new Chart(document.getElementById('c6').getContext('2d'), {
    type: 'line',
    data: { labels: LABELS, datasets: c6ds },
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
})();
