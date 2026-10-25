/* Chart.js legend behavior only. The report owns colors, layout, and labels. */
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.FlexLegend = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  function nextVisibility(current, index, gesture) {
    if (!Number.isInteger(index) || index < 0 || index >= current.length)
      throw new RangeError('invalid legend index');
    if (gesture === 'double') {
      return current.filter(Boolean).length === 1
        ? current.map(() => true)
        : current.map((_, i) => i === index);
    }
    if (gesture !== 'single') throw new Error('unknown legend gesture');
    return current.map((visible, i) => i === index ? !visible : visible);
  }

  function controller(chart, refresh, delayMs = 300) {
    let pending = null;
    const visibility = () => chart.data.datasets.map((_, i) => chart.isDatasetVisible(i));
    const apply = (index, gesture) => {
      nextVisibility(visibility(), index, gesture).forEach((visible, i) =>
        chart.setDatasetVisibility(i, visible));
      refresh(chart);
    };
    return {
      click(index) {
        if (pending && pending.index === index) {
          clearTimeout(pending.timer);
          pending = null;
          apply(index, 'double');
          return;
        }
        if (pending) {
          clearTimeout(pending.timer);
          apply(pending.index, 'single');
        }
        const timer = setTimeout(() => {
          pending = null;
          apply(index, 'single');
        }, delayMs);
        pending = {index, timer};
      },
      all() {
        if (pending) clearTimeout(pending.timer);
        pending = null;
        chart.data.datasets.forEach((_, i) => chart.setDatasetVisibility(i, true));
        refresh(chart);
      }
    };
  }
  return {nextVisibility, controller};
});
