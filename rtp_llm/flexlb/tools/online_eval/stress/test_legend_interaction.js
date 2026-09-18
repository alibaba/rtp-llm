const assert = require('node:assert/strict');
const {nextVisibility, controller} = require('./legend_interaction');

assert.deepEqual(nextVisibility([true, true, true], 1, 'single'), [true, false, true]);
assert.deepEqual(nextVisibility([true, true, true], 1, 'double'), [false, true, false]);
assert.deepEqual(nextVisibility([false, true, false], 1, 'double'), [true, true, true]);

const visible = [true, true];
const chart = {
  data: {datasets: [{}, {}]},
  isDatasetVisible(i) { return visible[i]; },
  setDatasetVisibility(i, value) { visible[i] = value; }
};
let refreshed = 0;
const legend = controller(chart, () => refreshed++, 5);
legend.click(0);
legend.click(0);
assert.deepEqual(visible, [true, false]);
legend.click(0);
legend.click(0);
assert.deepEqual(visible, [true, true]);
legend.all();
assert.equal(refreshed, 3);
