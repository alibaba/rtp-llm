# DeepSeek-V4-Pro prefill performance (corrected-v4)

This report is **DeepSeek-V4-Pro only**.  The old v3 result is rejected and is
not mixed into these numbers.  The authoritative raw result is
`cache_grid_results.corrected_v4.json` in the delivered artifact directory.

## Final measured coverage

| Item | Result |
|---|---:|
| Raw completed grid rows | 4,819 / 4,819 |
| Successful rows with stable observed reuse | 4,523 |
| Unique physical geometries used for fitting | 4,292 |
| Distinct input lengths | 489 |
| Input length range | 256 .. 1,048,575 |
| Observed cache range | 0 .. 1,044,480 |
| Batch size | 1 |
| Measurements per geometry | 3 |

The runner aligns cache to physical blocks.  Therefore the requested cache is
not always the cache actually reused.  We model `cache_len_observed` (the
stable reuse reported by all three runs), collapse duplicate physical
geometries by median, and reject positive requests that observed zero reuse.
This is why 4,819 raw rows become 4,292 fitting geometries; it is not missing
data silently converted to cache hits.

## Representative 1M input

`input_len=1,048,575`, batch 1:

| Requested cache | Observed cache | Median prefill RT (ms) |
|---:|---:|---:|
| 0 | 0 | 173.124 |
| 130,816 | 126,976 | 173.196 |
| 261,888 | 258,048 | 180.226 |
| 392,960 | 389,120 | 178.476 |
| 524,032 | 520,192 | 179.442 |
| 655,104 | 651,264 | 182.965 |
| 786,176 | 782,336 | 177.115 |
| 917,248 | 913,408 | 191.022 |
| 996,096 | 995,328 | 176.917 |
| 1,048,064 | 1,044,480 | 180.391 |

These are DSV4 measurements; do not substitute the earlier GLM5 1M value.

## Formula

The fitted expression minimizes mean absolute error (L1/MAE) and uses only the
FlexLB-supported variables `tokens` and `hitCacheTokens` and arithmetic
operators.  For this fixed-batch dataset,
`tokens` is the full input length and `hitCacheTokens` is the **observed**
reused cache length.  Output is milliseconds:

```text
174.752216622553
+ 0.0135885335168253 * tokens / 1024.0
- 0.00699083362731531 * hitCacheTokens / 1024.0
- 5.13588525008408e-06 * (tokens / 1024.0) * (tokens / 1024.0)
- 9.85209664239846e-06 * (tokens / 1024.0) * (hitCacheTokens / 1024.0)
+ 1.38198202857097e-05 * (hitCacheTokens / 1024.0) * (hitCacheTokens / 1024.0)
```

No `sum`, `max`, `batchSize`, `computeTokens`, or Python-only syntax is used.
The formula is valid for batch 1 and the measured DSV4 range only; it is not a
batch-scaling formula.

## Fit accuracy

| Split | N | MAPE | p95 absolute relative error | Max relative error | MAE (ms) |
|---|---:|---:|---:|---:|---:|
| Train | 3,148 | 3.310% | 13.398% | 52.820% | 6.859 |
| Validation | 535 | 2.987% | 12.891% | 35.339% | 5.922 |
| Test | 609 | 3.246% | 12.977% | 39.099% | 6.665 |
| All fitting geometries | 4,292 | 3.260% | 13.245% | 52.820% | 6.715 |

The all-data MAE is 6.715 ms and the test MAE is 6.665 ms, lower than the
previous squared-error fit.  Relative-error p95/max remain higher, so this is
an audited absolute-error-optimized fit, not a claim of worst-case production
SLA accuracy.

## Reproducible artifacts

- Raw DSV4 JSON: `dsv4_corrected_v4/cache_grid_results.corrected_v4.json`
- Input audit: `dsv4_corrected_v4/formula/input_audit.json`
- Fit report: `dsv4_corrected_v4/formula/fit_report.json`
- Formula text: `dsv4_corrected_v4/formula/deepseek_v4_prefill_formula.txt`
- Per-geometry predictions: `dsv4_corrected_v4/formula/predictions.csv`
- Dense 3D chart: `dsv4_corrected_v4/deepseek_v4_prefill_3d.svg`

The chart uses X=measured prefill RT (TTFT, ms), Y=observed cached tokens,
and Z=uncached compute tokens (`input_len - observed_cache_len`).  The updated
3-D view is an isometric projection: the three axes are drawn from one origin,
the dots are the measured physical geometries, and the guide lines show
selected fixed-cache and fixed-compute slices.  Colour identifies the slice
family; it does not encode RT.  Failed rows and positive-cache requests with
zero observed reuse are excluded.

## Cache-miss trend (2-D)

The companion chart is separate from the 3-D view.  It keeps only
`observed_cache_len=0`, so each dot answers one question: how does cold
prefill RT change as `seq_len` grows?  The horizontal axis is linear in tokens,
and the small inset magnifies 0–131K tokens because a 1M-wide axis compresses
short sequences near the origin.  Each of the 489 dots is the median of the
three successful measurements for that physical geometry.

Run both charts together with:

```bash
python3 rtp_llm/test/perf_test/generate_prefill_3d_chart.py \
  --input /data0/luoli.hn/work/glm52-prefill-perf-results/dsv4_corrected_v4/cache_grid_results.corrected_v4.json \
  --output /data0/luoli.hn/work/glm52-prefill-perf-results/dsv4_corrected_v4/deepseek_v4_prefill_3d_readable.svg \
  --cold-output /data0/luoli.hn/work/glm52-prefill-perf-results/dsv4_corrected_v4/deepseek_v4_prefill_cold_miss_2d.svg \
  --batch-size 1
```

The generated files are `deepseek_v4_prefill_3d_readable.svg` and
`deepseek_v4_prefill_cold_miss_2d.svg`.  In this DSV4 dataset the 1M cold point
(`input_len=1,048,575`, `cache=0`) is 173.1 ms.  That value belongs to DSV4;
it must not be compared with the earlier GLM5 1M result without changing the
model label.
