# DeepSeek-V4-Pro Prefill 性能数据审计与临时拟合

这份报告只保留通过严格校验的数据。旧报告把 `invalid_reuse` 当成可用样本，导致 4,131 个 cache 未按请求命中的 case 混入图表和拟合。旧版 4,292 点公式、误差和图表全部作废。

## 数据结论

| 项目 | 数量 |
|---|---:|
| 原始 case | 4,819 |
| 严格有效 case | 688 |
| 排除 case | 4,131 |
| 有效测量轮次 | 2,064 |
| cache=0 case | 489 |
| cache>0 且精确命中 case | 199 |
| input_len 数量 | 489 |
| input_len 范围 | 256–1,048,575 |
| batch size | 1 |

一个 case 只有同时满足以下条件才会进入统计和拟合：

- `status=ok`；
- 三轮测量全部成功；
- 三轮 `input_len` 与请求一致，`output_len=1`；
- 三轮 `reuse_len` 完全相同，并且精确等于请求的 cache length；
- latency 为有限正数。

4,131 个被排除的 case 均为 `status=invalid_reuse`，三轮实际 `reuse_len` 与请求值不一致。它们虽然完成了 HTTP forward，但不代表请求的 cache geometry，因此不能用于 cache 性能统计。

## 测量口径限制

当前 JSON 记录的延迟来自服务端 `aux_info.first_token_cost_time`。runner 没有记录每个请求的客户端 HTTP wall time，也没有保存 local、remote、memory、device cache 的复用明细。

这会造成明显的合理性问题：`input_len=1,048,575, cache=0` 的三轮服务端数值为 333.363、171.510、173.124 ms，中位数 173.124 ms；但整个三请求 case 的 wall time 是 13.346 秒。现有数据无法解释两者之间的差值，所以 173.124 ms 不能写成端到端 TTFT。

下面的公式只拟合 JSON 中的服务端 `first_token_cost_time`，用于检查数据形状和拟合流程。它不是可上线的端到端 TTFT 公式。

## 严格数据重拟合

目标值取每个有效 case 三轮 `prefill_time_ms` 的中位数。拟合目标为 MAE，表达式只使用 FlexLB 当前支持的 `tokens`、`hitCacheTokens` 和四则运算：

```text
177.638246514008
+ 0.00666514075108166 * tokens / 1024.0
- 0.0258212149796806 * hitCacheTokens / 1024.0
+ 3.15503708549298e-06 * (tokens / 1024.0) * (tokens / 1024.0)
+ 5.44659822292038e-07 * (tokens / 1024.0) * (hitCacheTokens / 1024.0)
+ 2.65730353692932e-05 * (hitCacheTokens / 1024.0) * (hitCacheTokens / 1024.0)
```

| 数据集 | N | MAE | MAPE | p95 相对误差 | 最大相对误差 |
|---|---:|---:|---:|---:|---:|
| Train | 507 | 14.196 ms | 6.327% | 31.434% | 46.456% |
| Validation | 84 | 10.084 ms | 4.808% | 20.084% | 34.437% |
| Test | 97 | 16.904 ms | 7.363% | 33.137% | 38.590% |
| 全部严格有效数据 | 688 | 14.076 ms | 6.288% | 31.364% | 46.456% |

结论很直接：有效 cache 数据只有 199 个，tail error 很大；加上 TTFT 测量口径未闭环，这个公式不能上线，`production_acceptance=false`。

## 图表

- `deepseek_v4_prefill_cold_strict_489.svg`：489 个 cache=0 case 的服务端指标趋势。
- `deepseek_v4_prefill_3d_strict_688.svg`：688 个严格有效 case，X=服务端指标，Y=cached tokens，Z=compute tokens。

图中已经彻底排除 4,131 个 `invalid_reuse` case。由于服务端指标尚不能等同于端到端 TTFT，图标题和正文不得再写“真实 TTFT”。

## 产物

- 原始输入（含失败记录，仅供审计）：`dsv4_corrected_v4/cache_grid_results.corrected_v4.json`
- 严格有效数据：`dsv4_corrected_v4/cache_grid_results.strict_valid_688.json`
- 严格拟合：`dsv4_corrected_v4/formula_strict_valid_688_final/`
- 公式：`formula_strict_valid_688_final/deepseek_v4_prefill_formula.txt`
- 输入审计：`formula_strict_valid_688_final/input_audit.json`
- 误差报告：`formula_strict_valid_688_final/fit_report.json`
- 逐点预测：`formula_strict_valid_688_final/predictions.csv`

下一轮采集必须补上逐请求客户端 wall time，并记录完整 cache reuse 分项。只有重跑后通过 1M 冷点合理性检查，才能生成正式 TTFT 报告和生产公式。
