# 性能回归

按[框架 README](../../README.md)运行，设置 `CASE_FILE=cases/performance_regression/case.yaml`。每次运行独立使用 Case 中的绝对阈值。

Case 使用两个 Master、48 个 Prefill、192 个 Decode 和两个 Client，以总 0.1 倍速回放 DeepSeek。Client 和 Prefill 使用相同的 128-token 基础块与 CP4，Master 用同一组路由键做缓存亲和性调度。请求和 Prefill 实例均启用缓存复用。组件配置由本 Case 维护，流量引用共享数据，结果使用框架默认大盘。

回放启动后先等待 370 秒，再检查两个连续的 60 秒窗口。每段命中率都须在 40%～46%，任一窗口越界即结束 Case。两次检查通过后再等待 65 秒，让首个性能窗口完整落在确认稳定之后。第一组性能门禁约在启动回放后 495 秒执行，每项查询汇总过去 60 秒并预留 5 秒采集余量。

初始阈值基于开发机 133 对 `035b266a6c` 的同配置校准。12 个完整窗口中，第 6～12 个窗口的命中率为 40.65%～43.25%，已发 QPS 为 84.9～93.5，成功率为 100%。吞吐下限比这一段的实测最小值留出约 13%～15%，延迟使用独立上限。校准记录位于 FlexLB run `20261109_033835_c0348f12` 的 `calibration-routed-result`。

含缓存 Prefill TPS 对应 Real 的 `context_tps_with_cache`，统计含复用 token 的输入吞吐。输出 TPS 按已提交输出 token 计算墙钟吞吐。TTFT 包含所有收到首 token 的请求，TPOT 统计成功且输出多于一个 token 的请求。

指标接口的 mode 默认为 interval，Case 中省略不写；逐点检查时显式写 mode: series。interval 返回一个值，gauge 取区间平均并跨 Worker 求和，比率用分子分母的区间增量汇总后相除，延迟均值用耗时增量除以样本数，P95/P99 合并 bucket 增量后求分位数。series 返回按采集周期排列的数值列表，gte/lte 要求每个点都满足。派生 counter 查询的内部统计长度固定为四个采集周期。缺点、缺实例、NaN 和零分母均报错，不能当作通过。

性能窗口内先检查成功率。随后 `client/stop` 停止回放并排空，检查所有已发请求的终态完整性。再等 5 秒收齐 Prometheus 样本，检查覆盖预热、性能窗口和排空的全程成功率。

不再支持数值指标的 `range: latest`。省略 range 时使用最近 60 秒；需要逐点门禁时明确使用 `mode: series` 和时间范围。
