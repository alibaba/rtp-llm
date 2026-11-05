# 固定流量性能门禁

运行方式见[框架 README](../../README.md)，将 `CASE_FILE` 设置为 `cases/version_regression/case.yaml`，为各版本分别生成 launch 和结果目录，保持 Case、流量与阈值一致。

参考 HA Case 的两个 Master、48 Prefill、192 Decode 和计算模型，启用 P/D 及请求级缓存复用。使用同一个 Case 配合不同版本的 launch 配方分别运行，各次独立使用绝对阈值。

Master、Prefill、Decode 配置由本 Case 的 `configs/` 独立维护；DeepSeek 流量引用共享的 `data/traffic/deepseek.binpb.zst`。省略 view，使用框架默认大盘。

前 120 秒预热，180 秒后依次查询各指标最近 60 秒的汇总值，结束点留出 5 秒采集余量。随后停止回放并排空，所有已发请求必须有终态。此 Case 使用 launch 中默认的 1 秒 Prometheus 采集周期；排空后再等待 5 秒，收集最终终态计数，然后检查本次运行全程成功率。每个查询相对自己的 Step 时间，不要求跨接口冻结查询时刻。

当前阈值是初始配置，完整部署尚需在目标机器校准。含缓存 TPS 对应 Real 的 context_tps_with_cache，表示完整输入吞吐。Output TPS 使用提交输出计数器计算墙钟吞吐。TTFT 包含所有收到首 token 的请求，TPOT 只包含成功且输出多于一个 token 的请求。

启动参数 reuse_cache 控制请求级复用，Engine 配置中的 reuse_cache 控制实例策略，两者共同生效。

指标接口的 mode 默认为 interval，Case 中省略不写；逐点检查时显式写 mode: series。interval 返回一个值，比率用分子分母的区间增量汇总后相除，延迟均值用耗时增量除以样本数，P95/P99 合并 bucket 增量后求分位数。series 返回按采集周期排列的数值列表，gte/lte 要求每个点都满足。派生 counter 查询的内部统计长度固定为四个采集周期。缺点、缺实例、NaN 和零分母均报错，不能当作通过。

此 Case 的 interval 成功率针对测量区间内完成的请求。client/stop 另检查全部已发请求终态完整性，返回的 outcomes 保留排空期间失败，供结果审查。

不再支持数值指标的 `range: latest`。省略 range 时使用最近 60 秒；需要逐点门禁时明确使用 `mode: series` 和时间范围。
