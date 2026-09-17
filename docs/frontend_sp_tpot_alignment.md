# Frontend SP TPOT 透传

`py_rtp_frontend_tpot_ms` 改为逐条上报引擎 `rtp_llm_sp_estimate_tpot_us` 对应样本，唯一数值变换是除以 1000。原请求级 `(finish - first_output) / remaining_tokens` 计算已移除。TTFT、请求 RT、TPS 指标保持原有逻辑。

引擎与透传共用 `estimateTpotUs()`，计算窗口、上一轮接收长度的使用方式和有效轮次条件保持不变，没有以新公式修改作为对照的引擎指标。仅 MTP 有效样本透传，旧后端或非投机请求没有样本时不回退到其他 TPOT 口径。

## 样本传递

每轮从实际执行 batch 中选择一个开启 frontend 指标的真实请求承载样本，避免按 batch 大小重复加权。样本通过内部 GenerateOutputs/protobuf envelope 传递，不进入公开 AuxInfo、usage 或用户文本。DashScope、OpenAI renderer、pipeline 均从原始后端帧的 observer 接收。

同步 dispatch 可能在 step 计时结束前将最终响应入队，因此使用 promise 保存待完成样本，RPC 序列化等待真实计时结果；执行线程不等待 RPC。异常或无效轮次析构会取消待完成样本，释放等待方。异步 dispatch 同样读取最终完成的样本。

同一请求所有响应帧共享指标缓冲；metric-only 合帧不会丢弃独立样本，每个样本只被序列化一次。递增序号按请求单元/重试 attempt 在 frontend 去重，frontend 不再进行请求平均或 token 加权。

## 查询口径

这是执行 batch 级指标，在 frontend 使用 `rank_id`、`server_id` 容器标签上报，不以任意承载请求的 priority/source/streaming 归类。旧看板若过滤这些请求标签，需要改为容器维度。

对齐的是成功透传的引擎样本值和每轮一次的权重，并非保证两侧任意时间窗口的聚合值完全相同：网络延迟、取消/断连、没有启用 frontend 指标的请求，以及不同实例筛选仍会影响采样集合或窗口。引擎参考指标继续保留。

## 验证

- frontend：28 项通过，覆盖逐样本换算、重复观察、重试序号、收尾耗时不影响值、旧后端不回退。
- DashScope：95 项通过，覆盖 raw observer 到指标上报。
- RPC Python：17 项中 16 项通过、1 项跳过，覆盖新 protobuf 解码和公开 AuxInfo 不泄漏。
- backend visitor：31 项通过。
- OpenAI raw observer 定向测试：1 项通过。
- 实际 C++ 指标缓冲的 CPU 编译/运行通过，覆盖完成前等待、取消释放、合帧保留和只消费一次。
- C++ `query_converter_test` 编译被已有测试引用阻断：`PrefillRpcServer::effectiveOutputTokenBudget` 已不存在，HEAD 原有测试仍调用它。新加的序列化断言尚未执行；错误日志为 `query_converter_build.log`。未为本次指标改动修改这些无关的 PD 预算测试。

最终三目标增量构建全部通过：`//:th_transformer` 66.837 秒，`//rtp_llm:rtp_llm` 7.867 秒，`//rtp_llm:rtp_llm_lib` 1.123 秒。

构建和测试日志：`artifacts/frontend_sp_tpot/`。最终源码的三目标构建记录在 `build_final/results.tsv`。构建命令：

```bash
bazelisk build //:th_transformer --verbose_failures --config=cuda13 --test_output=errors --test_env="LOG_LEVEL=INFO" --config=sm10x
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=cuda13 --test_output=errors --test_env="LOG_LEVEL=INFO" --config=sm10x
bazelisk build //rtp_llm:rtp_llm_lib --verbose_failures --config=cuda13 --test_output=errors --test_env="LOG_LEVEL=INFO" --config=sm10x
```

本次未进行线上看板或整模型流量比对，不能将单测视为线上聚合值已验证一致。

## 提交时仍存在的 review 问题

以下两项 P2 尚未修复，以上测试结果不能证明正常链路无回归：

- 旧业务响应读取共享缓冲时，`take()` 可能等待后续轮次尚未完成的样本，延迟 token 发送及取消检查。需要按响应限制样本水位，或避免等待未完成样本。
- stream-async 下承载者选择未排除 finished/error stream，可能选中已结束且终帧已消费的请求，导致本轮指标丢失。需要过滤不可再产生响应的请求，并增加异步结束回归。
