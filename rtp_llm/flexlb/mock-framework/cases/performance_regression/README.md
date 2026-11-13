# 性能回归

在开发机用 **2 Master、55 Prefill、192 Decode、160 ViT、4 Client** 回放 316,367 条采样请求。前 120 秒以 0.25 倍速播放原始流量的前约 30 秒，随后以 **1.0 倍速**接着播放余下约 270 秒。整份数据只播放一遍，预计用时约 390 秒，原速阶段平均目标约 1,055 QPS。

按[框架 README](../../README.md)运行，设置 `CASE_FILE=cases/performance_regression/case.yaml`。Case 默认使用仓库内的 `data/traffic/deepseek-v4.1.mckidx`。基准文件的 SHA-256 为 `14b96f91201044a3cbf9f453107fbc9cfe002c0b4f7050735fa1a28d7e44eda9`。

业务配置来自 2026-11-10 的 Whale 部署 `dsv41_mock_no_kvcm_align`。P/D/V 和 Master 副本数、计算公式、KV 容量、CP4 和 Master 调度参数按该部署设置。开发机使用本地发现和 ZooKeeper，由 Client 回放代替线上 Frontend。ViT 使用采样宽高走计算路径，省略图片下载。

此前的冷启动原速验证跑在开发机 133，源码 `a49398941f`，FlexLB run `20261110_035731_cde01974`。启动配方将 Prometheus 采样设为 5 秒。Master 设置 `-XX:ActiveProcessorCount=32 -Xmx96g -XX:MaxDirectMemorySize=16g -XX:MaxGCPauseMillis=150`，P/D/V 各设 2 个可见处理器和 2 GiB 最大堆，Client 各设 16 个可见处理器和 16 GiB 最大堆。各角色共享开发机 CPU。

切速过渡期的 TTFT 和命中率在 View 中观察。Case 在回放约第 395 秒回查约第 360–390 秒的稳态窗口，在同一窗口检查吞吐、延迟和 Prefill 命中率，命中率门槛为 90%。该窗口与 300 秒预热基线对应同一段原始流量。随后执行 `client/stop/playback`，排空已发请求。启用 Prometheus 时，该步骤在排空后等待 10 秒，给 5 秒一次的采样留出时间。全程成功率用 `[-980s, 0s]` 覆盖本轮回放，预热、切速过渡期和排空阶段的失败、超时及取消都计入。

原始流量每 30 秒约为 1,025–1,085 QPS，门禁范围设为 960–1,150 QPS，全程成功率门槛为 99.9%。300 秒预热原速基线的稳态 TTFT 均值为 416 ms、P95 为 1.23 s，稳态门槛设为 600 ms 和 1.5 s。Prefill Forward、Master 调度响应和 Prefill 等待在 View 中用于诊断。

Prefill TPS 和含缓存 Prefill TPS 使用各 Worker 原生 gauge 的窗口均值之和。Decode TPS 是输出 token 的墙钟吞吐。实际缓存命中率和成功率按窗口内分子、分母增量汇总后相除。TTFT 按首 token 样本统计，TPOT 按成功且输出多个 token 的请求统计。ViT 的工作量及错误率保留在大盘供排查。
