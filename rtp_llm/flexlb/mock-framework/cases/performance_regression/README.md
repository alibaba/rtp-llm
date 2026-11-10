# 性能回归

在开发机用 **2 Master、55 Prefill、192 Decode、160 ViT、4 Client** 回放 316,367 条采样请求。先以 0.25 倍速预热 300 秒，停止预热并排空已发请求。保留已暖的缓存，再从数据起点以 **1.0 倍速**正式回放，目标时长约 5 分钟，平均目标约 1,055 QPS。预热和正式回放使用同一份数据，约前 75 秒的原始流量参与预热。

按[框架 README](../../README.md)运行，设置 `CASE_FILE=cases/performance_regression/case.yaml`。将采样文件放到共享根的 `data/traffic/frontend-300s-paged.mckidx`，执行 `compile`、`launch` 和 `run` 时都传 `--shared-root <共享根>`。本次基准文件的 SHA-256 为 `14b96f91201044a3cbf9f453107fbc9cfe002c0b4f7050735fa1a28d7e44eda9`。

业务配置来自 2026-11-10 的 Whale 部署 `dsv41_mock_no_kvcm_align`。P/D/V 和 Master 副本数、计算公式、KV 容量、CP4 和 Master 调度参数按该部署设置。开发机使用本地发现和 ZooKeeper，由 Client 回放代替线上 Frontend。ViT 使用采样宽高走计算路径，省略图片下载。

此前的冷启动原速验证跑在开发机 133，源码 `a49398941f`，FlexLB run `20261110_035731_cde01974`。启动配方将 Prometheus 采样设为 5 秒。Master 设置 `-XX:ActiveProcessorCount=32 -Xmx96g -XX:MaxDirectMemorySize=16g -XX:MaxGCPauseMillis=150`，P/D/V 各设 2 个可见处理器和 2 GiB 最大堆，Client 各设 16 个可见处理器和 16 GiB 最大堆。各角色共享开发机 CPU。

正式回放约第 305 秒回查该轮第 210–240 秒和第 240–270 秒两个命中率窗口。两次均达到 90% 后，检查约第 270–300 秒的性能窗口。随后停止 Client 并排空，按本轮 `client/stop` 的终态汇总检查完整性和成功率。查询时间为各步骤开始时刻，指标窗口预留 5 秒采集余量。

原始流量每 30 秒约为 1,025–1,085 QPS，门禁范围设为 960–1,150 QPS。原速验证触及 Prefill 批容量，最后一个窗口命中率为 82.03%、成功率为 85.86%。吞吐和延迟门槛沿用四分之一速校准值，全程成功率门槛为 99.9%。原速性能基准待满速通过后重新校准。

Prefill TPS 和含缓存 Prefill TPS 使用各 Worker 原生 gauge 的窗口均值之和。Decode TPS 是输出 token 的墙钟吞吐。实际缓存命中率和成功率按窗口内分子、分母增量汇总后相除。TTFT 按首 token 样本统计，TPOT 按成功且输出多个 token 的请求统计。ViT 的工作量及错误率保留在大盘供排查。
