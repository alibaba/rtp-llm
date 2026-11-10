# 性能回归

在开发机用 **2 Master、55 Prefill、192 Decode、160 ViT、4 Client** 回放 316,367 条采样请求。按原始到达间隔以 0.25 倍速回放，持续约 20 分钟，平均目标约 264 QPS。

按[框架 README](../../README.md)运行，设置 `CASE_FILE=cases/performance_regression/case.yaml`。将采样文件放到共享根的 `data/traffic/frontend-300s-paged.mckidx`，执行 `compile`、`launch` 和 `run` 时都传 `--shared-root <共享根>`。本次基准文件的 SHA-256 为 `14b96f91201044a3cbf9f453107fbc9cfe002c0b4f7050735fa1a28d7e44eda9`。

业务配置来自 2026-11-10 的 Whale 部署 `dsv41_mock_no_kvcm_align`。P/D/V 和 Master 副本数、计算公式、KV 容量、CP4 和 Master 调度参数按该部署设置。开发机使用本地发现和 ZooKeeper，由 Client 回放代替线上 Frontend。ViT 使用采样宽高走计算路径，省略图片下载。

基准跑在开发机 133，源码 `f0981e0c76`，FlexLB run `20261110_032511_428cb535`。启动配方将 Prometheus 采样设为 5 秒。Master 设置 `-XX:ActiveProcessorCount=32 -Xmx96g -XX:MaxDirectMemorySize=16g -XX:MaxGCPauseMillis=150`，P/D/V 各设 2 个可见处理器和 2 GiB 最大堆，Client 各设 8 个可见处理器和 6 GiB 最大堆。各角色共享开发机 CPU，结果用作这套开发机回放的回归基准。

回放后约第 905 秒和第 965 秒分别检查一个完整 60 秒命中率窗口。两次均达到 90% 后，再留出独立的性能窗口。随后继续回放至约第 1,205 秒，停止 Client 并排空，再检查全程成功率。各指标查询使用步骤开始时刻之前的 `[-65s, -5s]`，预留 5 秒采集余量。

阈值来自本次六个后段窗口。QPS 给观察上下沿各留约 6% 余量，吞吐下限取实测最小值的 85% 后向下取整，延迟上限取实测最大值的 125% 后向上取整。命中率下限比稳态最低值低约 2 个百分点，全程成功率门槛为 99.9%。实测完整回放全部成功，采集和清理均完成。

Prefill TPS 和含缓存 Prefill TPS 使用各 Worker 原生 gauge 的窗口均值之和。Decode TPS 是输出 token 的墙钟吞吐。实际缓存命中率和成功率按窗口内分子、分母增量汇总后相除。TTFT 按首 token 样本统计，TPOT 按成功且输出多个 token 的请求统计。ViT 的工作量及错误率保留在大盘供排查。
