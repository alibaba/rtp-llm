# 2026-09-28 FP8 集成进度快照

分支：`codex/luohaocheng-k3-fp8-opt-main-20260927`。源码快照提交为 `9233bf200d7a73e7974363b94df578cdf3cf1fb9`；当前目录单独保存任务级 3FS 并发读取原型和关键验证记录，便于开发机器清盘后恢复。目录内的启动脚本包含当时机器与个人数据盘路径，使用前要重新检查 GPU 独占、端口、容器、RDMA 和权重版本。

这份提交**不是**计划中的性能锚点或峰值激活锚点。四层 r41 FP8 TP8/EP8 PD flow 的 11 条请求全部返回 200，均记录到 PD 交接和 Native MTP draft，最慢 120.632 秒；四层输出不能作为完整 93 层答案正确性的证据。r41 的一次全 rank 64K trace 只有两条可跨 rank 对齐的共同请求，首条 rank0 NCCL 异常高，尚不能据此宣称整体性能胜出。r41b 使用任务级 64 线程 3FS 读取层后，111 Prefill 的 FastSafetensors guard 通过；首个 64K 预热请求在 TokenSpeed/CuTeDSL 冷编译时触发 32 秒 keepalive 超时，记录为失败，没有性能样本。完整 93 层 FP8 smoke、三方最终对照和两个正式锚点均未完成。

文件说明：

- `3fs-owner-read-report-20260928.md`：原始慢读、并发探针、FastSafetensors 加载路径及证据边界。
- `3fs-parallel-pread-prototype-20260928.md`、`parallel_3fs_pread.c`：只作用于任务进程的并发读取原型、启用条件和回滚办法。该原型尚未纳入 RTP 运行时代码。
- `bench_*.py`、`verify_fss_parallel_two_shards_113.py`、`run_fss_parallel_abba_113.sh`：单 shard 性能、全张量摘要和同大小双 shard 切换复核。
- `run_fp8_3fs_*.sh`、`run_64k_*.sh`、`analyze_r41_aligned_phases.py`：当时的四层 Prefill 启动和 timeline 复现脚本。
- `evidence/`：小型原始读速日志、guard 结果、逐 case flow 审计、全 rank 相位审计及冷编译失败日志。

大型逐 rank trace、完整请求 JSON、模型权重、Bazel/JIT 缓存没有存入 Git；当时的本地原件在 `/data1/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/`，111/112 的运行目录中也有对应日志。若这些机器清盘，本分支保留源码与本页列出的关键结果，但无法重建未上传的原始 trace。工作树的 `internal_source` 曾被本机改成绝对路径符号链接，仅为本地内部依赖定位；该机器路径没有提交，远端仍保留原来的相对符号链接。
