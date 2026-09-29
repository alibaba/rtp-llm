# r41b 四层 64K FP8 PD 热态记录

2026-09-28 在 111 Prefill、112 Decode 的既有 TP8/EP8 服务上采集。两端各有 8 个本任务 rank，GPU 上未看到外部计算进程，RDMA bond 均为 ACTIVE。Prefill 从 3FS 通过 FastSafetensors SHM 直接加载，只有 111 的任务进程使用 64 线程 `pread` 原型；112 沿用原读取路径。启动配置启用 FP8 GEMM、FP8 KV、TokenSpeed Prefill、Native MTP，通信为 BF16 NCCL。服务运行在当时的 `66379cc5` 工作树及尚未提交的任务修改上；后来提交的源码快照不能证明与已加载二进制逐字节相同。这组数据只用于定位旧服务的热态路径。

第一次 64K 请求曾在 TokenSpeed/CuTeDSL 冷编译时触发 32 秒 keepalive 超时。复用原服务后，第一次重试完成 10 次预热和 8 次请求，均为 HTTP 200，但 profiler 配置 16 步只生成 rank0 trace，未用于全 rank 统计。本目录保存第二次重试：预热 10 次，末三次 HTTP 完整耗时偏离中位数最多 1.76%；随后 8 次请求均返回 200。每条记录的输入为 65,536 token、输出为 8 token、PD Decode handoff 为 61,440 token，Native MTP draft rounds 为 7。四层输出不用于判断自然语言答案正确性。

第二次 profiler 配置 8 步，八个 rank 均生成 trace。每个 trace 含 4 个 target/draft scope，但 rank0 比其余 rank 早一条请求进入采集窗口。按 target CPU scope 时间戳在 2 ms 内匹配，只有 **3 条共同请求**。关联 CUDA launch 后，每条请求的最慢 rank GPU span 为：

| 共同请求 | Prefill | Target | Draft |
| --- | ---: | ---: | ---: |
| 1 | 138.694 ms | 75.144 ms | 63.550 ms |
| 2 | 140.085 ms | 76.225 ms | 63.858 ms |
| 3 | 138.426 ms | 74.533 ms | 63.870 ms |
| 中位数 | **138.694 ms** | **75.144 ms** | **63.858 ms** |

这些是同请求、全 rank GPU kernel 的时间跨度，Target、Draft 与 Prefill 的范围会重叠，不能直接相加。Target 最慢 rank 的内核族累计时间中位数包括 MoE 14.382 ms、NCCL 14.570 ms、NVJet GEMM 10.685 ms、DeepGEMM 7.321 ms、TokenSpeed MLA 5.381 ms；各族可能跨 stream 重叠，不能当作独占耗时。不同机器上的 `feat/k3_dev` 或 vLLM trace 尚需按相同配置和逐算子契约复核，不能据此宣布集成版胜出。

`manifest.json` 保存原始文件大小和 SHA256。八个 `*.json.gz` 是逐 rank trace；`requests.tar.gz` 包含 `input.json`、10 个预热响应和 8 个采集响应。`independent-response-audit.json` 用归档中的原始响应重新核对了固定输入 ID、PD handoff 与 MTP draft。`aligned-phase-audit.json` 是本目录分析脚本的结果。解压 trace 后，可用 `../../analyze_r41_aligned_phases.py`，指定 `--trace-dir`、`--trace-template 'k3_64k_prefill_r41b_retry2_wr{rank}_2.json' --min-common 3` 复核。原始 trace 已与 111 上的文件逐个比对 SHA256。
