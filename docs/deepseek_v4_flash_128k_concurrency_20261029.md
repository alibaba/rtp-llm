# DeepSeek V4 Flash：128K 输入并发压测（2026-10-29）

输入固定 131,072 token，输出固定 256 token，关闭 prefix reuse；对比 CSA sparse offload 开关。每档 C 个闭环客户端，每个客户端依次发两次请求，共 2C 次，首轮由 barrier 同时释放；两组正式请求共 1,008 次。吞吐包含首轮启动和最后排空时间，所有时间均为单轮实验观测值。

offload 将 D 单卡缓存池 HBM 分配从 43.16 降至 18.80 GiB，并使用 27.41 GiB Host；整卡显存峰值从 132.58 降至 108.42 GiB。客户端 TPOT P50 增加约 1.6%～3.9%，吞吐差异在 -0.5%～+1.7% 内。本配置的主要瓶颈是 P 端：即使客户端并发 128，采样到的 D 同时 RUNNING 峰值仍只有 2；客户端并发主要增加排队，不能据此推断 128 个请求同时 decode 时的 offload 性能。

4 张 L20D，GPU 0/1 为 P（CP2/EP2），GPU 2/3 为 D（TP1/DP2/EP2），MegaMoE SE，FP8 KV，完整本地 DeepSeek-V4-Flash-0731。DSpark gamma=3，D CUDA Graph capture 为 1/2/4/8/16/32/64，P eager，保留默认 CSA/HCA 融合。P 沿用 max_context_batch_size=1，一次 prefill 一个请求。两组依次运行。decode 端的两个 DP rank 通过轮询均匀接收请求，原始返回的 grpc_port 分布已逐条验证。

## 性能

TPOT 是每个请求从首个到最后一个输出 token 的客户端流式耗时除以 255；表中 P50/P99 是这些“请求平均 TPOT”的分位数，不是逐 token 间隔的 P99。DSpark 可能一次输出多个 token。另保存引擎侧 TPOT 供交叉核对。

| 并发 | 请求数/组 | 吞吐 关→开 tok/s | 变化 | TPOT P50 关→开 ms | TPOT P99 关→开 ms | TTFT P50 关→开 s | TTFT P99 关→开 s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 115.3 → 116.8 | +1.3% | 7.93 → 8.24 | 8.41 → 9.13 | 5.49 → 5.38 | 7.97 → 7.72 |
| 8 | 16 | 125.2 → 126.4 | +1.0% | 7.99 → 8.12 | 8.38 → 8.73 | 12.94 → 12.85 | 15.25 → 15.02 |
| 16 | 32 | 130.6 → 130.6 | -0.0% | 7.95 → 8.18 | 8.76 → 9.72 | 27.90 → 27.79 | 29.82 → 29.92 |
| 32 | 64 | 133.3 → 132.7 | -0.5% | 8.00 → 8.19 | 9.32 → 9.69 | 56.71 → 57.35 | 59.28 → 60.07 |
| 64 | 128 | 133.3 → 134.1 | +0.6% | 8.06 → 8.24 | 9.14 → 10.19 | 114.50 → 112.54 | 120.80 → 119.51 |
| 128 | 256 | 133.0 → 135.2 | +1.7% | 8.10 → 8.36 | 14.02 → 18.70 | 232.07 → 223.51 | 245.22 → 250.69 |

P99 使用线性插值；4 并发只有 8 个请求，尾部估计接近最大样本，不能视为稳定的生产 P99。summary.json 同时保留 P95 和引擎侧 TPOT。

高并发时前端/客户端的流式交付延迟会影响首个与最后一个 token 的到达间隔。例如 128 并发启动阶段可能出现服务端已生成若干 token，客户端才开始收到输出。因此另列引擎侧 TPOT（aux_info.cost_time-first_token_cost_time，再除以 255），帮助区分生成耗时和流式交付尾延迟；不把客户端 TPOT 偶发偏低解释为引擎加速。

| 并发 | 引擎 TPOT P50 关→开 ms | 引擎 TPOT P99 关→开 ms |
|---:|---:|---:|
| 4 | 7.92 → 8.24 | 8.40 → 9.12 |
| 8 | 7.99 → 8.14 | 8.37 → 8.73 |
| 16 | 7.98 → 8.19 | 8.76 → 9.70 |
| 32 | 8.09 → 8.25 | 9.51 → 9.67 |
| 64 | 8.08 → 8.28 | 9.14 → 9.93 |
| 128 | 8.05 → 8.35 | 9.54 → 9.96 |


## KV cache 分配与占用

下表为每个 rank 的缓存池实际分配，包含 CSA/Indexer/HCA、SWA、压缩状态和 draft cache；offload HBM 数值包括 resident KV 与 native generation array，不包括 Python runtime 的映射/选择表、临时 workspace 和 CUDA Graph 内存。整卡显存另在后表列出。池在服务启动时预分配，因此六档并发的池分配相同。

| 方案 | 角色/单卡 | 缓存池 HBM GiB | 缓存池 Host GiB |
|---|---|---:|---:|
| resident | prefill | 39.121 | 0.000 |
| resident | decode | 43.161 | 0.000 |
| offload | prefill | 39.121 | 0.000 |
| offload | decode | 18.800 | 27.407 |

逻辑占用是 1 秒采样中 D0/D1 较高的“已分配给请求的 block / 可分配 block”峰值，包含 KV_ALLOCATED 等待 prefill 的预留，不代表这些 block 都已填入 KV 数据。每个 D rank 可分配 36,863 个 block（9,436,928 个原 token），其余为 sentinel。offload 请求的 CSA backing 走 Host，Indexer/HCA 等继续使用 HBM；DEVICE/csa_kv 日志标签是逻辑池名，不能直接当作 HBM 用量。

| 并发 | 逻辑 KV 占用峰值 关→开 | D0 / D1 已预留 token 峰值 关 | D0 / D1 已预留 token 峰值 开 |
|---:|---:|---:|---:|
| 4 | 4.17% → 4.18% | 262656 / 393984 | 394240 / 262656 |
| 8 | 5.57% → 6.96% | 525312 / 525312 | 525312 / 656640 |
| 16 | 12.53% → 12.53% | 1050624 / 1182208 | 1050624 / 1182208 |
| 32 | 25.05% → 25.05% | 2232576 / 2363904 | 2232576 / 2363904 |
| 64 | 47.32% → 51.49% | 4465152 / 4333824 | 4333824 / 4859136 |
| 128 | 94.63% → 94.63% | 8536320 / 8930304 | 8798976 / 8930560 |

## 实际并发与显存

并发数指客户端在途请求数。每秒通过现有 worker_status 接口分别读取两个 D rank 的 TASK_PHASE_RUNNING 数量和所有已登记任务数；后者包括等待 P 的 KV_ALLOCATED 请求，不能当作 decode batch。采样峰值可能漏掉短暂的运行峰值。

| 并发 | D 同时 RUNNING 峰值 关→开 | D0 / D1 RUNNING 峰值 关 | D0 / D1 RUNNING 峰值 开 | D 单卡显存峰值 关→开 GiB | P 平均利用率 关→开 |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 → 2 | 1 / 1 | 2 / 1 | 132.58 → 108.42 | 100.0% → 98.7% |
| 8 | 2 → 2 | 1 / 1 | 1 / 2 | 132.58 → 108.42 | 85.7% → 85.7% |
| 16 | 2 → 2 | 2 / 1 | 1 / 2 | 132.58 → 108.42 | 98.6% → 100.0% |
| 32 | 2 → 2 | 2 / 2 | 2 / 2 | 132.58 → 108.42 | 94.6% → 96.0% |
| 64 | 2 → 2 | 2 / 2 | 2 / 2 | 132.58 → 108.42 | 97.9% → 95.6% |
| 128 | 2 → 2 | 2 / 2 | 2 / 2 | 132.58 → 108.42 | 97.3% → 98.1% |

两组逻辑缓存容量均固定为 36,864 个 block，block=256 个原 token；状态/SWA 固定池为 512 block，并发上限 128，KV 预算 64 GiB。offload 仅在 D 开启，Host 预算 32 GiB，强制 HBM backing block=2（含 sentinel；用于测试强制 Host 路由）。实际 CSA Host block=36862、逻辑 block=36864、resident 上限=262144 行，Host backing 每张 D 卡 27.407 GiB、resident KV 3.047 GiB。Indexer/HCA/SWA/状态池及 draft cache 仍在 HBM。此测试比较同容量时的搬运成本和 HBM 分配，不代表自动生产容量策略。

显存为每 5 秒 nvidia-smi 的实际设备总使用量，按各档正式请求时间窗取峰值。原始 GPU 利用率和 worker 状态均已保存；设备物理内存 268.6 GiB/卡。

## 验证与复现

1,008 个正式请求逐条验证输入 131,072、输出 256、P/D 成功、prefix reuse=0、DSpark 执行，且每档两个 D rank 各接收 C 个请求。两组 capture buckets 均成功，正式计时外另有请求期间的 graph replay 日志。offload 的 H=2 包括 sentinel，因此至多 1 个活跃 block 能使用 GPU backing；周期日志中的单 rank CSA 占用达到 34884 个 block，至少 34883 个使用 Host backing。本轮 probe 轮询到了 D rank 1，DEBUG 设置作用于 rank 0，因此没有逐 block 的导入 DEBUG 记录；不沿用上一轮的导入条目数作为本轮证据。两组输出文本会受 FP8/MTP 批次影响，不将吞吐波动直接归因为 offload 加速。

首次检查发现旧脚本的 D 路由仅列出 rank 0。该轮在 16 并发中断，单独归档到 initial-single-dp/，所有正式表格都来自补齐两个 D 地址后的完整重跑；因此不能直接和上一轮单 D 路由的 128K/4 并发数据混用。

```bash
/opt/conda310/bin/python3 /home/renyuanming.rym/v4-flash-128k-concurrency-20261029/run_comparison.py
/opt/conda310/bin/python3 /home/renyuanming.rym/v4-flash-128k-concurrency-20261029/summarize.py
```

目录 `/home/renyuanming.rym/v4-flash-128k-concurrency-20261029/` 保存 bench.py、完整 CLI/env 配置、逐请求 SSE/JSON、results/summary/verification.json、comparison.csv、cache-allocations.json、worker-status.jsonl、gpu-telemetry.csv 和服务日志。source.patch 与 provenance.json 保存源码及 native library 来源；软件库与上一轮 offload 适配相同，本轮仅修改实验脚本和文档。两组服务和监控均已停止，GPU 已释放。
