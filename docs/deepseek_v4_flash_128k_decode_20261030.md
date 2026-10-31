# DeepSeek V4 Flash：128K 真实 decode 并发与 offload（2026-10-30）

D 单卡缓存池 HBM 分配从 43.16 降至 18.80 GiB，Host 分配 27.41 GiB。128 并发观察到 decode TPS 9031 → 8156，TPOT P50 13.45 → 14.58 ms。

完整 DeepSeek-V4-Flash-0731，输入固定 131,072 token、输出固定 256 token。8 张 L20D：GPU 0–5 为三个独立 CP2/EP2 prefill 副本，GPU 6–7 为 TP1/DP2/EP2 decode。FP8 KV、MegaMoE SE、DSpark gamma=3；D 开 CUDA Graph，P eager。

通过 CPU gRPC 代理等待每个请求的真实 D KV LOAD 回执与 P GENERATE，再同时放行 decode。保留实际 KV 传输与原始生成参数，生产 scheduler 和模型源码均未改动。因此这里的 128 并发确实对应 D 端同时运行的 128 个长请求，而非仅有 128 个客户端排队。

本报告含共享 GPU 的测量。背景服务计算活动有变化，开关差异无法完全归因于 offload；整卡显存包含其他进程，但缓存池分配只统计本实验。

主对照的 resident 使用此前已经完成的干净基线。本次 resident 重测的 C4 出现外部计算干扰，这些样本另存，不混入主对照；其他档位的重测用于复核。原 resident 运行在正式矩阵完成后，计时外的 DEBUG 诊断中断，原始状态保留 aborted；后续独立补充了 Graph 检查，与基线源码/native 哈希一致。

每档每组 3 个正式批次，共 3C 个请求；正式对照合计 1,512 个请求。TPS 取三批中位数；TPOT P50/P99 汇总三个批次的请求平均 TPOT，使用线性插值。TPOT 从首个 D 输出到最后输出，按期间新增 token 数计算，包含 gRPC 交付，不包含 prefill、KV 传输、barrier 等待、P 输出和首个 D 输出。TPS 是各批 decode 区间的产出速率，包含开始和排空的偏差，不是持续在线端到端 TPS。

表内数值均为 **关闭 offload → 开启 offload**。

| D 总并发 | 请求数/组 | Decode TPS tok/s | TPS 变化 | TPOT P50 ms | TPOT P99 ms | 逻辑 KV 占用峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 12 | 710.3 → 662.8 | -6.7% | 5.33 → 5.45 | 6.07 → 6.03 | 2.78% → 2.78% |
| 8 | 24 | 1317.0 → 1266.7 | -3.8% | 5.91 → 6.20 | 6.72 → 6.53 | 5.57% → 5.57% |
| 16 | 48 | 2257.6 → 2055.1 | -9.0% | 6.54 → 7.22 | 7.19 → 25.25 | 11.14% → 11.14% |
| 32 | 96 | 3804.7 → 3675.0 | -3.4% | 7.76 → 8.00 | 8.33 → 8.59 | 22.27% → 22.27% |
| 64 | 192 | 6189.9 → 5891.3 | -4.8% | 9.67 → 10.10 | 10.32 → 10.80 | 44.54% → 44.53% |
| 128 | 384 | 9031.5 → 8155.7 | -9.7% | 13.45 → 14.58 | 14.43 → 15.71 | 89.11% → 89.08% |

低并发每组样本较少，例如 C4 只有 12 个请求；这些 P99 不能当成稳定的生产尾延迟。TPOT 分位数也不是逐 token 间隔的分位数。C16 offload 的 25.25 ms P99 来自第三批；例如请求 12002 从输出计数 5 到 256 耗时 6.543 秒，请求平均 TPOT 为 26.07 ms。已逐条用保存的 token 计数/时间戳复算，未剔除尾部。现有共享负载记录不能确定停顿由 offload、其他 GPU 计算还是流式交付造成。

缓存池在启动时预分配，下面的每卡分配适用于全部六档并发。HBM 包含各 KV/状态池与 offload resident KV，不包含 Python 映射表、临时 workspace、权重及 Graph 内存。

| 方案 | 角色/单卡 | 缓存池 HBM GiB | 缓存池 Host GiB |
|---|---|---:|---:|
| resident | P | 39.121 | 0.000 |
| resident | D | 43.161 | 0.000 |
| offload | P | 39.121 | 0.000 |
| offload | D | 18.800 | 27.407 |

两组逻辑容量固定为每张 D 卡 36,864 个 block（含 sentinel），block=256 个原 token。SWA/状态池各 512 block，并发上限 128。offload 仅作用于 CSA KV，Host 预算 32 GiB，测试强制 HBM backing H=2（含 sentinel），resident/selection 上限各 262,144 行。Indexer、HCA、SWA、draft 等继续使用 HBM。此配置用于同容量下验证 Host 路径和测量开销，不代表生产自动容量配置。逻辑占用按 block 计量，含请求的尾部预留。resident 在 C128 仍能容纳全部请求，本轮未测出整卡 HBM 的最大并发容量。

每个正式批次的独立 worker_status 采样峰值均达到目标并发，两个 D rank 各承担 C/2 个请求。采样间隔 250 ms；它是 RUNNING 请求数，不是逐步执行 batch 的完整 trace。另检查各请求 decode 区间存在共同重叠窗口。

| 并发 | 最短全并发窗口：关 → 开 s | 预填充/KV 等待 P50：关 → 开 s | 端到端 TPS：关 → 开 |
|---:|---:|---:|---:|
| 4 | 1.231 → 1.270 | 3.77 → 3.78 | 195.5 → 191.7 |
| 8 | 1.351 → 1.409 | 5.69 → 5.69 | 281.6 → 278.5 |
| 16 | 1.457 → 1.581 | 11.33 → 11.35 | 310.4 → 305.5 |
| 32 | 1.743 → 1.790 | 20.80 → 20.78 | 354.5 → 354.7 |
| 64 | 2.125 → 2.246 | 41.49 → 41.57 | 369.9 → 368.1 |
| 128 | 2.966 → 3.233 | 80.73 → 80.58 | 386.7 → 385.8 |

端到端 TPS 包含该批 prefill、传输和等待；这是预备整批 decode 的实验，不能据此声称 6P 能持续供给 128 个在线 128K/256-token 请求而不排队。

两组都验证输入/输出长度、P/D 成功、prefix reuse=0、DSpark 执行及 CUDA Graph 实际 replay。两组 native library 和模型源码 patch 哈希相同。额外的 DEBUG Graph 诊断在计时外运行。

数据目录：

- resident: `/home/renyuanming.rym/v4-flash-6p2d-20261029/resident`；宿主机进程检查 56 次。
- offload: `/home/renyuanming.rym/v4-flash-6p2d-20261029/shared-20261030-021339/offload`；宿主机进程检查 61 次。

汇总 JSON/CSV：`/home/renyuanming.rym/v4-flash-6p2d-20261029/shared-20261030-021339/reference-comparison`。原始目录保留完整 CLI/env、代码来源哈希、输出 token IDs/事件时间、KV 分配、Graph replay、worker_status、GPU 利用率和宿主机 GPU 进程记录。
