# K3 双机 smoke 回归

双机启动入口固定使用 Prefill TP8/EP8、Decode DP8/KTP8/EP8、Decode CUDA Graph、原生 MTP、chunk prefill、chunkwise RDMA 和 RDMA。历史 KV 展开仍受 expanded-KV 预算控制。Eagle3、Prefill TP 非 8、关闭 chunkwise RDMA 会在启动前被拒绝。Page-RR 是独立设置，保留原有默认值 0。

`SMOKE_SUITE=all` 使用完整模型做答案验收；四层模型的 `flow` 仅检查连通与分块流程。默认 batch=4 且 require-MTP 时，all 有 34 个阶段、117 个正式请求；另外保留有界 RDMA 预热。请求记录中的 `phase` 区分预热与正式请求。

## 新增与优化

| 用例 | 构造 | 判定 |
|---|---|---|
| owner_records_rotate_0/3 | 全部 8 个 owner，改变请求到 owner 的排列；每请求独有档案标签 | 从三条记录中按 key 检索 value，严格 JSON 相等，核对返回的 owner 地址 |
| owner_last_only | 仅 owner 7 接收请求 | 唯一答案正确；Decode 日志另证明 rank0 空闲、rank7 活跃 |
| historical_four_squares | 80–83 平方，owner [0,0,1,2]，短输入 | 最终 content 仅允许预期数字和首尾空白，禁止同时夹带其他答案 |
| graph_slot_wave_0/1/2 | owner 7 分别提交 7、5、6 个不同字段复制请求 | 答案严格对应本请求；Decode 日志必须出现同一 bucket 8 的有效数 7→5→6 |
| dp_rolling_refill | 窗口 8，16 个不同输出长度的请求，完成一个后补入下一个 | 所有结果与 owner 正确；保存每请求响应。HTTP 滚动窗口不被宣称为确定的引擎 batch |
| prefix_A_seed/B_partial/A_return/AB_mixed | 同一公共前缀，尾部记录不同，A→B→A 后并发 A/B，切换 Decode owner | 服务 tokenizer 计算公共 token 前缀；准确检查页对齐 reuse 长度、当前尾部答案 |
| padding_tail_1/7_cold/hit | 服务 tokenizer 精确构造 chunk_budget+1 和 +7 的输入，冷请求后换 owner 再次请求 | 检查实际 input_len、reuse；Prefill 日志必须观察到补 7 和 1 token 的真实 round |
| whole_chunk_single_miss/hit | 合并原 mtp_chunk_prefill_miss 的检查到单请求长输入 cold/hit | 同时要求 chunk 和接受过 MTP draft token，减少一个重复长 cold 请求 |

数学题采用完整答案匹配。非数学题由请求内档案生成期望答案，不依赖常识、外部数据集或另一个模型评分。JSON 答案只接受期望字段和值，拒绝重复字段、额外字段、解释与代码块。

RDMA 预热仅重试连接异常和 HTTP 408/429/502/503/504。模型答错、缺少 PD 元数据、错误 owner、格式错误、cache 判定失败均直接终止；即使同批另有连接失败，也优先报告语义错误。正式 case 不重试。

发起预热前，两端都必须等全部 rank 的 RDMA transport 和模型 gRPC listener 就绪，避免 HTTP health 已通过、某个 Decode listener 尚未启动时发生路由重试。

每请求在验证前保存输入和原始响应，验证后更新结果。单个请求失败不会丢弃已完成的同批请求。精确构造的输入另保存 tokenizer token IDs，便于复现。

## 运行时证据

双机入口设置 `KIMI_K3_SMOKE_EVIDENCE=1`。两个 Python 运行时位置增加了受此开关控制的主机端日志，不改变张量计算或调度逻辑：

- `kimi_k3.py` 在 chunk round 提交后记录逻辑/物理 token 数与请求数。
- `ktp_step.py` 记录每个 KTP rank 有请求时的 step 序号、有效 batch、物理 batch、Graph bucket、模式和 token 宽度。空闲轮询不计入序号，确保请求结束后可以读取稳定的多 rank 记录。

`kimi_k3_smoke_runtime_evidence.py` 在两端分别生成 `smoke-runtime-coverage.json`。Prefill 要求 padding 算术正确且实际出现 padding=1/7。Decode 要求 8 rank 逐步记录一致、全部 owner 活跃、rank0 空闲时 rank7 活跃、Target Verify 使用 bucket 1/2/4/8、bucket8 有效数按顺序出现 7→5→6，并存在实际 Graph replay 日志。日志副本按 rank/step 去重，冲突副本判失败。

每个 all-suite 必须同时通过答案检查和两端运行时检查。若调度没有形成预期状态，会报告覆盖不足，不能用请求并发数替代真实 batch 证据，也不能靠重复执行洗掉失败。

这些证据证明对应引擎状态出现过；当前没有把每条 KTP event 逐请求关联，也没有证明每一步都实际 replay。全局 replay 日志与逐步 Graph 选择证据分别记录，不能据此声称完全排除了中途 fallback。7→5→6 是同一物理 bucket 的跨请求槽位复用，并不代表某个指定请求持续占据同一槽位。

## 尚未覆盖的分支

- MTP 全部拒绝、部分接受、全部接受及拒绝后的 cache frontier。当前仅检查实际接受过 draft token，不能从 prompt 推断每类接受结果。
- dummy 不发布 cache 的逐请求、逐 block 证明。padding round 和随后的 hit/答案已覆盖，但发布元数据完整审计仍依赖引擎测试。
- PD/RDMA 中途取消、迟到完成事件与 block 复用。
- cache 淘汰及 CPU cache 恢复、单 owner 超时恢复、长期反复复用。
- 原历史多请求内部 round=63746 的精确同批复现；当前单请求边界稳定覆盖 TP padding=1/7，HTTP 四请求不能证明同批 Prefill。

这些分支需要可控引擎测试或故障注入，应独立补充，不计入本版通过范围。

## 无 GPU 检查

在仓库根目录，使用 Python 3.10+ 与 Linux Bash：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -S -m unittest \
  example.k3.kimi_k3_full_model_two_host_pd_smoke_driver_test \
  example.k3.kimi_k3_full_model_pd_cases_test \
  example.k3.kimi_k3_long_prefix_case_test \
  example.k3.kimi_k3_smoke_regressions_test
```

这组单测验证测试程序、失败保留、重试策略、用例构造、并发补入和日志判定。它不替代完整模型在双机上的 GPU 验收。

## 1M 与字段复制的预算

完整模型的长前缀用例默认 1M tokens、历史 KV 展开预算 6 GiB。Decode 保留 32 个 KDA 状态块以覆盖并发请求和 speculative 预留页，hybrid cache 预算为 29000 MiB。SSM replay 新增约 465 MiB 预留后，28500 MiB 配置实际只有 244 个 FULL blocks（999424 tokens），不足以完成 1M 用例；追加预算保留 32 个 KDA 状态块及长前缀余量。

字段复制用例为 reasoning 与最终 JSON 一起预留输出 token。16 个字段值的旧 512-token 上限曾导致 reasoning 或 JSON 被截断；增大预算后仍要求完整 JSON 精确相等、owner 正确，正式请求不重试。

Prefill 的 HTTP 并发上限为 32，容纳不均匀 DP 用例同时提交的 10 个请求（4+3+2+1）；若设为 8，入口会拒绝其中两个请求。Decode 每个 owner 的并发上限为 8，与 Graph 最大 batch 桶一致，避免按并发 32 为 MTP 初始化 128 个物理 token 的额外显存峰值。滚动补位仍执行全部 16 个请求，保持 8 请求窗口；Graph 的 7→5→6 波次不变。CPU 回归从实际用例收集并发 batch，检查 Prefill 入口能同时接纳、Decode 的 owner 容量足够。

Decode 固定 NCCL_MAX_CTAS=8，限制通信内部缓冲区占用。8 卡小复现中，默认 32 条 P2P 通道在剩余 3 GiB 显存时仍于首次 AllToAll 报 CUDA OOM；限制到 8 条后，剩余 1.5 GiB 时 AllGather、AllToAll 与 CUDA Graph 捕获和回放均通过。该配置用于正确性 smoke，不据此给出吞吐结论。
