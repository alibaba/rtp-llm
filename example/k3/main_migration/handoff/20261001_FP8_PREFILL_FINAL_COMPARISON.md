# Kimi K3 FP8 Prefill 算子择优与最终验收（2026-10-01）

## 固定代码与验证范围

集成分支 `codex/luohaocheng-k3-fp8-collective-fusion-proto-20260930` 保留两个提交锚点：性能择优 `34d6d58d9ed90bc3260a7e39d503bb13ca2fd9fd`，峰值激活生命周期 `2b6218d56ccb15a977de4324a18f952a93f59d32`。下述最终服务、smoke 和 timeline 均使用第二个锚点，两端以个人账号在各自 `lhc_GPU_k3_rdma_20260929` 中用 CUDA13/SM10x 构建。114 执行 Prefill，115 执行 Decode，TP8/EP8、target FP8 E4M3 attention 与 KV、原生 MXFP4 MoE、BF16 native MTP；TP 内融合 AllGather 传 E4M3 值和 UE8M0 scale，融合 ReduceScatter 输出 BF16。FastSafetensors 通过个人 `/data0` 元数据视图读取已核实的 3FS shard，读取进程使用任务内 64 路 pread，没有修改全局 3FS 配置或 loader guard。

固定对照为 `feat/k3_dev` 提交 `a9bf762e878fc54ee9176da5c34ffbe6babc8d45` 和 vLLM K3 源 `3df4ae153`。四层负责候选筛选；完整 93 层只在代码固定后复核总体 Prefill 和模块趋势。三方均为双机 PD、TP8/EP8，114/115 GPU 测量独占。两版 RTP 开启并记录 Native MTP draft。vLLM 只测 target，NIXL consumer 的 successful transfer 由 0 增至 120、failed 保持 0；其服务实际完成 PD，而非单机代测。固定 vLLM 模板把相同消息内容计为 **65,537** prompt token，比 RTP 的 65,536 多 1 token，且完整模型把当前输入分成两个约 32K chunk。vLLM 的 attention 投影在实测中是 BF16 `aten::linear`；它的 FP8 范围包括 MLA operand/cache 和 MoE，不是等精度的 FP8 投影候选。因此三方总体时间可作服务配置对照，微小差值或投影数值不能当成同精度同分块的优劣。

## 四层算子选择

四层包含 3 个 KDA 和 1 个 MLA，64K 输入、相同路径预热后逐项比对。原始配对样本、逐 rank trace 和数值结果见个人 artifact `/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/fp8-prefill-operator-selection-ledger-20260930.md` 及其中引用的文件。

| 功能块 | 等价候选与观测 | 固定选择 |
| --- | --- | --- |
| KDA paged short-conv、QKV/beta | 原分离路径 1.324 ms，通用 paged 融合 0.524 ms，同输入 Q/K/V 最大绝对差 0.00390625，beta/cache 一致。feat 的 paged kernel 与通用移植版核心逻辑一致；93 层各调用 69 次，单 rank 核累计 35.281/35.965 ms，跨服务差不能证明 feat 有不同的更快算法。 | 保留通用 paged 融合和 cuLA KDA；四层 PD flow 已通过。 |
| KDA output norm/gate/FP8 producer | 同卡 30 对热态 A/B：集成 0.037296 ms，feat 0.046912 ms；值及 scale 逐项相同。 | 集成版融合 producer。 |
| MLA TokenSpeed FP8 与历史 KV | 同输入 64K E4M3 Q/K/V，一次 64K 6.518 ms，两次 32K 6.793 ms，输出相同。历史 prefix 默认每 rank 6 GiB 展开后 K/V 预算，小预算 0.05 GiB 的多片路径已过四层双机 flow。 | 单次 64K 当前输入＋按需历史 prefix 分片；不迁整模型 Chunk Prefill。 |
| MLA QKV 量化与 cache 写入 | 同卡相同输出：RTP 分离 0.498596 ms，vLLM fused 0.369052 ms，通用 fused 0.364758 ms；四层服务 trace 的融合核中位 0.361379 ms，旧量化＋cache 写入累计 0.479075 ms。 | 通用 FP8 epilogue；历史 prefix 和非单位 KV scale 保留已验证回退。 |
| MLA output gate producer | 100 对配对中位：集成 0.031904 ms，feat 0.031792 ms，差值处于波动；值和 scale 相同。 | 保留集成实现。 |
| FP8 AG＋输入 GEMM；输出 GEMM＋BF16 RS | feat 的前段使用 PyTorch symmetric-memory E4M3 值/scale pipeline 与 DeepGEMM，后段使用 `fp8_gemm_rs_nt`。两段已移入通用 `Fp8CollectiveProjection`，数值测试覆盖 TP8 local rows 128 和 8192，四层 PD flow 通过。固定 vLLM 是 BF16 NCCL AG/RS＋独立 BF16 投影，按完整阶段对照。 | 保留通用融合实现；完整模型 barrier 等待的差距见下一节。 |
| MoE router、打包、routed experts、shared expert | Router 同卡相同权重/输入的 50 对测试选择非连续 `torch.mm` 和现有 top-k；集成完整 router 0.141952 ms。MoE FP8 pack 30 组 CUDA Graph：集成 0.016999、vLLM 0.017123、feat 0.030929 ms，输出逐项相同。原生 MXFP4 DeepGEMM 2.8 同路由八 rank 输出 SHA 相同，RTP/vLLM 中位约 2.583–2.589/2.586–2.592 ms；shared gate/up 和 down 同卡差别在波动内。 | 保留集成 router、向量化 pack、原生 MXFP4 routed GEMM 和单份 gate/up views；不迁整包 MoE。 |
| Dense gate/up | 同机同形状四路对比，分离 RTP 9.079 ms、feat 风格 9.090 ms、合并 RTP 9.033 ms、vLLM 风格 9.036 ms，输出一致；合并路径瞬时多约 528 MiB。 | 采用合并 gate/up，另以生命周期优化控制峰值。 |

## 93 层双机热态对照

每版先做至少 10 条相同路径预热，正式请求从八 rank GPU timeline 取 target 最大 rank span；RTP 两版还取 target＋draft。选取 profiler 起步后的三个完整 all-rank 请求，三次 target span 均在中位数 ±5% 内。vLLM 四条正式请求的最大 rank span 均在约 1.714–1.718 s。HTTP 包含 PD 传输，不用于 Prefill GPU 时间。集成版 114/115 的采样窗口由 16 次进程监控覆盖，未见外部 GPU 进程；之前一组与外部进程重叠的采样只保留诊断，不进入下表。

| 完整 93 层目标路径 | target GPU span 中位 | target＋draft GPU span 中位 | 说明 |
| --- | ---: | ---: | --- |
| 集成 `2b6218d` | **1390.957 ms** | **1456.998 ms** | 10 次预热、6 条 profiled，选完整 all-rank 第 4–6 条；64K 一次处理。 |
| feat `a9bf762e` | 1475.576 ms | 1540.494 ms | 11 次预热、6 条 profiled，选完整 all-rank 第 4–6 条；64K 一次处理。 |
| vLLM `3df4ae153` | 1715.716 ms | 不适用 | 11 次预热、4 条 profiled；约 32K×2，实际 prompt 65,537 token，未开 MTP。 |

集成 target GPU span 比固定 feat 低约 5.73%，比该 vLLM 服务配置低约 18.93%。这说明**当前完整模型服务配置**的 target Prefill 更快；vLLM 的投影精度、分块和 1 token 差异限制逐算子归因。

融合段按每层 CPU 标注映射 CUDA launch，再取该层相关 GPU kernel 的最早开始到最晚结束，汇总 93 层；这不是请求关键路径，GPU stream 重叠和等待均保留在区间里。vLLM 将每条请求两个 chunk 的分离 BF16 阶段区间相加。中位来自正式请求的 8 rank。

| 等价阶段，93 层累计 GPU 区间 | 集成 FP8 融合 | feat FP8 融合 | vLLM BF16 分离 |
| --- | ---: | ---: | ---: |
| AG＋attention 输入 GEMM | 158.467 ms | **152.387 ms** | 385.161 ms |
| attention 输出 GEMM＋RS | 150.275 ms | **146.149 ms** | 214.462 ms |

**这两段尚未达到“集成版每段时间均快于 feat”的字面目标。** 已检查 feat 与集成代码：前段均调用 `_pipelined_multi_all_gather_and_consume`，后段均调用 `fp8_gemm_rs_nt`；rank 1 的 93 层 DeepGEMM 前段累计约 157.30/157.50 ms、后段约 116.18/115.93 ms，接近。另用固定 feat 源中的原函数与通用类方法做同形状、同权重、同 FP8 输入的 TP8 直接 A/B：每版预热 10 次、交替测 30 对，八 rank 中位的 AG＋GEMM 为通用/feat **2.351688/2.351760 ms**，GEMM＋RS 为 **1.500736/1.503376 ms**，数值结果相同；差值处于波动内，未见 feat 有更快的未迁入计算实现。原始样本在个人 artifact `fp8-collective-exact-feat-vs-generic-tp8-115-20261001.json`。全层服务的前段 symmetric-memory barrier 累计 11.445/6.759 ms，后段 DeepGEMM RS barrier 累计 19.943/14.998 ms。对齐八 rank 同一请求的每层融合启动点后，AG 前 rank 到达偏差中位为集成/feat **0.128/0.043 ms/层**，RS 前为 **0.365/0.238 ms/层**；集成版 rank 0/4 的等待尤其明显。观测差距主要与 rank 到达同步点的偏差一致；前一层 SP8/TP8 调度可能影响到达偏差，但尚无独立因果 A/B。保留已经数值验证的通用融合实现，不为追逐跨服务等待差而引入模型专用代码。逐算子最优的结论只覆盖已经完成同形状、同精度直接 A/B 的候选；不能把总体更快表述为每个阶段都更快。

KDA core 全层核累计集成/feat 中位约 206.147/201.052 ms；rank 1 都是 69 次同类 cuLA kernel，主要差距分散于相同算术核的 2% 左右时钟/调度波动，分页短卷积核心逻辑一致。MLA core＋融合 epilogue 集成约 154.251 ms，feat 原 MLA pipeline 154.552 ms；vLLM 的约 142.878 ms core 与其单独 FP8 producer 9.861 ms 不能直接与 RTP 标注相减，且分块不同。Routed MXFP4 专家核累计集成/feat/vLLM 约 408.465/464.436/425.031 ms，真实路由分布与包装不同；同路由同权重的原生核 A/B 在四层筛选时已确认 RTP 与 vLLM 相当。模块累计不能加和成请求关键路径。

## 正确性、显存与遗留边界

固定 `2b6218d` 的 93 层 FP8 双机 PD 限时 smoke 共执行 **121/121**，独立答案复核 `checked=121, errors=[]`；覆盖 PD 状态交接、MTP draft、短输出答案、乱码、重复吐字和截断检查。原最后三条长输出 repeat 与三条已知超过 5 分钟的 case 共 **6 条跳过**，不计通过，也不声称原完整 suite 全通过。结果在个人 artifact `smoke-93layer-fp8-peak-anchor-114115-20261001-r2/independent-final-audit.json`。

第二锚点的 FP8 MLA 生命周期使 TokenSpeed 入口少保留 384 MiB 的 BF16 投影（融合路径）或 480 MiB 的 BF16 K/V（回退路径）；最初的存活量测量使用替身 attention。后续同 64K 形状、真实 FP8 epilogue＋TokenSpeed 的单卡配对 A/B 中，`torch.cuda.max_memory_allocated` **减少 192 MiB**，30 对热态 CUDA event 中位 **6.1494/6.1512 ms**，未见可分辨的耗时变化；投影 GEMM 仍是等形状分配替身。真实四层 FP8＋MTP 双机 PD 服务再做 10 对预热、30 对交替测量：八个 rank 的 target MLA 方法作用域 allocator 峰值，配对中位均减少 **192 MiB**；80 条响应内容相同，均完成 PD 交接与 7 轮 draft。原始记录和作用域边界见峰值锚点文档及个人 artifact `mla-service-peak-fourlayer-114115-20261001-r5/`。这仍不能证明整次请求的峰值下降。最终完整服务的 NVML 20 ms 采样 470 次，在 64K 请求前后所有 rank 的 `memory.used` 均无可见增量；该指标包含常驻权重、KV 和 allocator reserve，不能分辨实际激活峰值。首次四层 NVML A/B 因测试代码版本不含融合路径改动，已明确作废。`skip-head-mid`、历史 KV 每 rank 默认 6 GiB 分片、shared gate/up 单 buffer 和 chunk 容量 workspace 已在当前代码；整模型 Chunk Prefill、无证据收益的 AttnRes bank 和额外 scratch 释放未迁。

O01/O02/O03 与本轮 Prefill 有关的 cuLA、paged conv、FP8 TokenSpeed、target FP8 投影/融合通信和 native MTP 精度隔离已覆盖。feat 的 `linear_serial_replay` Decode endpoint 与独立 BF16 dense FlashMLA 不在本轮 FP8 Prefill 择优范围；Decode CUDA Graph＋MTP 已过正确性，但 Decode 性能择优仍暂缓。TP16 的源码分片/scale 分支已检查，未做设备验证。

原始材料主要位于个人 artifact `/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/`：集成全层 `timeline-64k-peak-anchor-full93-114115-r4-20261001/`（114），feat 全层 `timeline-64k-feat-full93-114115-r3-20261001/`（114），vLLM 全层 `timeline-64k-vllm-full93-114115-r4-20261001/`（114），逐模块副本 `integration-full93-r4-all-rank-module-audit-20261001.json`、`feat-full93-r3-all-rank-module-audit-20261001.json`、阶段与 rank 到达归因 `*-fused-stage-window-audit-20261001.json`、`*-fused-collective-rank-arrival-audit-20261001.json`、vLLM 分离段 `vllm-full93-r4-separate-stages-allrank-20261001.json`、独占窗口 `integration-full93-r4-exclusive-interval-audit-20261001.json`、真实 TokenSpeed 峰值与计时 `mla-real-tokenspeed-64k-allocator-peak-r3-115-20261001.json`（115）。服务采样后，114/115 本任务的 RTP 服务进程组已清理。
