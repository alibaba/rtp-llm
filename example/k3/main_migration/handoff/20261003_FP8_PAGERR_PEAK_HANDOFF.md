# Kimi K3 FP8 PageRR 与长 KV 验证（2026-10-03）

本记录接续 [10 月 1 日的 Prefill 算子择优](20261001_FP8_PREFILL_FINAL_COMPARISON.md)。代码从已验证的 BF16 集成开始，依次保留 FP8 Prefill、峰值激活和 PageRR/DCP 的提交边界；原实验分支 `codex/luohaocheng-k3-fp8-collective-fusion-proto-20260930` 的 `815f34d0e` 是可回退的远端检查点。最终重组分支以 `origin/main` 的 `95ce33f43` 为基底，后者相对旧基底仅增加 context token 指标。

## 固定实现范围

Prefill 仍是 Query CP1、TP8/EP8；PageRR 将 MLA KV 页按 CP8 分布。Decode 开启 DCP8、CUDA Graph 和 Native MTP。Attention 投影与 MLA operand/cache 使用普通 E4M3 FP8；TP 内融合 AG 携带 FP8 值和 scale，融合 RS 输出 BF16；routed MoE 使用原生 MXFP4。MTP 的 hidden 与通信保持 BF16。整模型 Chunk Prefill 的入口和三个模块已移除；历史 MLA prefix 仍按每 rank **6 GiB 展开后 K/V** 的默认预算分片。已接入的峰值措施包括 skip-head-mid、MLA 的 BF16 临时投影提前释放、shared expert 单份 gate/up 输出及按单轮 token 容量配置 workspace。feat 的静态 AttnRes bank、独立 MLA scratch 和 MTP hidden 释放接口没有原样搬入：前者在集成版未测出整模型峰值收益，后两者与当前 collector 的所有权不同；详见峰值锚点记录，不能将它们列为已迁移特性。

## 四层先行验证

四层模型包含 3 层 KDA、1 层 MLA 和 MoE。`P163` 使用固定的 2M KV 种子、约 64K 当前 Q、TP8/EP8 双机 PD 与 MTP；63/63 个种子输出、3 次预热及 1 次测量的选择均与改动前 `P158` 逐项相同。八 rank target GPU span 中位为 **437.863 ms**，改动前 **443.621 ms**，固定 `feat/k3_dev` 的 `F143` 为 **452.523 ms**。进程 PyTorch allocated 峰值中位为 **23.531251 GiB**，与 `P158` 相同，低于 `F143` 的 **24.674423 GiB**。原始 Prefill/Decode 全 rank trace、内存快照和分析在个人 data0 的 `artifacts/k3-fp8-opt-20260927/longkv-integrated-fourlayer-P163-quantkv-timelines-20261003/`。

历史 prefix K/V 的 E4M3 量化复用了通用融合 kernel，避免两次独立 launch。同输入配对热态 A/B 在 327,680 行为 **0.674/1.097 ms**（融合/分离），835,584 行为 **1.660/2.736 ms**；24 种边界输入的 FP8 位和 scale 逐项相同。两种长度的峰值仍分别为 1200/3060 MiB。记录为 `P162-mla-prefix-quant-ab-110gpu1-20261003.json` 和 `P162-mla-kv-quant-candidate-numeric-110gpu1-20261003.json`。

## 完整 93 层、2M KV + 64K Q

固定请求的实际 token 形状是历史 prefix **1,998,848**、当前 Q **65,482**、合计 **2,064,330**。`P164` 的 63/63 个种子输出，以及 10 次同路径预热和 1 次测量输出，与改动前 `P159` 完全一致；最后三次首 token 时间为 12.121575、12.149687、12.145116 秒，满足中位数 ±5% 的收敛条件。八 rank、三条时间对齐的 target GPU span 中位为 **11,241.530 ms**；`P159` 为 **11,354.974 ms**。固定 `feat/k3_dev` 的 `F3` 是同形状的一条热态请求，target GPU span 为 **13,110.672 ms**，不能把单样本当成分布。

同一批 trace 中，集成版 TokenSpeed MLA kernel 的最后三次八 rank 中位再取中位为 **8794.234 ms**；`F3` 的八 rank 中位为 **8840.899 ms**，差约 **-0.53%**，按持平解释。这个数字只计 TokenSpeed kernel；MLA 数据整理、量化和状态合并另计，不能直接与完整 MLA scope 相减。rank1 的历史 prefix 量化累计由 `P159` 的约 **311.3 ms** 降至 `P164` 的约 **174 ms**；`F3` 为 **180.868 ms**。原始核级审计在 `P164-vs-F3-full93-tokenspeed-kernel-parity-20261003.json`。

八 rank 的整进程 PyTorch allocated 峰值中位为 **203.290235 GiB**，与 `P159` 相同，低于 `F3` 的 **205.349668 GiB**（差 **2.059433 GiB**）。四层的绝对峰值也是集成／feat **23.531／24.674 GiB**。两版快照的起点并不等价：feat 在达到峰值前释放了两个记录开始前就已存在、各约 895 MiB 的大张量，直接计算“绝对峰值减起点”会将 feat 的本次新分配低估约 **1.751 GiB**。从 alloc/free 事件重建“快照开启后新分配且在峰值时仍存活”的八 rank 中位，四层和 93 层均为集成 **13.089 GiB**、feat **13.385 GiB**，集成少约 **0.296 GiB**。这与绝对峰值方向一致，但它只统计 PyTorch allocator 的记录区间，快照没有 Python 调用栈，不能进一步归因到某个模块或非 PyTorch 分配。逐 rank 可复算脚本及 JSON 为个人 artifact `analyze_peak_history_fourlayer_full93_20261003.py` 和 `peak-history-fourlayer-full93-reconciled-20261003.json`。`P164` 八份 Prefill trace、八份 Decode trace、八份内存快照及请求对齐分析在 `longkv-integrated-full93-P164-quantkv-timelines-20261003/`；`F3` 对照在 `longkv-feat-full93-F3-timelines-20261002/`。以上目录均位于 `/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927/`。

## 完整 smoke 与边界

改动前的 `P162f` 在 113 Prefill 并发 16、111 Decode 并发 8 下完成 **169/169** 条正式请求，独立答案审计 **169/169**；原末尾 3 条长输出 repeat 和 3 条已知超过 5 分钟的 case 共 **6 条未发送**，不计通过。`P164` 的正确性 smoke 使用 Prefill 并发 8，在第 50 条 batch case 收到 429，因此它不是最终验收。

PageRR/DCP 固定源码检查点 `815f34d0e` 的 `P165i` 在 113 Prefill／114 Decode、TP8/EP8、Prefill Query CP1、Decode DCP8、CUDA Graph、Native MTP 下完成 **169/169** 条正式请求；独立复核 `checked=169, runner_cases=169, skipped=6, errors=[]`。169 条均经双机 PD 路由并返回 HTTP 200；18 条记录了 MTP draft，18 条跨越 Decode KV 页边界。六条未发送用例中，三条长输出 repeat 按要求暂缓，三条历史超时 case 跳过；`full_original_suite_passed=false`。逐 case 原始结果与答案审计在个人 data0 的 `artifacts/k3-fp8-opt-20260927/smoke-93layer-fp8-page-rr-P165i-final-113114-20261003/`。其中 `result.json` 的旧 `not_applicable` 模板仍写着 DCP/PageRR；它不能作运行配置证据，应以本次 launcher、Decode 日志中的 `[MLA_DCP] backend=a2a tp=8`、Prefill Query CP1 配置和 trace 为准。

四锚点分支 `dd805583e` 在此检查点的运行源码上仅叠加新版 main 的 context-token 指标文件。113/114 分别以个人账号在 `lhc_GPU` 内用 CUDA13/SM10x 重新构建，均为 **23,714/23,714 action 成功**；113 的 PageRR cache、prefix kernel、Prefill 参数三个定向 Bazel 测试均通过。两端重新通过本地 data0 的 target 96 shard、MTP 9 shard 和 FastSafetensors guard，然后在精确源码提交 `dd805583e` 重新执行 `P166` 完整双机 smoke：**169/169** 条正式请求、独立答案审计 `checked=169, runner_cases=169, skipped=6, errors=[]`，全部有 PD 路由，18 条有 MTP draft 且 18 条跨 Decode 页边界；最大单条耗时 **52.257 秒**。六条原约定 case 仍未发送，`full_original_suite_passed=false`。原始结果在 `artifacts/k3-fp8-opt-20260927/smoke-93layer-fp8-page-rr-P166-dd805-113114-20261003/`，113→115 副本的 `result.json` 与审计 JSON 的 SHA256 分别为 `3f3ffb3cb8ff1a2fed05cb5f1d8c4f6985a7633bbbb4420bce010e2b90b94213` 和 `0e7d2cc7f90071ef132d0f97ea01abc0819d35ef38e4ae9b99eeefef14b0adcf`。`P166` 的 Decode 日志再次确认 `[MLA_DCP] backend=a2a tp=8`；CUDA Graph 启用配置与 Prefill Query CP1 同 launcher 保存。文档后续修订不改变该运行源码。

113/114 两端使用个人账号、data0 本地权重和 FastSafetensors。114 的 target 配置与 index SHA256 与 115 一致，96/96 个 shard SHA256 也与已验证清单一致；MTP 9/9 shard SHA256 一致。两端普通网络互通，114 的八个 `mlx5_bond` HCA 均为 ACTIVE。111 在切换时被另一位用户的活跃 8 rank 服务占用；115 的 GPU0 在 PyTorch 中不可见。110 虽然通过同样的权重校验，但 GPU0 留有无 OS PID 的 NVML 显存占用，不适合独占性能测量。

110 的两次与 114 的前两次 Decode 启动均在 CUDA Graph 首次编译 TokenSpeed MLA Decode 时触发同一个 Cutlass DSL 错误：旧 `mla_decode_fp16.py` 的 SHA256 为 `7c099ab21b25f9a263c616e9794065778275d05adc995294811d9b4c9d9a3822`，动态 `if` 内将 `cta_m_rows` 从 `None` 改为整数。110 的启动日志明确记录此错误，因此其启动失败不能归因于残留显存。113/111/115 已使用修复后的同一源文件（SHA256 `a2b4447bba5f22b6b87b240633e6d1e475f134d33518aa504a4fe47a58e1c0d3`）；114 保存旧文件后同步该已验证版本，再启动 `P165i`。该 runtime 位于个人 data0 artifact，尚未由 Git 分支自动安装，复现时须固定其哈希。

## PageRR 后的 93 层 64K Prefill

`P165i` 服务启动后以同一条 65,536 token PD 请求充分预热 14 次，最后三次耗时落在中位数 ±5% 内；随后对六条正式请求采集八 rank timeline。只取补足 profiler 窗口的请求之前、八 rank 均完整的最后三条正式请求，target 最大 rank GPU span 分别为 **1467.836、1462.664、1470.228 ms**，中位 **1467.836 ms**；target＋draft 中位 **1534.781 ms**。六条正式请求均完成 PD/MTP，Prefill 前缀没有复用。八份 trace 从 113 复制到个人 data0 后逐份 SHA256 与源端一致；原始 trace、请求、预热、独占预检和按模块归因在 `artifacts/k3-fp8-opt-20260927/timeline-64k-page-rr-P165i-full93-113114-20261003/`。独占预检显示两端八卡利用率均为 0%，计算 PID 只有本任务；正式窗口没有连续的外部进程监控，因此不作更强的独占断言。

固定旧版 `feat/k3_dev` 同形状热态 target 中位为 **1475.576 ms**，这次 `P165i` 快约 0.5%，差值很小且来自不同时段／机器组合，按接近持平解释；旧集成无 PageRR 服务为 **1388.811 ms**，不能将配置与权重分片差异归因给 PageRR。尤须校正 shared expert 的作用域口径：固定 feat 把两次权重 AllGather 放在 `RTP::moe.shared_expert` 外层标注中，当前集成版放在 gate/up、down 子标注内。rank1 同一完整请求中，按通信＋两个 GEMM＋激活合并，固定 feat 为 **168.275 ms**，`P165i` 为 **168.033 ms**；单看 gate/up 或 down 的标注会误判成较大的退化。三方四层逐算子择优和旧版全层对照见 10 月 1 日记录；vLLM 固定服务的 attention 投影实际为 BF16，并把 65,537 token 拆成约 32K×2，不能当作与 RTP E4M3 投影完全同精度、同分块的逐算子胜负。

精确提交 `dd805583e` 的 `P166` 随后在同一 113/114 双机服务上重新采集全 rank timeline。14 次相同输入预热的末三次首 token 为 **1551.709、1546.060、1546.804 ms**，收敛到中位数 ±5%；六条正式请求均 HTTP 200、65,536 token、无 Prefill 前缀复用、完成 PD 和 MTP draft。十条随后发送的请求只用于结束 profiler 窗口。按八 rank 时间戳选出窗口内最后三条完整正式请求，target 最大 rank GPU span 为 **1467.134、1467.643、1465.277 ms**，中位 **1467.134 ms**；target＋draft 中位 **1534.674 ms**。这与 `P165i` 的 **1467.836/1534.781 ms** 基本一致。八份约 220 MB 的 trace 已在 113→115 复制后逐份 SHA256 核对；原始请求、trace、哈希清单和逐模块结果位于 `artifacts/k3-fp8-opt-20260927/timeline-64k-page-rr-P166-dd805-full93-113114-20261003/`。该精确提交比旧固定 feat 的 target 1475.576 ms 约快 0.57%，差距仍不足以排除跨时段／机器的波动，按持平解释；比旧 vLLM 服务配置的 1715.716 ms 快，但其 BF16 投影与双 chunk 不适合作等精度算子归因。
