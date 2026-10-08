# Kimi K3 FP8 集成工作交接（2026-10-08）

这份记录交给下一会话。当前代码和测试改动已整理成提交并推送；本文写作时没有启动新服务或重新跑 GPU。**先读本文件和证据，再决定是否继续开发。** 旧文档描述的是当时的代码和测试，不能替代这里记录的最新固定版本。

## 从哪里接手

| 项目 | 固定值 |
| --- | --- |
| 仓库 | `git@github.com:alibaba/rtp-llm.git` |
| 当前集成分支 | `codex/k3-decode-verify-final-anchors-20261008`，写本文前远端与本地均为 `44e892a8cbd56e9e344ed5d5d8462d856c392bf9` |
| 115 上的工作树 | `/data0/luohaocheng.lhc/worktrees/rtp-llm-k3-decode-verify-final-anchors-20261008`；写本文前干净 |
| 当前分支所基于的 main | `95ce33f438`；**这是固定基底，不表示此后 main 没有新提交** |
| 固定对照 | `origin/feat/k3_dev` 的 `edaaf0d8aeea61593729dec43a811e29217c32dc`；比较时不要把移动的分支名当版本号 |
| 旧实验检查点 | `codex/k3-decode-verify-opt-checkpoint-20261007` 的 `9b85c458a1f7aea064a7720b13774abec0d30758`；保留原分支，不在其上续写 |

原始 BF16 发布提交 `99471956b15db7de3dae1d693e4332e2bb08f9d3` 仍可在本仓库读取，但它**不是**当前 HEAD 的 Git 祖先：K3 开发历史经过重基和功能提交整理。当前 BF16 功能锚点是 `e140b0a3d0`。不要仅凭 ancestry 判定 BF16 内容丢失，也不要把原始发布提交写成当前分支的直接父提交。

### 提交边界

| 提交 | 内容与状态 |
| --- | --- |
| `e140b0a3d0` | 已验证 BF16 K3、双机 PD、Native MTP 基础 |
| `e3b291be22` | 通用 FP8 modeling 与筛选后的 Prefill 算子；attention 投影及 MLA operand/cache 为 E4M3，MoE 保持原生 MXFP4 |
| `caa7f9ce5d` | 历史 MLA KV 分片与大激活生命周期优化；不含整模型 Chunk Forward |
| `9efba1d296` | MLA PageRR KV 放置与 Decode DCP |
| `68592e2b17` | 默认精简完整 FP8 PD smoke；四层快检与 93 层正式模式由 checkpoint 层数选择 |
| `17366759e1` | 非对称 PD：MLA 页按 Prefill owner 交接，KDA recurrent/conv 状态按 attention TP 重分片，Decode DP owner 交接 |
| `f2f4873228` | 四层／全层 smoke 与 DP owner 测试整理 |
| `fc8714604d` | Decode Verify 性能改动合为一个提交：FP8 TP 融合投影、Decode KDA/MoE、CUDA Graph 旧 block-table 行处理；旧版散落的候选提交不再逐个作为交付锚点 |
| `44e892a8cb` | Decode Graph 真实 batch 63/64 边界测试；默认仍是 368 条正式 case |

写入本文件前，`44e892a8cb` 的代码与测试树 OID 是 `6346602bf04494c86224afc1089e6325ef9d928f`，与旧检查点 `9b85c458` 相同。因而下面引用在旧检查点上保存的运行记录，对应同一份最终代码与测试；本文件的文档提交会使整个 Git tree OID 改变。性能 JSON 中出现的 `576cf612` 是整理前的产品代码检查点；它早于部分 smoke 整理，并不是当前分支 HEAD。**下一会话若再改产品代码，必须重新构建并跑完整默认双机 smoke。**

## 当前实现及边界

主模型在 [kimi_k3.py](../../../../rtp_llm/models_py/model_desc/kimi_k3.py) 组装 KDA/MLA、Dense/Latent MoE 和 AttnRes；[KDA 实现](../../../../rtp_llm/models_py/model_desc/kimi_linear.py) 区分 Prefill chunk core 与 Decode recurrent core。[FP8 融合投影](../../../../rtp_llm/models_py/distributed/fp8_collective_projection.py) 复用通用类：TP 内 AG 携带 E4M3 值和 scale，QKV/AttnRes GEMM 与其融合；输出 GEMM 与 BF16 RS 融合。用户已允许这条 AG 使用 FP8 通信。Native MTP 的 hidden、权重与通信仍为 BF16，`GEN_NUM_PER_CIRCLE=3`；Decode 的 target Verify 和 draft 都要在运行证据中看到。

Prefill Query CP 固定为 **1**。只有 **MLA KV 页**采用 PageRR；KDA 保持常规 attention TP 状态切分。[MLA Prefill](../../../../rtp_llm/models_py/modules/kimi_k3/mla_prefill.py) 使用 TokenSpeed，并对历史 prefix 做 KV-up 分片；[分片规划器](../../../../rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/mla_prefix_chunk_plan.py) 默认每 rank **6 GiB 展开后 K/V**，当前轮 Q 不在这个历史预算内。[MLA Decode](../../../../rtp_llm/models_py/modules/kimi_k3/mla_verify.py) 用 PageRR/DCP 与 CUDA Graph。PD 的 C++ [传输规划器](../../../../rtp_llm/cpp/cache/KVCacheTransferPlanner.h) 分开处理页 owner 和 KDA head shard；`17366759e1` 另对格式、scale 和资源所有权做校验。只验证过本地每端不超过 8 卡的拓扑；16 卡目标仅做静态／单测检查，没有设备验收。

峰值优化已包含 skip-head-mid、历史 prefix 分片、部分 BF16 暂存提前释放、shared expert 单份 gate/up 输出和按单轮容量配置 workspace。**没有**引入整模型 Chunk Forward；`feat/k3_dev` 的静态 AttnRes bank、独立 MLA scratch、MTP hidden 释放接口也没有原样搬入，不能把它们写成已迁移项。旧峰值对照及其边界见 [PageRR/峰值记录](20261003_FP8_PAGERR_PEAK_HANDOFF.md)。四层 2M 历史 KV＋约 64K Q 的 PyTorch allocated 峰值为集成/feat **23.531/24.674 GiB**；93 层为 **203.290/205.350 GiB**。同文档的快照事件重建还给出本次新分配且峰值时存活 **13.089/13.385 GiB**。这些都是 PyTorch allocator 口径，不等于 NVML 整机占用；feat 全层时间只有一条热态样本，不能作为性能分布。

用户限定后续只关注 LLM modeling，不迁多模态、整模型 Chunk Prefill、KTP 或未获允许的大特性；尽量在现有 main 架构里收敛实现。Decode 不应为节省 Prefill 峰值而额外走 shared expert 先拆后合的路径。主模型若继续优化，优先在初始化时选择算子实例，让层 `forward` 保持统一调用。

## 已验收的正确性

正式 smoke 由 [launch_bf16.py](../launch_bf16.py) 的 `--orthogonal-smoke --fp8-gemm --fp8-kv-cache` 等参数拉起服务，再运行 [text_smoke.py](../text_smoke.py)。**runner 没有选择性关闭 case 的 suite 开关**：checkpoint 是四层时固定走快检，是 93 层时固定走精简完整模式，并始终选择 cache、cancel、page、chunk、decode 五组正交阶段。`--orthogonal-smoke` 是服务启动配置，不是关闭测试组的选项。正式每条请求上限 300 秒。

93 层最终默认配置：Prefill TP8/EP8、Query CP1、Device＋Memory Cache、reuse cache 开；Decode DP2/TP4/EP8、DCP、CUDA Graph、Native MTP、Memory Cache 关、reuse cache 关；MLA PageRR、KDA TP。每端 8 卡，Decode 全局并发目标 64，两个 DP owner 各最多 32。Prefill context batch/并发 64、总 batch token 容量 262144，未命中 Q 预算 65536；历史 Chunk KV 用例通过逐轮 seed 准备约 2M prefix。服务的 `launch.json` 是最终生效参数的权威记录，不能只看预期命令。

最终 93 层默认 smoke 在这一代码树上跑了 **368/368 条正式请求**，独立审计 `answer_passed=true`、`path_passed=true`，耗时 **376.87 秒**（约 6 分 17 秒）。原末尾 3 条长输出 repeat 及 3 条历史超过五分钟的 case **没有发送，合计 6 条跳过**；`full_original_suite_passed=false`。逐 case 结果在 115 的 `/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/default-smoke-dp2-final576-v3-20261008/result.json`，独立审计在同目录 `audit/audit.json`；交付摘要为 `/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/deliverables-20261008/default-dp2-368-smoke-audit.json`。独立审计程序为 [audit_orthogonal_smoke.py](../audit_orthogonal_smoke.py)。

审计的 `path_results` 逐项为真：Device/Memory/partial/miss 混合进入同一次 Prefill forward、Host 加载期间取消及恢复、4096/32768 token 页边界、实际多片历史 Chunk KV、PD/DP owner 交接，以及 Decode Graph/DCP 的真实 batch 1、4、8、63、64。原始 JSON 中另有各 case 的答案、MTP draft、cache reuse 和分页证据；不能只拿 case 名称或 HTTP 200 代替路径证明。四层只用来先排除实现问题，**不能**替代上述 93 层验收。已有等 TP8→TP8 回归及 TP4→TP8 四层反向分片记录；最终正式拓扑仍是 TP8→DP2/TP4。

## 性能测量的最终口径

固定条件是 FP8 target、原生 MXFP4 MoE、BF16 MTP `n_step=3`、Prefill TP8/EP8、Decode **DP1/TP8/EP8**、64K 历史 KV、真实 Decode batch 32、CUDA Graph bucket 32，Verify 行数 128。对照仅计算 `target_model_verify` 范围及其 GPU 工作，按每步最大 rank 关键路径；draft forward、input prepare、draft update、TPOT 均不计。SSM replay 不参与模块择优，但原始 Verify 总时间仍包括它。四层先筛候选，再用 93 层复核。每版预热一组 32 请求以产生多次相同形状 replay，采三段独立窗口、每段 8 rank，最终对齐绝对 Verify ordinal **80–111**。完整请求输入 token ID 的 SHA256 为 `c64fed40ddda484d245ffa9aa085322f51433cd9e86461387d31afcf52159e56`。

| 指标 | 集成版 | 固定 feat | 解读 |
| --- | ---: | ---: | --- |
| 四层 Verify 最大 rank GPU span | 2380.318 µs | 2661.217 µs | 集成快约 10.6%；四层输出仅用于流程／性能，不作语言精度金标 |
| 93 层 Verify 最大 rank GPU span | 41762.526 µs | 40763.870 µs | 集成**慢 2.45%**，在本轮约定的 5% 持平阈值内；不能写成更快 |
| 93 层 KDA conv kernel 累计 | 298.621 µs | 436.563 µs | 集成快约 31.6% |
| 93 层 MLA TokenSpeed kernel 累计 | 3415.227 µs | 3436.586 µs | 接近持平 |
| 93 层完整 MoE | 22524.323 µs | 22218.537 µs | 集成慢 1.38%，仍在阈值内 |
| 93 层专家 kernel 累计 | 18257.256 µs | 17336.473 µs | 集成慢 5.31%；两边调用相同 DeepGEMM 实现，尚无可直接挑选的独立 feat 实现 |

以上模块累计时间可重叠，**不能相加得到 Verify 总时长**。Embedding AllGather 单独核很长，但包含 rank 等待，不能据 kernel 名称推断算子固有速度。四层与全层逐模块选择理由在 115 的 `/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/deliverables-20261008/module-selection-table-20261008.json`；三方窗口原始数值在同目录 `absolute93-verify80to111-threeway-20261008.json`。旧的 `head370`/较早 `576` 64K B32 诊断一度显示集成慢约 **12%**；其窗口及代码口径与最终绝对 ordinal 对齐记录不同，保留作历史诊断，以最新的 **2.45%** 作为本轮结论。

曾仅对任务运行环境尝试 `FT_DISABLE_CUSTOM_AR=0` 的 symmetric-memory 候选。四层 Verify 好一些，但四层 Embedding AG 变慢；93 层完整 Verify 为 **42569.072 µs**，比正式配置 `FT_DISABLE_CUSTOM_AR=1` 的 **41762.526 µs** 慢约 1.93%。因此没有接入产品配置。93 层配对的 **128 条**集成/feat 请求输入 ID、输出 ID、文本完全一致；这是两版相互一致的证据，不代替独立的参考模型 golden。对应 JSON 是 `absolute93-longout-paired-response-audit.json` 与 `integration93-symm93r1-vs-disabled-response-audit.json`。

额外已采的 93 层热态对照：64K Prefill 集成/feat target 最大 rank 中位 **1521.216/1498.666 ms**，集成慢 **1.505%**；8K KV、batch 64、`n_step=3` 的 Decode Verify 为 **47.615/49.801 ms**，集成快 **4.39%**。它们是另外两种负载，不能与 64K KV、batch 32 的 Verify 数字混写。三窗口和各 24 份全 rank trace 路径见 `final576-vs-feat-edaaf-timeline-comparison-20261008.json`。10 月 1 日的 [Prefill 三方算子择优记录](20261001_FP8_PREFILL_FINAL_COMPARISON.md) 包含 vLLM 对照；固定 vLLM 的投影是 BF16，且全层请求拆成约两个 32K chunk，不能声称它与当前 RTP FP8 投影在完全相同精度、分块下逐算子排名。

## 原始材料在哪里

115 的交付索引：`/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/deliverables-20261008/manifest.json`。同目录有逐算子选择表、三版相关对照摘要、正式 smoke 审计和预热样本。正式 smoke 的逐请求 JSON 及审计在上节所述目录。性能复现用的临时启动脚本、host selector 输出也保存在 115 的 `/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/`；先读脚本和其中的 `launch.json`，不要盲跑旧命令或占用原端口。

Decode 64K/B32 的**原始** trace 不在这个交付摘要目录，而在 **113**：

- 集成 93 层：`/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/integration-93-decode-absolute93r1/traces/`，3 窗口 × 8 rank，共 24 份。
- feat 93 层：`/data0/luohaocheng.lhc/artifacts/k3-decode-target-verify-20261005/feat-113/feat-edaaf-decode-verify-93-absolute93r1/decode/runtime/work/decode/`，24 份。
- 四层两版：见 `manifest.json` 的 `raw_traces` 段，均在 113，各 24 份。
- 64K Prefill 原始 trace 在 **112**，8K/B64 Decode 原始 trace 在 **113**；准确 glob、文件数、字节数见 `final576-vs-feat-edaaf-timeline-comparison-20261008.json` 的 `raw_trace_manifest`。

既有更早证据：[BF16 原始交接](../HANDOFF_BF16_20260925.md)、[118 条问答](QUESTIONS_ANSWERS_118.md)、[FP8 性能锚点](20261001_FP8_PREFILL_PERFORMANCE_ANCHOR.md)、[峰值锚点](20261001_FP8_PEAK_ACTIVATION_ANCHOR.md)、[PageRR/2M KV 记录](20261003_FP8_PAGERR_PEAK_HANDOFF.md)。阅读时按各文件标注的提交和服务版本解释，不把旧数字覆盖当前结论。

## 机器、权重与构建约束

110–115 的个人开发根目录统一优先 `/data0/luohaocheng.lhc`；112 的 HOME 另在该目录下 `.home`。这份交接不保证接手时哪两台机器空闲。2026-10-08 的性能预检曾选同集群 112/113，原始选择结果在 artifact `host-selection-integration-prelaunch.json`；该记录会过时。新实验应重新核对登录身份、GPU 显存/进程、磁盘/inode、容器、端口、普通网络和 RDMA，两端各最多 8 卡；性能和 timeline 必须独占，测试后清理自己的服务。勿把显存占用且 GPU 利用率为 0% 自动视为可杀进程：先查属主及其任务。

本机 115 当前可见的 3FS 根是 `/mnt/hf3fs/3fs/models/kimi`：target `kimi-k3` 的 config/index SHA256 为 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` / `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`，index 指向 **96** 个 shard；四层 `kimi-k3-4layers` 为 `8754ec8b41ae59791aa6370386276467bde6e8ce8502915dcfb9769ba9990da8` / `eb064b56fc8a782b97594d9888314a430450128d1fe4a7da66662f6d641122b5`，**7** 个 shard；MTP `kimi-k3-mtp` 为 `6e5457c4113f59f28f9fd7323c89240fa14ce6f0108f43427579595417d2d45b` / `12a705576dd2e39726d3d84f0652ffb6ce7874fb1d75591f4f693080ed1aeb8d`，**9** 个 shard。这是**本次元数据只读检查**，不等于此刻对全部 shard 重算哈希；正式启动前仍要验证所选两端文件、版本与完整性，不能凭目录名认定相同。

用户明确希望后续从 3FS 直接由 FastSafetensors 加载；[launcher](../launch_bf16.py) 的 `--allow-hf3fs-root` 与 loader guard 是现成的受控通道，`LOAD_METHOD=fastsafetensors`，不要跳过 guard 或放宽断言。若 3FS 明显变慢，先量读取，再仅对实际读权重的机器调任务级并发并记录原值、改后值、吞吐和回滚；不要改 110–115 全部机器或集群全局。源码、Bazel 输出、JIT 缓存和日志放本机个人 data0；在各自 `lhc_GPU` 内以 `luohaocheng.lhc` 构建，使用 `--config=cuda13 --config=sm10x`。不要写 `/3fs-data/3fs/mtp_test`，也不要把账号密码、OSS 凭证或 token 写进仓库和交接文件。

## 还需要做什么

1. **先固定接手基线。** Fetch 集成分支，核对远端 HEAD、Git 状态及本文件；按 `manifest.json` 找到产物。若目标是提交 main，先审查当前 main 的差异及 ABI，不能把已验收的旧基底误称为最新 main 已验收。
2. **需要开发时从四层开始。** 用实际生效的 FP8/PD/MTP/Graph 参数验证数值和路径，再运行 93 层默认 368 条完整 smoke，并独立复核答案和事件。六条约定跳过仍单独报告。不要添加 suite 逃生开关，也不要加入未约定的功能。
3. **性能仍有可研究点。** 64K KV/B32 全层 Verify 当前在 5% 持平阈值内，但慢 2.45%，专家 kernel 累计慢 5.31%；64K Prefill 慢 1.505%。若继续择优，应先在四层同形状预热、全 rank 归因，找出真正可替换且数值合格的实现；不要因旧 12% 诊断、单一 kernel 名称或跨服务不同 scope 直接迁代码。任何采纳候选后重做全层时间线和正式 smoke。
4. **上线前仍有范围外验证。** 16 卡拓扑没有设备验收；旧完整 suite 的六条被跳过；当前相互一致的 128 条长输出不是跨框架 golden。这些都不能写成已完成。当前只在 GB300/SM103 路径开发，不新增跨机通信协议或多模态代码。

此文档本身只交付代码与证据位置，没有创建、迁移或注册新的 Codex 会话。下一会话按自己的目标选择工作树和机器；不要复用旧进程或直接覆盖这个工作树。
