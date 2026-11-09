# GLM53 Flash decode 优化收敛记录

验收配置：L20C 8 卡，CUDA Runtime CC 10.0 / 148 SM，D 端 TP1/DP8/EP8，
总 batch384（每卡48请求），每个请求输入81,920 token，真实 MTP4 接受，FP32 SSM state。
P 端 TP8/CP8/EP8。未强制接受投机 token；性能窗口不包含 profiler 和 Prefill 加载阶段。

最终代码合入基线 `c20e6f259143506091354113f8d4df14b5a2d1de`，保留其新增的cache正确性修复。
以下性能来自此前 `e86096bf33635d92dda7aaa8be7b61a86de93e19` 上的配置，两次 TPOT 为
**9.5335 / 9.5995 ms**，合并有效窗口 **9.5672 ms**，平均每卡 **301,029 TPM**。
原版 MTP3 为14.2605 ms，吞吐201,956 TPM/卡；组合延迟降低32.9%，吞吐提升49.1%。
按用户接受的“9.53 ms 配置”收敛，不再做吞吐搜索。以下收益均来自已完成的测量，
并非本次补测重新测得。收敛后的默认值、适用性检查和测试修复未再做整机吞吐测量。

## 默认值和开关

下表中的算子开关默认启用，显式设为 `0` 关闭、`1` 启用，**修改后需重启服务并重新捕获 CUDA Graph**。
Router 的分解项数使用 `0/2/3`，默认 `2`，`0` 恢复整批 FP32 linear。
所有回退继续保留原版整批 Router；不恢复逐 token GEMV。
权重分解和MoE预变换用于冻结权重的推理；更新权重后需重建相关模块和CUDA Graph。

“语义无损”指不近似模型算术；“规约/舍入敏感”指有限精度计算路径可能不同，
不等同于已经发生评测集精度下降。当前没有评测集分数，数值对比和输出前缀不能替代评测。

| 优化 | 默认值 / 关闭方式 | 泛化性和回退条件 | 精度分类 | 已有大致收益 |
|---|---|---|---|---|
| 整批 FP32 Router | 原版已包含；下面分解开关置0即可使用 | Decode/MTP整批，Prefill保持固定256行投影 | FP32原路径；不同batch可能改变规约顺序 | 已在origin，不重复计入本轮收益 |
| MTP异步 prepare | `RTP_LLM_MTP_ASYNC_PREPARE=1`；`0`关闭 | CUDA MTP的通用调度；保留prepare事件依赖、CP上下文回退和异常收尾 | 语义无损 | 新版安全优化组合合计约0.75 ms，未隔离本项 |
| target token gather / bookkeeping批量拥有副本 | `RTP_LLM_BATCH_MTP_BOOKKEEPING_COPY=1`；`0`恢复逐stream处理 | 不固定batch/MTP长度；tensor view持有同一底层副本，接受长度仍逐请求处理 | 语义无损 | 历史bookkeeping约0.19 ms；单独target gather没有可靠正收益 |
| Greedy接受/拒绝融合 | `RTP_LLM_GREEDY_REJECTION_FUSION=1`；`0`关闭 | CUDA、greedy、连续int32、提议数小于32；强制接受、显式generator和非greedy回退 | 语义无损，整数比较，保留真实接受长度及CPU回传事件 | 历史约0.12 ms；新版计入0.75 ms组合 |
| sampler历史工作区压缩 | `RTP_LLM_MTP_COMPACT_SAMPLER_HISTORY=1`；`0`关闭 | 仅greedy、单beam、无历史惩罚、无logits processor、无logprobs；其余完整历史回退 | 语义无损，**不截断模型上下文或逻辑step** | 历史约0.08 ms；新版计入0.75 ms组合 |
| KDA gate前移 / T4 stage配置 | `GLM53_KDA_HOIST_GATE=1`；`0`关闭；`GLM53_KDA_VERIFY_CONFIG=16,1,3`，`0`取消显式配置 | N48、H/HV64、K/V128、FP32状态等条件；T4保持原BV16/warp1，仅stage改变；gate前移适用于已覆盖的verify长度 | 保持原规约几何时语义无损 | 历史gate前移约0.14 ms；新版计入0.75 ms组合 |
| KDA多token低warp verify | `GLM53_KDA_FAST_VERIFY=1`；`0`关闭 | **SM100、每卡N48、T2～6（MTP1～5）、H/HV64、K/V128、FP32 K-major状态、非varlen、有block_map、low_warps已开**；其他形状保留原launch。T=提议数+1，**MTP4命中T5** | BV32/warp4到BV16/warp1改变规约，FP32状态保持 | 新版同MTP4：15.839→13.857 ms，约**1.98 ms**（包含T5 gate前移及launch调整） |
| 跳过重复TopK填充 | `GLM53_INDEXER_SKIP_TOPK_FILL=1`；`0`关闭 | 仅保证全部写出（含短行的-1 padding）的native TopK；torch/空宽度保留初始化 | 语义无损 | 历史约0.08 ms；新版计入0.75 ms组合 |
| Indexer Hadamard/Q量化融合 | `GLM53_INDEXER_FUSED_Q_QUANT=1`；`0`关闭 | KPool + rotate_q；保留FP32 gate及UE8M0 scale | 规约/舍入敏感，未降低gate dtype | 与下项及其他内核组合内，未隔离单项 |
| Indexer FP32 gate + Q量化融合 | `GLM53_INDEXER_GATE_QUANT_FUSION=1`；`0`关闭，上一开关置0也会停用 | Decode，H4096、head_dim128、heads32/64，连续BF16输入/FP32权重，无weights分块；不满足条件走原FP32投影和量化 | 规约/舍入敏感；8组独立检查Q-FP8逐位一致，folded gate相对L2最大约4.72e-7 | 见4.11 ms组合，不单独归因 |
| Router双BF16分解 | `GLM53_ROUTER_BF16_PARTS=2`；`0`回FP32；`3`为未做性能验收的三项分解 | CUDA SM100/103、连续BF16输入、H4096、E256/288、1～384行、非Prefill metadata；其余FP32 | **FP32权重近似**，相对L2约2.5e-6；near-tie路由可能变化 | 新版同MTP4的Router三项组合约**1.05 ms** |
| Router split-K | `GLM53_ROUTER_SPLIT_K=1`；`0`关闭 | 双BF16、E288、M48/192/240/288；其余使用普通router投影 | 规约/舍入敏感 | 包含在上述1.05 ms中，不重复相加 |
| Router reduce/TopK融合 | `GLM53_ROUTER_REDUCE_TOPK=1`；`0`关闭 | 上述split-K条件，单group、选1组、top8，TF32全局关闭；不支持的输出布局/配置走GroupTopK | 保持给定split-K logits的native tie/FTZ规则，整个Router仍含近似 | 包含在上述1.05 ms中 |
| MegaMHC post/pre/RMSNorm | `GLM53_MEGA_MHC=1`；`0`回普通HC；`GLM53_MEGA_MHC_SMS=96` | **仅SM100、H4096、HC4、M48/192/240/288**；验证BF16 residual/norm、FP32 Fn/post/comb等ABI；外部扩展/完整JIT头文件缺失或不兼容时回退 | **Fn显式使用TF32 Tensor Core / TFLOAT32载入**，并非仅FP32规约变化；不受全局allow_tf32=False约束 | 见4.11 ms组合，没有新版单项消融 |
| mHC到MoE输入/scale packing融合 | `GLM53_MHC_PACK_FUSION=1`；`0`关闭 | 依赖MegaMHC和FP8 fused shared expert，routed TP1/EP>1、上述M/H；使用当前graph clone自身buffer，owner/shape不符报错 | 量化/舍入位置敏感 | 历史单开约0.05 ms；新版包含在4.11 ms组合 |
| 原生paged sparse MLA直接读KV | `GLM53_NATIVE_SPARSE=1`；`0`回selected-KV gather；`GLM53_NATIVE_SPARSE_BATCH_INVARIANT=0` | **仅SM100**、64 heads、512维NoPE Q、block128/528B FP8 KV+FP32 scale、2176个物理索引、总行数≤384；依赖私有扩展。verify长度来自真实plan metadata，普通B80不会被推断成16×5；不满足条件回退 | attention规约/舍入敏感；batch invariant路径未做性能验收 | 历史约0.37 ms；新版包含在4.11 ms组合 |
| DeepGEMM PDL | `GLM53_DG_PDL=1`；`0`关闭 | SE wrapper在buffer/warmup前设置；MegaMHC同开关；旧库无set_pdl时保持库原行为 | 语义无损的调度 | 组合内，未隔离；不把环境变量存在本身当作已生效证据 |
| Decode整数TopK整理融合 | `GLM53_DECODE_FUSED_TOPK_CANONICALIZE=1`；`0`回torch整理 | 仅在下面canonicalize已开启时生效；int32及已支持宽度/行数，其余回退；不修改Prefill开关 | 语义无损，15组eager/graph整数逐位一致 | 旧torch多算子→融合：9.752→9.580 ms，约**0.17 ms** |

KDA的形状dispatch覆盖T1～7、batch1/47/48/49/64/96和SM89/100/103，
独立FP32递推及完整状态池对比覆盖N48/T2～6、边界81921/82047、开关两种状态及动态Graph回放。
SM103测试仅模拟SM100的Python调度选择，实际编译仍用CUDA Runtime识别到的SM103；
整机性能来自真实L20C/SM100。本轮验收的是MTP4，不保证换batch、卡型、MTP长度后仍有同样收益。

## 11 月 9 日补充：KDA verify 的 T=2/6/8 快路径

`GLM53_KDA_WIDE_VERIFY` 默认是 `1`；设为 `0` 后重启 Decode，只回退本次新增形状，保留已有 MTP4 和 B48 快路径。`GLM53_KDA_FAST_VERIFY=0` 仍关闭多 token verify 优化的总分派。

新增路径限于 SM100、每卡 8–64 个请求、固定 T=2/6/8（MTP1/5/7）、H/HV=64、K/V=128、128-token page、FP32 K-first paged state、原地写回。使用可用的 block map 和 sequence lengths；不接管 ragged verify、无状态写回的 replay 或未验证的设备和布局。调度选择 BV16、单 warp、3 stages，保留现有显式 launch 配置。

状态页、checkpoint 位置和写回语义保持不变；规约顺序会改变有限精度舍入。L20C 的 15 个算子对照中，13 组快约 1.33–2.03 倍，原有 B48×T2/T6 持平。端到端收益须按相同 batch、上下文和实际 MTP 接受数比较。不要从小上下文外推到容量不够的长上下文：MTP7 的 FP32 SSM checkpoint 池比 MTP4 大，本次调度改动不改变池的分配。

## 需要随部署匹配的参数

这些参数保持显式配置，不把全局服务默认改成只适用于当前8卡集群的值。

| 配置 | 本轮值 / 关闭或恢复方式 | 精度和收益 |
|---|---|---|
| CPU线程预算 | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；可改回服务原值 | 不近似模型；历史约0.08 ms，新版计入0.75 ms组合 |
| 投机长度 | **P/D均 `GEN_NUM_PER_CIRCLE=4`**；可改3 | 精确投机算法保持target分布，verify batch变化可能改变舍入。新版只把MTP3改4时13.514→15.839 ms，**慢约2.32 ms**，因此必须配套T5 KDA调度，不能单独宣称MTP4收益 |
| KV block / paged Indexer | **P/D均 `SEQ_SIZE_PER_BLOCK=128 KERNEL_SEQ_SIZE_PER_BLOCK=128`**；可同改64 | 布局语义相同；KPool压缩后page32启用另一score kernel，规约敏感；包含在4.11 ms组合 |
| Routed/shared expert融合 | `MOE_STRATEGY=mega_moe_fp8_se`；可选回 `mega_moe_fp8` | 权重dtype同为FP8不代表无损；若输入是原始FP32 scale，shared权重转换为UE8M0可能重量化，且combine顺序改变；包含在4.11 ms组合 |
| 索引排列 | **D `DSV4_INDEXER_TOPK_CANONICALIZE=0`，P保持1**；D设1恢复排序 | 仍用 `GLM53_DECODE_INDEXER_TOPK_BACKEND=topk_v3_tie_break` 保持稳定成员；仅改变索引排列/attention规约，非逐位无损。与9.580 ms保序方案的差异在重复波动内，无可靠单项收益，可独立恢复 |

相对13.857 ms的fast-KDA MTP4配置，block128、Router、MegaMHC、native sparse、shared expert、
Indexer融合及PDL一起到9.752 ms，组合约**4.11 ms**；后续整数整理到9.580 ms。
Router的1.05 ms是这个组合内的独立对照，不能再次叠加。历史估计来自10月22/23日旧基线，
仅用于量级参考，不能从当前9.53 ms直接减去，也不能把表内各行相加预测新配置。

## 依赖、回退和精度对照

MegaMHC完整扩展和JIT头文件可安装在 `rtp_llm/libs/mega_mhc_isolated/`，或设置
`GLM53_MEGA_MHC_LIBRARY_DIR`。私有FlashMLA扩展可安装在
`rtp_llm/libs/flash_mla_glm53_22.abi3.so`，或设置 `GLM53_NATIVE_SPARSE_LIBRARY`。
代码默认允许使用，只有依赖和ABI/形状均满足才执行；普通安装缺依赖时日志会标明回退。
受支持路径上的真实kernel执行错误继续抛出，不静默掩盖。

逐项排查精度时可独立关闭上表开关。要同时回到新增优化之前的数值路径，可先设：

```bash
export GLM53_ROUTER_BF16_PARTS=0
export GLM53_ROUTER_SPLIT_K=0
export GLM53_ROUTER_REDUCE_TOPK=0
export GLM53_KDA_FAST_VERIFY=0
export GLM53_KDA_HOIST_GATE=0
export GLM53_KDA_VERIFY_CONFIG=0
export GLM53_MEGA_MHC=0
export GLM53_MHC_PACK_FUSION=0
export GLM53_NATIVE_SPARSE=0
export GLM53_INDEXER_GATE_QUANT_FUSION=0
export GLM53_INDEXER_FUSED_Q_QUANT=0
export DSV4_INDEXER_TOPK_CANONICALIZE=1
export MOE_STRATEGY=mega_moe_fp8
```

如要与原版MTP3/block64逐位对照，还需**同时在P/D改回提议长度和block大小**、重建KV/SSM缓存并重启；
保留FP32 SSM state、当前FP32 indexer head gate和UE8M0 Q scale。上述环境变量不是评测集验证的替代品。

## 补测与review范围

- 长上下文MTP3/MTP4 compact sampler，默认/显式0/显式1，以及历史惩罚回退。
- Greedy CUDA融合：提议数0/1/3/4/5/31、target stride1/3，覆盖每个拒绝位置及全接受。
- KDA：MTP4对应T5，T2～6的独立FP32递推、全部状态池、页边界、动态Graph；其他形状/架构/状态dtype和layout的回退。
- Router：默认BF16近似与显式FP32、非连续输入回退、Prefill固定256行精度策略、M48/192/240/288的融合路由与native GroupTopK比较（含全tie）。
- Indexer：FP32 head gate边界用例及Graph、UE8M0 Q量化，跳过fill时短行/空宽度/-1 padding全写保证。
- Native sparse：缺依赖/关闭/其他架构回退；真实metadata驱动的decode B80、T4/T5/T6与scheduler缓存复用。
- MegaMHC：缺依赖回退、形状/dtype/HC几何、禁用开关检查；原生kernel整机覆盖来自此前验收测量。
- Code review修正：原生sparse的行数猜测、默认开启后的可选依赖处理、Router连续性/输出ABI检查、Indexer head_dim/权重连续性、MTP批量gather的关闭路径。移除未参与验收的early-PD-cache-release实验。异步C++夹具使用与真实CUDA模型一致的输出/初始状态，并在断言前等待Prefill/Decode bookkeeping完成，按CUDA PDFUSION的约定检查GPU proposal镜像（CPU proposal的-1是省去D2H的标记），保留原期望token及概率/隐藏态数值断言。

不把已有 `DEVICE_INPUT / STREAM_ASYNC / DROP_BROAD_SYNC / indexer share` 再计为本轮新增收益。
不包含无收益的read-only views实验或没有调用点的async-device-state helper。
`REUSE_CACHE / RTP_LLM_ASYNC_PREPARE_CACHE` 不计入本轮decode优化收益；验收测量均关闭。

验收后的补测结果：Python/GPU首轮319项通过，追加复查46项通过；C++57个用例覆盖默认、显式关闭和异步配置，另验证异步时关闭批量gather/bookkeeping的回退。所有原期望token、概率及隐藏态数值断言保留。
