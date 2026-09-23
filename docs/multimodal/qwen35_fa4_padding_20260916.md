# Qwen3.5 ViT：Q/K 72→80 布局优化

2026-09-16。仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource；分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。仅 GPU0，NVIDIA L20D / SM103，148 SM，驱动 580.105.08，Torch 2.11.0+cu130。保留此前工作区改动，未提交或推送。

## 结果与范围

已将 Q/K 的每个 head 从 72 个 BF16 元素补零至 80，并入原 RoPE 的 FP32→BF16 类型转换。V 与 attention 输出保留 72 维；softmax_scale 保持 72**-0.5。真实 397B ViT 的 27 层、merger 和 token assembly 后，batch=1/32 的全部 embedding 与位置编码均与修改前逐位一致。

下表是完整多模态子系统的实测结果，起点 MMProcessEngine.mm_embedding_rpc 进入，终点 embedding/hash 在 GPU 完成；包含视频加载、NVDEC、GPU resize/预处理、ViT、merger、token assembly。排除 gRPC 序列化、网络/RDMA、LLM prefill/decode。batch 是一次打包的视频数，不是 HTTP 并发数。

基线为本轮修改前已有的 dense FA4 72 维路径；不是最初 varlen。每档双方预热至连续两次变化小于 5%，再交替 AB/BA 四轮。计时不含 profiler、模型加载/JIT/预热、正确性检查。GPU0 各样本前后均无其他计算进程；未锁频。

| 视频 batch | 基线中位 ms | Q/K80 中位 ms | 耗时降低 | 基线平均 ms | Q/K80 平均 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 719.702 | 698.182 | 2.99% | 719.939 | 699.433 |
| 32 | 22235.070 | 21885.135 | 1.57% | 22234.210 | 21875.605 |

每档只有四次正式样本，不报告 P99；不能将本次百分比和此前另一轮 varlen→dense 的百分比直接相加。

原始 ms：
- batch=1，baseline：716.008, 724.345, 717.736, 721.669。
- batch=1，production：697.199, 697.497, 704.170, 698.867。
- batch=32，baseline：22195.275, 22274.864, 22281.442, 22185.257。
- batch=32，production：21867.825, 21762.053, 21970.097, 21902.445。

峰值 torch allocated（请求及随后完整正确性检查的共同区间，不是纯请求区间；不含 allocator reserved）：

| 视频 batch | 基线 GiB | Q/K80 GiB |
|---|---:|---:|
| 1 | 8.433 | 8.172 |
| 32 | 238.038 | 229.676 |

## 为什么选择只补 Q/K

此前诊断显示 72 维布局的片上访存请求偏多，同时 FA4 存在 K/V 供数、softmax/correction/MMA 的阶段依赖。BF16 head 的物理间隔由 144 B 变为 160 B，改善 32 B 对齐条件。该修改改善布局供数效率；不代表移除了全部阶段同步或达到了 Tensor Core 满载。

同一模拟输入、同一进程的单层 attention 对照，6 轮正反交替、每轮 20 次，排除 padding 和分配：

| 方案 | 中位 ms |
|---|---:|
| native72 | 11.35374 |
| qk80 | 10.41703 |
| v80 | 10.40240 |
| padded80 | 10.03279 |

只补 Q/K 的纯 attention 耗时降低 8.25%。模拟输入全量输出逐位一致。这不是完整模型速度，且不应跨实验绝对时间比较。

候选初筛再使用真实视频和真实权重跑完整多模态子系统，batch=1 四轮中位：dense72 724.167 ms，只补 Q/K 701.069 ms，全补 Q/K/V 707.407 ms。只补 Q/K 的完整路径更快，避免了额外 V 搬运和输出裁剪；因此全补 Q/K/V 仅保留在实验脚本中。

## 实现边界

- 仅 SM103、BF16、16 heads、原 head_dim=72、等长段且长度至少 1024、FA4 无梯度推理启用补零；其余原有路径保持回退。
- 保留原 FP32 RoPE 中间结果及舍入顺序，不把 FP32 求和与 BF16 写出跨越原舍入边界融合。转换 kernel 写入有效 72 维并同时将尾部 8 维置零。
- Q 转换后立即释放 FP32 Q，再转换 K。转换索引使用 int64，覆盖 batch=32 的大张量。
- FA4 仍调用已安装的原生公开 dense 接口及默认 128×128 tile；Q/K/V 分别按各自维度 view，每个视频段独立。V 与输出仍为 72，scale 不改。
- 无新增用户参数，无依赖 wheel 或 FA4 源码修改。

[attention 接入](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:447)，[cast+padding kernel](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/vision_kernels.py:113)，[仅本轮改动 patch](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/this-turn.patch)。

## 正确性和输入

12 个针对性测试通过，0 failure/error/skip，包括 BF16 halfway 舍入、零尾部、非连续 QKV、空输入、autograd 回退、混合段长/短段/FP16/其他 head_dim 回退、跨段隔离、CUDA Graph 重放后更新输入。使用已有构建运行时执行 Python 测试，本轮没有重新执行 Bazel。

真实权重完整多模态子系统 batch1/32 的双方预热检查及全部正式样本通过：embedding 每视频 [57868,4096] BF16 CUDA、position [57868,3]、有限值、所有元素与同 batch 修改前结果逐位相等，max_abs=0。真实权重基准关闭 CUDA Graph；Graph 在单层测试覆盖。

路径计数：{"baseline": 513, "production": 405}；实测 Q/K/V 维度：{"baseline": [[72, 72, 72]], "production": [[80, 80, 72]]}。确认测试执行到新路径。

- 权重：/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8，ViT BF16。
- 视频：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4。
- fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180；每视频 23 段，每段 10032 tokens。
- vit_concurrency=64，vit_max_queue_size=64，gpu_max_batch_size=32，gpu_max_batch_images=256，gpu_batch_wait_ms=10，NVDEC workers=32。
- embedding/URL/hash 缓存关闭；mm_cache_gpu_max_bytes=0，mm_cache_cpu_max_bytes=0，CUDA Graph 关闭。

## 新布局硬件计数器

本轮独立 NCU 采集当前 Q/K80、V72 布局与 native72，同一模拟输入，B1，每种一个代表 kernel、4 replay passes。仅用于归因，不以 profiler 耗时作为性能表数据。NCU 未控制 GPU clocks/cache，保留原始报告；不是全模型或 batch32 的利用率。

| 指标 | native72 | Q/K80,V72 |
|---|---:|---:|
| Tensor pipe active | 56.42% | 66.01% |
| 实际 Tensor 运算 TFLOP/次 | 12.19368 | 12.19368 |
| L2 tag requests/次 | 2,300,113,376 | 1,878,070,266 |
| issue active | 45.72% | 53.56% |
| eligible warp/scheduler | 0.562 | 0.667 |
| achieved occupancy | 23.44% | 23.44% |
| DRAM throughput | 2.89% | 3.37% |

L2 tag 请求降低 18.35%，L2 sector 降低 17.57%；TMA 指令请求均为 11,953,232。Tensor 运算数相同，驻留 launch 参数也相同：148 blocks、512 threads/block、128 registers/thread、228352 B dynamic shared memory。Tensor 活跃提高不是多算了补零维度，也不是增加驻留 CTA。

DRAM 字节从 2.134 GB 增至 2.239 GB，HBM 吞吐远未饱和；与减少片上请求、提高供数效率的诊断一致。未将收益进一步拆成 TMA 内部事务、边界和 cache 请求合并各自的比例。Q/K80 的 66.01% 是本轮实测，不引用此前全补 Q/K/V80 的 68.18%。

[本轮 NCU report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-qk80-counters-20260916-wq7726bz/counters.ncu-rep)、[原始 CSV](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-qk80-counters-20260916-wq7726bz/counters.csv)、[计数器汇总](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-qk80-counters-20260916-wq7726bz/counters-summary.json)。

## 限制与后续

收益已验证到真实视频的完整多模态 embedding；本轮没有测完整 LLM 请求、ViT→LLM 实际传输或独立请求负载到饱和。不能把本表称为 LLM 服务端到端收益，也不从小样本报告吞吐饱和点。不同视频尺寸、混合段长、其他架构和 dtype 不从本样本外推。

本项缓解了布局效率问题，剩余 K/V 与 softmax/correction/MMA 的交接依赖和单 CTA 驻留资源限制仍在。继续优化需针对这些交接等待，而不是追求通过补到 128 增加无效计算得到更高活跃率。此前补到 128 的控制没有显示相应延迟收益。

## 可复现产物

- [正式结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/benchmark-result.json)
- [正式脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/experiment.py)
- [正式命令与源码哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/command.json)
- [测量约定与环境](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/measurement-contract.json)
- [最终检查](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/final-checks.json)
- [未修改的 FA4 依赖哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-final-20260916-6449b356/dependency-hashes.json)
- [测试结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-pad80-tests-20260916-vjl4rb54/result.json)
- [测试日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-pad80-tests-20260916-vjl4rb54/test.log)
- [实现前源码快照](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-pad80-implementation-20260916-zdd1oh5j/before.json)
- [真实权重候选对照](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-pad80-choice-20260916-us272e67/benchmark-result.json)
- [单层候选原始样本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-padding-choice-20260916-rztw630p/bench-result.json)
- [NCU 命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-qk80-counters-20260916-wq7726bz/ncu-command.json)
