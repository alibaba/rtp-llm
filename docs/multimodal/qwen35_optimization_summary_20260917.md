# Qwen3.5 视频与 ViT 已完成优化汇总

更新日期：2026-09-17。范围：视频输入到多模态 embedding 就绪，包括预处理、解码、调度、ViT、token 组装和缓存。本文整理已有实现与已有测量，本次未启动新测试。

仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource；分支 feat/qwen35_vl_omega；HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。下文区分已提交内容、工作区实现和实验方案，不能仅凭 HEAD 判断实际运行版本。

## 1. 当前路径已接入的优化与修复

| 项目 | 已做内容及作用 | 当前状态 / 代码 |
|---|---|---|
| Qwen3.5 预处理对齐 | 对齐帧数、均匀采样、奇数帧的 temporal padding、时间戳和网格规则；明确显式宽高、请求每帧像素限制与 processor 总视频预算的优先级，避免重复采样或静默改变工作量。 | 已提交；[采样与尺寸规则](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/qwen3_vl_video.py:25) |
| 延迟到 GPU batch 内解码 | CPU 阶段只保留压缩视频和 metadata，GPU 执行前才展开帧；避免预处理进程提前建立 CUDA context、无界展开视频。 | 已接入；[GpuVideoInput 与准备阶段](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py:27) |
| NVDEC 硬件解码 | 使用 PyNvVideoCodec，选中帧保留在 GPU；融合 NV12→RGB 转换；处理 FFmpeg 动态库符号冲突、decoder surface 生命周期和失败 session 丢弃。 | 已接入；[NVDEC 与颜色转换](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py:125)、[RGB kernel](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/vision_kernels.py:9) |
| 有界并发解码 | 同一个 scheduler batch 内，用独立 decoder/session 和 CUDA stream 并发解码；有界预取、按输入顺序交回、record_stream 保护、异常排空。 | 工作区已接入；[VideoDecodePool](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py:279)。默认 workers=2，最近压测设32；并非32路一定更快。 |
| GPU resize / processor | 去掉 resize 前后的帧 D2H/H2D；bicubic、antialias、归一化和 patch 整理沿 CUDA 路径执行。 | 工作区已接入；[preprocess_video_cuda](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py:350)。CPU/GPU resize 不逐位等价，见下文正确性限制。 |
| 多媒体打包及预算 | 由 MMScheduler 组批，拼接 patch 后一次 packed ViT forward，再按媒体拆回；按 input patches、output tokens、workspace、attention work 做预算。CPU 保留 grid metadata，避免逐层取回形状。 | 已接入；[batch 预处理与 forward](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_mixin.py:209)。沿用 scheduler，不增加独立 MMAsyncExecutor。 |
| 视频 token 合并组装 | 在 mixin 内拼接时间戳、vision_start/end 与每帧视觉 embedding；整个 batch 一次查 CPU 文本词表、一次搬运所需文本 embedding，完整词表不驻留 ViT GPU。保持媒体顺序与 MRoPE 位置。 | 已提交；[批量 token 查表](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_mixin.py:282)、[视频组装](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_mixin.py:351) |
| 大 batch 正确性 | RoPE 大偏移使用 int64；LayerNorm 按完整归一化行分块，限制单次元素数，解决大 batch 索引回绕和数值错误。 | 已提交；[安全 LayerNorm](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:240)。当前 vLLM 风格模型继续使用此 LayerNorm；其 [rotary kernel](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/vllm_rotary.py:50) 也使用 int64 索引。 |
| Patch 投影 | 历史修复通过分块 Conv3d 保证大 batch 结果；当前 vLLM 风格实现把不重叠 patch 的 Conv3d 转成 F.linear/GEMM，避开旧的大规模 Conv3d 路径。 | 历史修复已提交；GEMM 实现在工作区；[patch GEMM](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py:265)。没有在本轮单独量化其收益。 |
| 融合 rotary / 位置处理 | 当前 vLLM 风格模型使用 fused rotary、cos/sin cache、Triton 位置插值与 spatial merge 重排，减少分散的张量操作与中间结果。 | 工作区已接入；[fused rotary 入口](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py:291)、[位置插值 kernel](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py:76)。没有把整体收益全部归因于这些算子。 |

NVDEC 并发限于同一 batch 的视频解码；当前尚未完成“下一批解码/预处理与当前批 ViT forward”的跨 batch 流水重叠。读取压缩文件、metadata、解码、resize 是不同阶段，不能用 NVDEC 耗时代替视频文件加载时间。

## 2. 已迁移的通用能力

| 能力 | 实现与当前使用方式 |
|---|---|
| 并发与准入 | 接通 VIT_CONCURRENCY、VIT_MAX_QUEUE_SIZE 和参考实现的 async_submit/get_embedding_result 生命周期；仍由 MMScheduler 负责计算组批。[MMProcessEngine](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/mm_process_engine.py:569) |
| GPU / CPU embedding cache | 字节预算、缓存分层、淘汰及状态观测。两级预算都为0时，每次创建独立任务，不共享在途计算；最近性能测量均关闭缓存。[缓存关闭时的行为](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/mm_embedding_cache.py:458) |
| 独立 hash cache 与 FeatureHash | 迁移 CPU/CUDA feature hash、独立 hash 字节预算、协议版本1和 cache metadata/keys 接口。最近计时保留生产 feature hash，没有通过跳过 hash 制造加速。[FeatureHash](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/cpp/multimodal_processor/FeatureHash.h) |
| ViT CUDA Graph | 已有固定 shape 的 graph 缓存与容量/patch 数限制。默认最大4096 patches；当前视频80960 patches，最新测试明确关闭graph，不能把其性能归功于graph。[graph 条件](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/vision_graph.py:54) |
| RPC 等待修复及参考传输能力 | 已修复重复媒体等待，迁移参考库的 pinned 张量转换等接口；最近无gRPC对照没有测量这些传输能力的收益。[RPC 请求路径](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/server/vit_rpc_server.py:136) |

此前自行扩展的 MMAsyncExecutor、关闭缓存时的额外任务去重、自定义 hash v2、额外路由/接收池等已经撤回，不列为保留优化。迁移范围见 [cache_and_concurrency.md](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/cache_and_concurrency.md)。

## 3. FA4：已验证的优化与当前启用状态

| 方案 | 作用与结果 | 当前状态 |
|---|---|---|
| 等长段 dense FA4 | 将独立等长段 view 成 [段数,段长,heads,dim]，使用原生 dense FA4 的 persistent 调度；保持段间隔离，不把各帧连接成全局 attention。最近同为D72的对照中，纯FA4耗时降低约12–13%。 | 旧 qwen3_5_moe_vit.py 已实现条件分流；当前 vLLM 风格 VisionAttention 仍直接调用 varlen。最近 dense 是实验进程内切换，尚未设为当前默认。 |
| Q/K 72→80 布局 | 在旧 RoPE 的转换阶段补零，V/output保持72，scale仍为72**(-0.5)；改善物理布局与供数效率。历史纯attention对照约降低8.25%。 | 旧ViT helper中已实现并验证；当前vLLM风格路径没有使用。最近varlen/dense对照两边都没有启用该padding。 |
| vLLM FA4 依赖对齐 | 独立测试vLLM固定FA4源码和新CUTLASS依赖；保留输入、输出和kernel记录。 | 属于实验，没有证明值得替换当前BF16路径；没有将依赖升级收益计入本次结果。 |

当前实际入口：[mixin 选择 Qwen3_5VllmVisionModel](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_mixin.py:71) → [VisionAttention varlen 调用](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py:312)。旧 [dense helper](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:67) 仍在仓库，但存在该函数不等于当前运行到了它。

当前工作区另外接入了 vLLM QKVParallelLinear、RowParallelLinear、ColumnParallelLinear 及其权重加载器。这属于实现/加载语义对齐，不能直接声称带来性能提升；相关变更发生在最近复测进程加载代码之后，没有重新运行过修改后的整ViT对照。

## 4. 已有测量能证明的收益

各行都是各自实验内部对比；不能跨行相加，也不能把不同分辨率和batch条件的绝对时间混用。

| 优化 | 测量条件 | 前 → 后 | 证据 |
|---|---|---|---|
| NVDEC 串行→2路 | 历史高分辨率，单请求4个视频，完整MM engine | 7724.03 → 7409.06 ms，耗时−4.08% | [NVDEC 并发记录](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_nvdec_parallel_20260916.md) |
| CPU→GPU resize | 历史1216×2112、46帧、固定batch32，完整MM engine | 41.733 → 22.802 s，耗时−45.36%，视频吞吐+83.03% | [GPU resize 记录](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_gpu_resize_20260916.md) |
| 旧ViT varlen→dense | 历史每段10032 tokens，固定batch32，完整MM engine | 22894.30 → 22342.84 ms，耗时−2.41% | [旧dense接入验证](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_optimization_20260916.md) |
| 旧dense D72→Q/K80 | 同一历史高分辨率、固定batch32，完整MM engine | 22235.07 → 21885.14 ms，耗时−1.57% | [Q/K80验证](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_padding_20260916.md) |

GPU resize 是已验证的主要阶段优化，但与 CPU resize 不逐位等价：历史真实视频最终 embedding 余弦相似度约0.999442，相对L2差异约3.34%，位置编码一致。此前验证覆盖形状、finite、重复稳定性和媒体隔离；尚不能据此宣称最终LLM视频理解质量完全等价。

## 5. 最近同场景 varlen / dense 结果

物理GPU0，L20D/SM103；真实397B模型的ViT，BF16、16 heads、Q/K/V均72维。请求并发仅8和16，每种后端每档3轮，每轮20秒；脚本总repeat=6是两个后端交替各3轮。表中取中位QPS对应完整轮次，平均RT与百分位来自同一轮。

| FA4 | 请求并发 | QPS | 平均 RT ms | P99 ms |
|---|---:|---:|---:|---:|
| varlen | 8 | 5.5772 | 1423.91 | 1509.01 |
| dense | 8 | 5.7789 | 1370.81 | 1401.02 |
| varlen | 16 | 5.7815 | 2725.05 | 2824.94 |
| dense | 16 | 5.9624 | 2643.15 | 2732.60 |

同一视频实际第一层Q/K/V重放，按原stride复制成固定batch，单层FA4 CUDA-event计时如下。这里只包括正常FA4 API，不包含解码、RoPE、linear或其余ViT；不是服务请求RT。此独立kernel阶段使用6组交替样本，与上述每档3轮的请求压测分开统计。

| 视频 batch | varlen ms | dense ms | dense耗时减少 |
|---:|---:|---:|---:|
| 1 | 1.6959 | 1.4589 | 13.97% |
| 8 | 13.8086 | 12.1431 | 12.06% |
| 16 | 28.1141 | 24.4687 | 12.97% |

这组完整MM请求吞吐实测提高约3–4%，纯FA4耗时减少约12–13%。两种后端的完整embedding对照逐元素一致，原视频/变色视频的并发8、16正确性通过；1484条正式请求全部成功。CUDA完成等待放在实际生产stream，不只计算enqueue。

输入固定为46帧、显式height=704,width=1280，grid=[23,44,80]，每视频80960 patches、23个3520长度的attention段。纯视觉token为20240，另有184个时间戳/标记token，最终embedding=[20424,4096]。虽然请求仍带min_pixels=2500000,max_pixels=73728000,fps=6,max_frames=180，但显式宽高优先；此表不能与上文1216×2112的历史实验横向计算提升。

计时从本地请求构造/转换与engine调用开始，到GPU embedding就绪；包含文件加载、NVDEC、GPU预处理、排队、ViT、merger、token组装和生产hash。排除gRPC序列化/网络、输出D2H、RDMA和LLM。这是多模态模块性能，不是完整LLM推理性能。

同机GPU4–7有短时负载，GPU时钟未锁定，实际组批也自然变化，几个百分点的整体差异存在环境波动。复测中磁盘上的ViT linear/loader源码发生变化，version-after.json因此标记false；测试模块已加载且脚本不热重载。这里报告的是进程启动时的实现与进程内FA4 A/B，不将数据当作后来修改版本的性能验收，也不因无关磁盘修改否定已执行的FA4对照。

最近测试配置（配置上限不等于实际测试并发）：

```yaml
vit_concurrency: 64
vit_max_queue_size: 64
gpu_max_batch_size: 64
gpu_max_batch_images: 256
gpu_batch_wait_ms: 10
QWEN35_VIDEO_DECODE_WORKERS: 32
QWEN35_VIDEO_BACKEND: nvdec
mm_cache_gpu_max_bytes: 0
mm_cache_cpu_max_bytes: 0
mm_hash_key_cache_max_bytes: 0
mm_cache_item_num: 0
url_cache_item_num: 0
use_igraph_cache: false
CUDA Graph: 关闭
实际请求并发: [8, 16]
每种后端每档重复: 3
```

原始证据：[完整统计](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/comparison.json)、[逐请求数据](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/load/requests.jsonl)、[正确性与运行时](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/engine-state.json)、[独立FA4样本](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/isolated.json)、[实际后端计数](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/dispatch.json)、[执行命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/process.json)、[额外NVML采样](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-varlen-dense-recheck-20260917-xbm17qkf/nvml-resources.jsonl)。

## 6. 测过但未采用的方案

| 方案 | 结果 / 取舍 |
|---|---|
| 多CUDA stream同时跑多个ViT microbatch | 相对相同microbatch的单stream，没有明确速度收益，显存反而更高；保持实验实现，没有改生产默认。 |
| 更大的128×160 attention tile | 随机单层曾有改善，但真实27层ViT完整输出未通过预设数值门槛，未接入。 |
| 修改softmax时序和warp寄存器分配 | 原生dense已有主要收益，叠加自定义kernel的收益太小，没有为此替换正式依赖。 |
| split_P_arrive 96→64 | B1慢约0.64%，B32慢约2.50%，未采用。 |
| Q/K/V补到128以提高Tensor活跃率 | 活跃率提高但无效运算增加，实际耗时更慢；没有作为优化落地。 |
| 在当前vLLM路径启用dense与Q/K80 | 两者历史上已验证，但当前入口尚未完成接入；不能算成已启用默认。 |

证据：[多stream对照](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_concurrent_video_20260916.md)、[FA4候选取舍](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_optimization_20260916.md)、[split-P实验](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_split_p_20260917.md)、[FA4布局与瓶颈分析](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_pure_analysis_20260917.md)。

## 7. 已完成与待完成的边界

- 已完成：NVDEC与GPU预处理接入、可控解码并发、参考缓存/并发能力迁移、视频token组装、大batch正确性修复、多个FA4候选的真实输出/性能验证。
- 当前待接入：将已验证的等长dense优化接到实际vLLM风格ViT入口；Q/K80需要在该入口重新核对数值和完整路径收益。
- 当前尚未证实收益：最新vLLM parallel linear及权重加载改动，不能借用变更前的RT宣称提速。
- 尚未完成：跨batch解码/ViT流水重叠；GPU resize变更后的完整LLM质量回归；NVDEC pool退出等待超时的生产修复。最近测试进程已经退出，harness收尾不等于生产退出问题已修复。

可定位的已有提交：

| commit | 内容 |
|---|---|
| 1e658084a9 / a716ee9dc1 / 17646437dc | 视频预处理、timestamp/MRoPE与NVDEC语义对齐 |
| f0806e3585 | 多模态cache、并发、feature hash、graph等迁移与视频能力 |
| 4b2f8d0881 | ViT RPC重复媒体等待修复 |
| 5c2d78c044 | 大batch RoPE / LayerNorm / patch投影正确性 |
| d4bbfd789f | 将完整视频embedding和token组装移到mixin |

NVDEC pool、GPU resize、dense/QK80旧入口增强及vLLM风格ViT等仍含工作区未提交内容。本文不把所有能力归入某一个commit，也不把历史性能数据当作当前所有改动叠加后的总收益。
