Qwen3.5-397B ViT 并发 16 复测与阶段耗时（2026-09-17）

GPU0 / L20D / FA4，五轮共 930 个请求、0 错误。按 QPS 排序取中位完整一轮（第4轮）：**5.870 QPS，平均 RT 2725.230 ms，P50 2728.266 ms，P99 2737.189 ms**。该轮实际组成 12 个 B16。

计时从 MMProcessEngine.mm_embedding_impl 接收本地视频 URL 开始，到 embedding 的 CUDA 完成事件结束。包含读文件、元数据/采帧规划、NVDEC、GPU 预处理、调度排队、ViT、时间戳/标记组装和生产 feature hash。不含 gRPC 格式化、序列化、网络、RDMA 或 LLM；没有启动 gRPC 监听服务。

**输入、版本和配置**

视频：[第1条请求](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4)。原始 233 帧、30 fps、1280×720；SHA256 d8b10b3fd5669c6d4db8cab1d0b2cd970d4e9db09a2ce09059b3a3e3a76987b0。使用历史约2万token的有效尺寸，显式 width=1280、height=704，同时 fps=6、min_pixels=2500000、max_pixels=73728000、max_frames=180。尺寸并非由像素上下限自动决定。

单视频采样46帧，grid=[23,44,80]，80,960个输入patch，20,240个纯视觉token；加184行时间戳和边界标记后，最终 **embedding=[20424,4096]、positions=[20424,3]**。B16含1,295,360个patch、368个长度3520的attention segment。FA4非causal，16 heads、head_dim=72、BF16。

真实checkpoint为 /mnt/nas1/hf/Qwen3.5-397B-A17B-FP8，只加载ViT及所需文本embedding。ViT有456,010,480个参数、27层、hidden_size=1152、intermediate_size=4304、无deepstack；线性层BF16，checkpoint名称FP8不代表本次ViT用FP8。

配置：vit_concurrency=16，vit_max_queue_size=16，gpu_max_batch_size=16，gpu_max_batch_images=256，gpu_batch_wait_ms=10，QWEN35_VIDEO_DECODE_WORKERS=32，NVDEC和GPU resize，CUDA graph关闭。embedding GPU/CPU cache、hash cache、URL cache均关闭；实际hit、resident、inflight_dedup均为0。同步engine入口由16个调用线程施压，没有经过异步入口的admission队列。

源码：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource，feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f 加当时工作区改动。Torch 2.11.0+cu130 / CUDA13 / SM103；QKV为移植的QKVParallelLinear，权重类型ModelWeightParameter，FA4来自CUDA13 Bazel Python runfiles。源码及运行时文件哈希前后一致；本次只新增测试工具和报告，没有改生产实现。

**正式性能：无 profiler**

预热真实engine的B1–B16并检查每批输出，随后B16连续两次相邻耗时变化小于5%才开始负载。每轮发送窗口30秒，实测duration包含在途请求排空。QPS=成功数/实测wall秒数；RT=请求返回时间减入口时间。百分位线性插值。取中位QPS对应整轮，而非拼接不同轮的最佳指标。

| 轮次 | 成功数 | 秒数 | QPS | 平均RT/ms | P50/ms | P99/ms | batch:次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 192 | 32.659 | 5.879 | 2666.071 | 2725.244 | 2746.308 | {'16': 7, '9': 5, '7': 5} |
| 2 | 177 | 30.529 | 5.798 | 2739.989 | 2751.639 | 2770.624 | {'1': 12, '15': 11} |
| 3 | 192 | 32.468 | 5.913 | 2677.869 | 2700.479 | 2725.442 | {'14': 12, '2': 12} |
| 4 | 192 | 32.707 | 5.870 | 2725.230 | 2728.266 | 2737.189 | {'16': 12} |
| 5 | 177 | 31.477 | 5.623 | 2824.381 | 2861.015 | 3014.988 | {'1': 12, '15': 11} |

中位轮peak allocated=42.11 GiB，采样显存最大=53103 MiB，采样GPU utilization平均=87.0%（15个样本，约2秒一次；不是MFU）。全部五轮最大RT=3015.916 ms。本次未重现之前约12秒长尾，但不能据此证明之前长尾的具体原因。

16个并发完整embedding与各自串行参考bitwise相等，positions相等、无NaN/Inf；原视频与反色视频参考embedding不同。预热批次全部输出也与参考比较。930个负载请求做轻量输出校验，并非每个都做完整逐元素比较。6项单测沿用同源码上一轮结果，本次未重复执行Bazel。

**独立采集：固定B16分解**

第二个进程预热后开启Nsight。先跑64个真实C16请求，再用“一次请求携带16个视频”固定形成B16，连续采集4批。固定B16只用于阶段测量，不用于并发QPS。以下取全部4批平均，不挑最快批；纯ViT相邻波动分别为 0.569%, 0.259%, 0.327%。

| 阶段 | B16平均/ms | 口径 |
| --- | --- | --- |
| 整批scheduler处理 | 2749.461 | CPU wall，含完成等待和hash |
| NVDEC+GPU预处理 | 431.285 | CUDA event elapsed，含等待解码 |
| 纯ViT | 2296.204 | CUDA event，patch→27层→merger |
| embedding/标记组装 | 10.910 | 16段CUDA event elapsed之和 |
| 其余拼接/搬运/等待 | 6.646 | 外层减上述三阶段 |
| multimodal_batch合计 | 2745.045 | 上述四阶段合计 |
| 生产feature hash | 3.426 | CPU wall，在multimodal_batch外、整批处理内 |

以上为整批16个视频耗时，不是单视频延迟。纯ViT占multimodal_batch约83.6%。CUDA event elapsed包含流上的等待，并不等于kernel执行时间。CPU hash区间不能与GPU分项混为严格同口径。

| B16 ID | 预处理/ms | 纯ViT/ms | 组装/ms | 整批wall/ms |
| --- | --- | --- | --- | --- |
| 8 | 446.258 | 2285.352 | 11.320 | 2754.792 |
| 9 | 443.615 | 2298.356 | 10.278 | 2762.474 |
| 10 | 392.135 | 2304.317 | 12.039 | 2720.801 |
| 11 | 443.130 | 2296.791 | 10.002 | 2759.776 |

**纯ViT内部：27层累计kernel耗时**

NVTX明确标记层/算子边界，经CUDA runtime/driver correlation ID关联实际kernel，按最内层模块独占归类。下表是每B16的kernel duration累计值，再对4批求平均。占比以纯ViT GPU首末activity区间2296.193 ms为分母。12个采集批次全部验证为27层、27个FA4 kernel；40,088个kernel均找到launch correlation。

| 模块 | 每B16累计/ms | 占纯ViT区间 |
| --- | --- | --- |
| Patch embedding | 2.970 | 0.13% |
| 位置插值 | 1.886 | 0.08% |
| RoPE元数据 | 0.324 | 0.01% |
| QKV投影 | 164.568 | 7.17% |
| Attention布局/contiguous copy | 187.251 | 8.15% |
| RoPE | 158.044 | 6.88% |
| FA4 | 818.896 | 35.66% |
| Attention输出投影 | 64.636 | 2.81% |
| LayerNorm ×54 | 210.943 | 9.19% |
| MLP FC1 | 201.504 | 8.78% |
| MLP GELU | 154.862 | 6.74% |
| MLP FC2 | 235.667 | 10.26% |
| Residual add | 69.970 | 3.05% |
| Merger/projector | 20.148 | 0.88% |
| 其余ViT kernel | 1.284 | 0.06% |

ViT kernel sum=2292.955 ms；kernel+memcpy的去重activity union=2294.140 ms；GPU首末activity wall=2296.193 ms；未被activity覆盖间隙=2.053 ms。显式memcpy sum=1.194 ms。activity有少量重叠，sum不能直接代替union。Attention contiguous是copy kernel，已计入187.251 ms，不是显式memcpy activity。

FA4占35.7%；MLP FC1+GELU+FC2=592.033 ms；QKV+输出投影=229.204 ms。LayerNorm+RoPE+布局copy+GELU+residual=781.071 ms。主要时间已经在GPU kernel内，空隙仅约2ms；不代表Tensor Core满，也不是显存带宽饱和度结论。

Attention布局copy对应 [QK contiguous路径](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py:500)。代表层观察到16个direct_copy kernel位于QKV后、RoPE前；源码还有输出contiguous。未仅凭耗时声称达到带宽上限。

**解码与预处理的并发口径**

B16的16个NVDEC worker CPU wall累计平均4769.154 ms，重叠后的区间并集仅341.756 ms，证实worker在并发。区间含硬件解码调用、包循环、同步和CUDA后处理，不能把它称为NVDEC硬件引擎独占耗时。

| 预处理子项 | 每B16/ms | 口径 |
| --- | --- | --- |
| NVDEC worker并集 | 341.756 | CPU interval union |
| GPU resize | 66.905 | CUDA event elapsed之和 |
| GPU resize kernel | 65.373 | kernel duration之和 |
| GPU normalize/fold | 27.593 | CUDA event elapsed之和 |
| GPU normalize/fold kernel | 23.647 | kernel duration之和 |
| NV12→RGB kernel | 1.392 | kernel duration之和 |

这些行互有包含/重叠，不可相加。外层431.285 ms是整批解码+预处理对后续ViT的等待区间。本地视频反复读取可能命中OS page cache，未测冷盘。

**真实C16采集：请求耗时对账**

64个独立请求形成4个B2和4个B14。采集和调度时序会影响组批，因此不能把固定B16的阶段数字拼到这些请求上，也不能用采集组替代正式QPS。按request ID对齐得到：

| 阶段 | 平均/ms |
| --- | --- |
| 入口到提交scheduler | 54.817 |
| 排队至所属批次开始 | 537.164 |
| 所属批次完整处理 | 2099.751 |
| 批结束到GPU-complete返回 | 6.022 |
| 总RT | 2697.755 |

54.817+537.164+2099.751+6.022=2697.755 ms，逐请求对账误差为0。所属批次处理不除以batch size。输入阶段内文件读取平均2.554 ms，视频元数据/采帧规划平均32.202 ms，其余是工作项创建和线程等开销；均为含profiler的CPU wall。

**时间线与证据**

代表batch ID9：NVTX 14.701131661–17.463590074 s；纯ViT GPU 15.146713503–17.445057657 s。时间是导出Nsight时间线的ns转秒，以导出0为原点。

代表layer_12（第13层）GPU约16.162361–16.255927 s：norm1→QKV→16个布局copy→RoPE→FA4→输出投影→residual→norm2→FC1→GELU→FC2→residual。FA4开始16.187291 s，持续34.370 ms。全部层的起止、分类和gap见CSV。

| 材料 | 链接 |
| --- | --- |
| 完整trace | [engine.nsys-rep](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/engine.nsys-rep) |
| SQLite | [engine.sqlite](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/engine.sqlite) |
| 每层耗时 | [per-layer.csv](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/per-layer.csv) |
| 阶段原始/汇总 | [stage-events.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/stage-events.json) / [stage-summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/stage-summary.json) |
| kernel归因 | [nsys-analysis.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/nsys-analysis.json) / [all-kernels.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/diagnostic/all-kernels.json) |
| 正式性能 | [summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/summary.json) / [requests.jsonl](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/load/requests.jsonl) |
| 配置/资源 | [config.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/config.json) / [resources.jsonl](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/resources.jsonl) |
| 源码快照 | [source-before.patch](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/source-before.patch) / [version-before.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/version-before.json) |
| 复现入口 | [controller.py](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/controller.py) / [profile_controller.py](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/profile_controller.py) |
| 清理状态 | [cleanup.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c16-retest-profile-20260917-bwan4zpa/cleanup.json) |

本次只采集Nsight Systems，没有Nsight Compute，因此不报告MFU/MTU或内存带宽利用率。Nsight数据用于阶段归因，没有替代无profiler的正式性能结果。

两次运行退出时均复现NVDEC pool.close()超过10秒的旧问题。工具保存结果后退出进程，退出码0、无未完成output event；全部本次进程已退出，GPU0恢复117 MiB / 0%。推理测量已完成，关闭超时本身仍未修复。没有操作其他GPU或进程。
