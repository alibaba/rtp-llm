原版与 vLLM 移植版：并发16和ViT阶段对比（2026-09-17）

同机GPU0、同输入、同权重、同依赖重测。原版中位轮 **5.847 QPS / 平均RT 2717.230 ms**；移植版 **5.919 QPS / 平均RT 2669.385 ms**，吞吐变化 +1.23%。各自三轮范围有重叠，不能把这点差异宣称为明确整体提速。

“原版”明确指移植vLLM纯ViT之前保存的RTP实现，不是另一个随意选择的main版本。快照来自 [移植前快照](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-vllm-full-20260917-4ug5tgc6/baseline)；本次复制所需源码至 [隔离源码目录](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/original-source)，仅测试进程用import hook加载。当前分支和生产文件未回退、未替换。

两版仍在HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f对应工作区基础上；67个共用多模态文件哈希一致。生产差别集中在ViT模型选择、移植实现和权重加载。旧版是Qwen3_5MoeVisionModel+nn.Linear，移植版是Qwen3_5VllmVisionModel+QKVParallelLinear/ColumnParallelLinear/RowParallelLinear。Torch2.11.0+cu130、CUDA13、同一FA4 wheel，BF16 ViT，SM103 L20D。

**共同测量条件**

真实Qwen3.5-397B-A17B-FP8 checkpoint中的ViT，456,010,480个参数、27层。视频是之前的第1条请求；fps=6、min_pixels=2500000、max_pixels=73728000、max_frames=180，并保留历史显式width=1280、height=704。46帧、grid=[23,44,80]、80,960输入patch/视频；输出20,240纯视觉token，加184个标记/时间戳后是[20424,4096]。两版333组参数和原/反色视频预处理张量的哈希完全相同。

vit_concurrency=16、gpu_max_batch_size=16、gpu_max_batch_images=256、gpu_batch_wait_ms=10、decode_workers=32；NVDEC+GPU预处理、CUDA graph关闭，embedding/CPU/GPU/hash/URL cache关闭且实际hit/resident/inflight dedup为0。同步engine入口由16线程施压，未经过异步admission队列。

范围：MMProcessEngine接收本地视频URL→GPU完成的组装embedding，包括读取、元数据、解码、预处理、排队、ViT、组装和feature hash。不含gRPC格式化/序列化/网络、RDMA或LLM。这是多模态子系统性能，没有测试完整LLM响应或ViT→LLM传输。重复读取可能命中OS page cache，不是冷盘基准。

**正式C16结果：无profiler**

每版B1–B16完整预热，B16连续两次变化<5%后测试。各3轮×30秒发送窗口，含在途请求排空，按QPS取中位完整一轮。RT以单请求GPU完成为终点，QPS=成功数/实际wall秒数。

| 指标 | 原版 | vLLM移植版 |
| --- | --- | --- |
| QPS | 5.847 | 5.919 |
| 平均RT/ms | 2717.230 | 2669.385 |
| P50/ms | 2738.246 | 2701.725 |
| P99/ms | 2753.347 | 2726.511 |
| 三轮请求总数 | 546 | 561 |
| 错误数 | 0 | 0 |
| 中位轮batch:次数 | {'1': 12, '15': 11} | {'3': 12, '13': 12} |
| 中位轮peak allocated/GiB | 38.665 | 34.435 |
| 采样显存最大/MiB | 54151.0 | 53103.0 |

| 版本 | 轮次 | 成功数 | QPS | 平均RT/ms | P99/ms | batch:次数 |
| --- | --- | --- | --- | --- | --- | --- |
| original | 1 | 177 | 5.825 | 2726.810 | 2769.923 | {'1': 12, '15': 11} |
| original | 2 | 192 | 5.952 | 2654.036 | 2709.124 | {'3': 12, '13': 12} |
| original | 3 | 177 | 5.847 | 2717.230 | 2753.347 | {'1': 12, '15': 11} |
| vllm_port | 1 | 177 | 5.751 | 2762.373 | 2809.017 | {'1': 12, '15': 11} |
| vllm_port | 2 | 192 | 5.934 | 2647.846 | 2729.750 | {'5': 12, '11': 12} |
| vllm_port | 3 | 192 | 5.919 | 2669.385 | 2726.511 | {'3': 12, '13': 12} |

显存峰值对应各自中位轮实际形成的B15/B13，不应直接据此声称同B16内存降低。

**固定B16：各阶段同尺寸对比**

两个独立Nsight进程各自预热后，采集64个真实C16请求，再采集4次“一次请求携带16视频”的固定B16控制组。后者用于同尺寸分阶段，不冒充真实C16吞吐。下表为4个B16平均；变化=(移植版/原版−1)，耗时负值表示更快。采集耗时不用于替代上面的无profiler性能。

| 阶段 | 原版/ms | 移植版/ms | 差值/ms | 变化 |
| --- | --- | --- | --- | --- |
| NVDEC+GPU预处理 | 398.889 | 396.156 | -2.733 | -0.69% |
| 纯ViT | 2258.043 | 2293.441 | 35.398 | +1.57% |
| embedding组装 | 10.728 | 10.454 | -0.274 | -2.56% |
| multimodal_batch合计 | 2673.807 | 2705.365 | 31.558 | +1.18% |
| scheduler整批wall | 2678.087 | 2709.542 | 31.456 | +1.17% |

除scheduler整批wall为CPU墙钟外，以上为CUDA event elapsed，包含GPU流上的等待。预处理、ViT、组装之外还有拼接/搬运/同步；feature hash在multimodal_batch外、scheduler内。每B16是16个视频整批的耗时，不是单视频RT。

**纯ViT内部：每个B16的27层累计**

由NVTX层/算子边界和CUDA launch correlation ID归因到真实GPU kernel，按最内层算子独占计数。下表是kernel duration之和、4批均值；含内存copy kernel。显式memcpy和未覆盖间隙单列，不能把kernel sum误称为完整wall时间。

| 模块 | 原版/ms | 移植版/ms | 差值/ms | 变化 |
| --- | --- | --- | --- | --- |
| Patch embedding | 202.664 | 2.970 | -199.694 | -98.53% |
| 位置/RoPE元数据 | 8.148 | 2.209 | -5.939 | -72.89% |
| QKV投影 | 169.304 | 164.289 | -5.015 | -2.96% |
| Attention布局copy | 0.000 | 186.031 | 186.031 | 新增/原为0 |
| RoPE | 225.163 | 158.276 | -66.887 | -29.71% |
| FA4 | 595.594 | 821.606 | 226.012 | +37.95% |
| Attention输出投影 | 68.719 | 64.623 | -4.096 | -5.96% |
| LayerNorm | 244.018 | 208.328 | -35.690 | -14.63% |
| MLP FC1 | 206.023 | 201.343 | -4.681 | -2.27% |
| GELU | 181.981 | 153.347 | -28.633 | -15.73% |
| MLP FC2 | 253.740 | 235.944 | -17.796 | -7.01% |
| Residual add | 70.515 | 69.926 | -0.588 | -0.83% |
| Merger/projector | 21.569 | 20.101 | -1.468 | -6.81% |
| 其余ViT kernel | 1.285 | 1.284 | -0.001 | -0.09% |

| 版本 | B16 ID | 预处理/ms | 纯ViT event/ms | 整批wall/ms |
| --- | --- | --- | --- | --- |
| original | 8 | 392.174 | 2250.604 | 2661.757 |
| original | 9 | 405.027 | 2263.507 | 2691.274 |
| original | 10 | 402.995 | 2264.514 | 2689.642 |
| original | 11 | 395.359 | 2253.548 | 2669.675 |
| vllm_port | 8 | 396.989 | 2290.673 | 2707.900 |
| vllm_port | 9 | 407.780 | 2294.035 | 2721.665 |
| vllm_port | 10 | 398.604 | 2290.843 | 2708.308 |
| vllm_port | 11 | 381.249 | 2298.215 | 2700.297 |

| 预算口径 | 原版/ms | 移植版/ms |
| --- | --- | --- |
| 纯ViT首末GPU activity区间 | 2258.025 | 2293.430 |
| 纯ViT kernel累计 | 2248.723 | 2290.276 |
| 纯ViT显式memcpy累计 | 0.134 | 1.194 |
| 纯ViT kernel+copy时间并集 | 2248.844 | 2291.460 |
| 首末activity之间未覆盖间隙 | 9.181 | 1.969 |
| event区间超出首末activity的时间 | 0.018 | 0.012 |
| batch内其他拼接/等待 | 6.147 | 5.315 |
| feature hash CPU wall | 3.414 | 3.340 |

NVTX边界验证：每版12个采集批次都找到27层、27个FA4 kernel，全部kernel都有launch归因。event比首末activity更长的部分反映边界内GPU提交前后等待；GPU activity内部间隙另外统计。CPU hash与GPU event有异步关系，不强行混为严格可加的GPU分项。

**实现差异和调用参数**

原版FA4保持原有dense等长路径：Q/K从逻辑head_dim72补零到80，V与输出72，scale保持72^(-0.5)；B16是368个长度3520的独立segment。移植版保持vLLM varlen72路径。两者逻辑输入一致、内部kernel形状不同，这是本次比较的实现差异。旧版RoPE频率buffer为FP32，移植版cos/sin cache为BF16；这也是跨版本数值不逐位相同的实现差别之一。采集记录了实际Q/K/V shape与stride，见各版stage-events.json的fa4_inputs。

原版patch embedding用Conv3d，移植版用等价patch GEMM；原版RoPE使用原有融合路径，移植版先做QK布局contiguous再调用vLLM rotary kernel。位置元数据准备也不同。FC/QKV参数实际内容相同。性能归因须看这些分项共同变化，不能仅从FA4时间判断整条ViT是否提速。

**读取、排队和解码拆解**

| 预处理子项 | 原版/ms | 移植版/ms | 口径 |
| --- | --- | --- | --- |
| NVDEC worker累计 | 4406.452 | 4318.757 | 各worker CPU wall相加 |
| NVDEC worker并集 | 310.813 | 304.631 | 重叠区间去重 |
| GPU resize kernel | 65.373 | 65.371 | kernel累计 |
| GPU normalize/fold kernel | 23.650 | 23.645 | kernel累计 |
| NV12→RGB kernel | 1.391 | 1.392 | kernel累计 |

NVDEC worker包含包处理、驱动调用、同步与CUDA后处理，不等同硬件decoder引擎纯工作时间。表中存在重叠/包含，不能直接加总。

| C16采集请求项 | 原版平均/ms | 移植版平均/ms |
| --- | --- | --- |
| input_ms | 58.330 | 55.686 |
| queue_ms | 712.195 | 726.542 |
| batch_service_ms | 1859.760 | 1870.116 |
| return_ms | 5.253 | 5.282 |
| total_ms | 2635.538 | 2657.627 |
| file_read_ms | 2.419 | 2.124 |
| video_metadata_ms | 33.233 | 32.175 |

这张请求表来自真实C16诊断，实际组批分别为 {'3': 4, '13': 4} 和 {'3': 4, '13': 4}。这项还包含排队和线程调度，不能将其直接等同固定B16的纯kernel对比。每版逐请求均按入口→提交→批开始→批结束→GPU完成对账，误差为0；所属批次处理不除以batch size。

设备使用默认动态频率，未锁频。Nsight诊断中GPU0的采样SM频率原版约1372–2032 MHz、移植版约1597–1747 MHz，显存频率均3996 MHz；采样稀疏且时间戳在查询前记录，不能精确对齐某个kernel。其他GPU在基准采样期间均无负载。分项差异包含算子布局和动态频率/功耗影响，不能把LayerNorm等每项缩短都解释为独立代码优化。

**正确性和局限**

| 视频 | cosine | relative RMSE | mean abs | max abs |
| --- | --- | --- | --- | --- |
| 0 | 0.999738398 | 2.2877% | 0.001556573 | 3.469 |
| 1 | 0.999616690 | 2.7689% | 0.001695360 | 1.641 |

两版各自串行/并发输出与本版参考bitwise一致，positions一致，有限值检查通过，原/反色视频输出不同。跨版本参数、预处理输入完全相同，但embedding不是bitwise相同；上表是整张输出的差异，不是只比抽样。通过预先声明的cosine≥0.999、relative RMSE≤5%的实现对比门槛；不据此声称完整LLM输出等价。三轮负载每请求只做轻量校验，完整比较在独立正确性阶段进行。

本次是性能测量，没有修改生产实现、没有运行Bazel或完整LLM测试。未采NCU，因此不根据activity占比声称Tensor Core满、MFU或带宽饱和。四个运行均复现解码pool.close()超过10秒，工具保存结果后退出，关闭问题尚未修复；最终无本次残留进程，GPU0已回到117 MiB / 0%，详见cleanup.json。

**产物与复现**

| 材料 | 路径 |
| --- | --- |
| 全部对比数据 | [comparison.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/comparison.json) |
| 逐层对比CSV | [per-layer-comparison.csv](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/per-layer-comparison.csv) |
| 原版trace | [original engine.nsys-rep](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/original/diagnostic/engine.nsys-rep) |
| 移植版trace | [vllm_port engine.nsys-rep](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/vllm_port/diagnostic/engine.nsys-rep) |
| 原版阶段数据 | [original stage-summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/original/diagnostic/stage-summary.json) |
| 移植版阶段数据 | [vllm_port stage-summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/vllm_port/diagnostic/stage-summary.json) |
| 原版正式性能 | [original summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/original/baseline/summary.json) |
| 移植版正式性能 | [vllm_port summary.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/vllm_port/baseline/summary.json) |
| 共用源码核对 | [common-source-audit.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/common-source-audit.json) |
| 旧版源码来源/哈希 | [original-source.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/original-source.json) |
| 采集资源时钟记录 | [diagnostic-resources.jsonl](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/diagnostic-resources.jsonl) |
| 统一执行入口 | [controller.py](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/controller.py) |
| 清理状态 | [cleanup.json](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/cleanup.json) |

original 代表B16 batch=9：NVTX 14.506618815–17.197876160 s；纯ViT GPU 14.913663350–17.177154069 s；layer_12 GPU 16.033421540–16.108175412 s。时间以各自Nsight导出的0为原点；全部4批ViT相邻变化均<1%，全部纳入平均。

vllm_port 代表B16 batch=9：NVTX 15.044858669–17.766509992 s；纯ViT GPU 15.454594915–17.748616918 s；layer_12 GPU 16.477841461–16.564214274 s。时间以各自Nsight导出的0为原点；全部4批ViT相邻变化均<1%，全部纳入平均。
