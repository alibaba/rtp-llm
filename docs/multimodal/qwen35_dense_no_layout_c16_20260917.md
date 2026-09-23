# Qwen3.5 ViT：删除 QK 布局拷贝，恢复 dense FA4

日期：2026-09-17。代码位于 /home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，保留当前未提交修改。

## 改动

- 删除 attention 中 RoPE 前的 QK contiguous：Triton RoPE 直接读取 packed QKV 的 strides。
- 在 L20D / SM103、BF16、16 heads、等长且长度 ≥1024 的输入上恢复原 dense FA4 条件；本视频每段3520 patches。其他形状保留 varlen。
- Q/K 从逻辑72通道补零到80，V保持72，softmax scale仍为72**-0.5。补零融合到 RoPE 的输出写入，无额外 padding kernel。
- 序列长度由 CPU metadata 直接传入，避免每层回读 CUDA cu_seqlens。更新来源说明及回归测试。

## 验证

- 最终11项测试全部通过，无跳过：strided QK、部分/完整 RoPE、补零、独立FP32 SDPA参照、dense/varlen媒体隔离、CUDA graph重放。
- 最终代码的原视频与反色视频完整 embedding 均与修改前逐元素一致：每份83,656,704个元素，差异元素0，cosine=1，relative RMSE=0，positions一致，全部有限值。
- 权重使用真实 Qwen3.5-397B-A17B-FP8 检查点中的 ViT；333个参数张量校验，ViT共456,010,480参数，BF16。
- Nsight核对：10个batch，每个27个FA4 kernel；324次RoPE输入均非连续视图、输出80通道；attention_layout kernel计数为0。
- 相同输入的串行/批量输出逐元素一致；不同视频保持不同输出；缓存hit、inflight dedup及驻留项均为0。

## 并发16观测结果（无Profiler）

计时范围：MMProcessEngine接收视频URL至GPU-complete embedding，包含文件/metadata、NVDEC与GPU预处理、排队/合批、ViT、token组装和feature hash；排除gRPC序列化、网络、RDMA及LLM。

| 指标 | 修改前vLLM移植版 | dense且无布局拷贝 |
|---|---:|---:|
| QPS（3轮中位数对应完整一轮） | 5.919 | 6.612 |
| 平均RT（ms，同一轮） | 2669.385 | 2390.449 |
| P99 RT（ms，同一轮） | 2726.511 | 2469.108 |
| 3轮总请求 | 561 | 631 |
| 3轮错误数 | 0 | 0 |

本次三轮QPS：6.611753, 6.615997, 6.478107。代表轮实际合批：14批B3、13批B13；C16不等于每批固定16。

**比较限制：**本次基准期间观察到其他GPU同步增加显存占用，GPU0退出本次进程后也有显存占用变化；来源不在本次可见进程范围。使用默认动态时钟，未锁频。以上保留为当时观测值，不能当作排除背景负载影响后的稳定加速结论。诊断另起进程，在GPU0满足util≤5%、memory<1GiB连续3次采样后启动；完整资源记录见diagnostic-resources.jsonl。

## 固定B16诊断

以下为单请求包含16个视频的4次控制运行均值，用于阶段和kernel归因；不是C16客户端RT。Profiler计时与上面的吞吐基准分开。

| CUDA event阶段（ms/批） | 修改前 | 修改后 |
|---|---:|---:|
| nvdec_and_gpu_preprocess | 396.156 | 393.438 |
| vision_forward | 2293.441 | 1972.448 |
| frame_token_assembly | 10.454 | 10.341 |
| multimodal_batch | 2705.365 | 2385.598 |

| ViT kernel类别（27层合计ms/批） | 修改前 | 修改后 |
|---|---:|---:|
| attention_layout | 186.031 | 0.000 |
| fa4 | 821.606 | 594.292 |
| layer_norm | 208.328 | 244.092 |
| merger | 20.101 | 21.424 |
| mlp_fc1 | 201.343 | 206.029 |
| mlp_fc2 | 235.944 | 254.027 |
| mlp_gelu | 153.347 | 182.303 |
| output_projection | 64.623 | 68.581 |
| patch_embedding | 2.970 | 2.973 |
| position_interpolation | 1.885 | 1.886 |
| qkv_projection | 164.289 | 169.245 |
| residual | 69.926 | 70.760 |
| rope | 158.276 | 153.056 |
| rotary_metadata | 0.324 | 0.325 |
| vision_misc | 1.284 | 1.284 |

本次Nsight窗口未覆盖前两个C16 batch（0、1），其kernel指标记为缺失；全部64个请求仍有host/CUDA event记录，后续10个batch及4个固定B16控制运行完整覆盖。
CUDA event包含该stream的等待；kernel sum为核执行时长求和；interval union用于合并重叠。上述类别按最内层NVTX及CUDA correlationId互斥归属，不将子阶段再次加入外层总耗时。
修改后ViT CUDA event / GPU activity envelope / kernel sum / activity union / copy sum / uncovered gap（ms）：1972.448 / 1972.436 / 1970.276 / 1971.443 / 1.201 / 0.993。

## 配置与输入

GPU物理0：NVIDIA L20D，SM103，CUDA13，PyTorch2.11.0+cu130；checkpoint=/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8。vit_concurrency=16、gpu_max_batch_size=16、gpu_batch_wait_ms=10、decode_workers=32、FA4、NVDEC、graph关闭、所有embedding/hash/URL cache关闭。
沿用历史有效输入：fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180，同时固定width=1280、height=704。46采样帧，grid=[23,44,80]；每视频80,960 patches、20,240视觉tokens，组装20,424行embedding，shape=[20424,4096]。
视频SHA256：d8b10b3fd5669c6d4db8cab1d0b2cd970d4e9db09a2ce09059b3a3e3a76987b0。

## 限制与原始记录

本次基准、诊断、采样和控制进程均已退出，无本次遗留进程。

已有NVDEC pool关闭超过10秒的问题仍存在：测试脚本先保存结果，再退出进程释放资源。本次没有修改生产关闭流程。
吞吐基准完成后，最终复查仅修正部分RoPE时未旋转通道的copy目标上界；Qwen3.5本视频始终全72通道旋转，该分支不执行。随后重跑11项单测，并以最终代码完成真实权重/视频的诊断和输出一致性检查。

- [本次原始目录](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr)
- [单测结果](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/unit-result.json)
- [C16原始请求](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/baseline/load/requests.jsonl)
- [C16汇总](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/baseline/summary.json)
- [时间线](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/engine.nsys-rep)
- [阶段汇总](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/stage-summary.json)
- [每层明细](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/diagnostic/per-layer.csv)
- [路径检查](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/path-validation.json)
- [最终改动patch](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/changes.patch)
- [源码审计](/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/source-audit.json)
- [修改前记录](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-original-vllm-compare-20260917-_conoz1l/vllm_port)
- [进程退出检查](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-dense-no-layout-20260917-dz1vuedr/cleanup.json)
