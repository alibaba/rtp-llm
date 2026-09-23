# Qwen3.5：frame token 核对及 8/16 并发耗时

## 结论

当前已注入 frame timestamp 和 vision_start/vision_end token。约 24K 指整条请求的 LLM 输入长度；ViT 的纯视觉输出为 20,240 token，包含帧标记的多模态 embedding 是 20,424 行。

本次范围为视频文件进入 Python MMProcessEngine 到 GPU embedding 可消费，不启动/调用 gRPC，不计算网络或 embedding 序列化，不执行 LLM prefill/decode。吞吐主要由 ViT 内核耗时限制；请求 RT 还包含批前准备与排队，随调度实际形成的 batch 改变。

## Token 组成

| 项目 | 数量 |
|---|---:|
| 46 帧按 temporal_patch_size=2 合成的时间组 | 23 |
| 纯视觉 token = 23 × 22 × 40 | 20,240 |
| 时间戳文本，每组 6 token | 138 |
| 每组 vision_start / vision_end 各 1 | 46 |
| ViT 服务返回 embedding 行数 | 20,424 |
| 其余文字、聊天模板和外层标记 | 4,177 |
| 最终完整请求输入，tokenizer 计算 | 24,601 |

每组布局：<timestamp seconds><|vision_start|>[880 visual tokens]<|vision_end|>。帧标记使用实际语言词表权重，不是额外图像特征。当前 renderer 在展开前得到 4,178 token、包含 1 个 video_pad；用 20,424 行多模态 span 替换该 pad 后为 24,601。完整请求计数是 CPU tokenizer/模板核对，本次未运行 LLM 取 usage。
计数原始证据：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-frame-token-check-_ycs85st/token-count.json；当前 preprocess 和 _assemble_video 实现及实测 output shape 一致。

## 无采集时的引擎性能

GPU0 / L20D，真实 Qwen3.5-397B-A17B-FP8 ViT 权重，BF16；46 帧，704×1280，grid=[23,44,80]，80,960 patches。vit_concurrency=64、gpu_max_batch_size=64、gpu_max_batch_images=256、gpu_batch_wait_ms=10、NVDEC workers=32；全部 embedding/hash/URL cache 关闭，graph 关闭。

每档 20 秒持续发请求 × 3 轮，排空计入分母。取中位 QPS 完整轮；RT 等待生产 stream 的输出 event 完成。初次 C8 三轮有其他 GPU 的背景任务，保留证据但不作为最终受控对照；下表 C8 使用补测，C16 使用初测的无外卡忙碌可见样本。

| 并发 | 三轮 QPS | 中位轮 QPS | Mean ms | P50 ms | P95 ms | P99 ms |
|---:|---|---:|---:|---:|---:|---:|
| 8 | 5.808/5.674/5.645 | 5.674 | 1396.256 | 1404.519 | 1441.764 | 1442.983 |
| 16 | 5.212/5.791/5.865 | 5.791 | 2722.044 | 2753.162 | 2794.531 | 2796.970 |

## 请求 RT 的实际组成

用 request ID 关联 query/access 日志，再将完成时刻对齐实际 forward 结束。逐批验证匹配请求数恰好等于实际 media_count。日志为毫秒精度；剩余项保留 invoke 入口、日志时间精度、返回及 CUDA event 完成等待等，不能全部解释为某一算子。

| 并发 | 平均 RT ms | 批前读取/元数据/排队 ms | 本批处理 ms | 返回和未归因剩余 ms | 实际组批 |
|---:|---:|---:|---:|---:|---|
| 8 | 1396.26 | 221.99 | 1168.12 | 6.15 | {'8': 6, '7': 9, '1': 9} |
| 16 | 2722.04 | 642.20 | 2066.41 | 13.43 | {'14': 8, '2': 9} |

“本批处理”每个请求等待的是整个 packed batch 的完成时间，不能除以 batch size 当作单请求 RT。批前时间目前只能严谨拆到“读取/元数据准备+排队”，未在无采集轮把两者单独切开。诊断中读取/元数据均值约 41–65 ms，但 profiler/组批不同，不把该数直接从上表相减。

## 单批内部耗时：独立 Nsight 诊断

诊断开启 NVTX/CUDA events 和 Nsight Systems，32 条 C8 +64 条 C16 请求。其实际 batch 是 C8 的 4×B8、C16 的 4×B1 +4×B15，与基准轮实际组批不同；以下用于模块归因，不替代基准 RT/QPS。

| 批大小 | 整批处理 ms | NVDEC + GPU 预处理 elapsed ms | ViT elapsed ms | 帧标记组装 elapsed ms |
|---:|---:|---:|---:|---:|
| 8 | 1370.81 | 251.27 | 1108.54 | 5.57 |
| 15 | 2605.59 | 472.20 | 2114.61 | 9.99 |

典型 B8/B15 约 81% 批耗时落在 ViT、17–18% 在 NVDEC 与 GPU 预处理。CUDA event elapsed 包含该 stream 区间中的等待，不等同于纯 CUDA kernel 时间。CPU/GPU 或并行解码线程有重叠，不累加各 worker 时间。

全部 batch 时长和冷/稳态证据见 stage-summary.json。第一个 B15 出现 1.804 秒的预处理长尾，整批 4.152 秒；原始样本保留，不混入稳态 B15 代表值。后续 B15 整批分别 2.611/2.606/2.573 秒、ViT 2.116/2.115/2.111 秒，两次相邻变化均小于 5%，选择中间批次 9。B8 四个 pass 全部保留（ViT 1.108/1.109/1.190/1.162 秒），批次 1 仅是显式示例，不声称四个 pass 满足两次连续 5% 收敛。

NVDEC worker 覆盖解码、颜色转换及对应 CUDA stream 输出就绪等待。B15 后三批解码 worker 的并发活动窗口约 408–443 ms，GPU 预处理 CUDA kernel 合计约 85 ms；二者有流水重叠。NVDEC 专用硬件本身不完整体现在 CUDA kernel 表，不能把其少量 CUDA kernel 时间误当全部解码耗时。

## ViT 内部：代表性 B15 的 GPU kernel 耗时

| 模块 | GPU kernel 累计 ms | 占 ViT GPU 区间 |
|---|---:|---:|
| MLP | 604.36 | 28.67% |
| FA4 | 556.11 | 26.38% |
| Attention QKV/输出投影、RoPE | 432.36 | 20.51% |
| LayerNorm、残差等 | 294.19 | 13.96% |
| Patch embedding | 189.99 | 9.01% |
| Merger | 20.07 | 0.95% |
| 位置/网格等其他 ViT 操作 | 8.86 | 0.42% |

该 ViT GPU 区间 2107.833 ms，kernel 活动并集 2105.939 ms，未覆盖空档 1.893 ms。CPU launch 空档不是此区间的主要损失；这只描述 kernel 连续执行程度，不是 Tensor Core/MTU 利用率，本次未重采 NCU 计数器。

每批 27 个显式 layer NVTX，且 27 个 FA4 kernel；12 个 forward 共 324 个 FA4，所有 30,826 个 CUDA kernel 都通过 correlationId 关联到发起 API。层边界基于 module hook 和真实模型配置，无周期猜测。

| 代表区间 | Nsight 相对开始秒 | 结束秒 |
|---|---:|---:|
| B15 ViT GPU | 14.217794984 | 16.325627526 |
| B15 encoder layer 12 | 15.250899819 | 15.321470170 |

全部 forward、逐层 wall/kernel-sum/kernel-union/gap、代表层 kernel 顺序和类别见 nsys-analysis.json / engine.sqlite。CUDA memcpy 已单独保存，未混入 kernel 时间。

## 优化方向及范围限制

按可回收时间优先检查 MLP、Attention 投影和 FA4，而不是只盯 FA4。归一化/残差及 patch embedding 也有明显份额。帧 token 组装约几毫秒，占整批不到 0.5%，不是这次吞吐的主要限制。

当前 GPU scheduler 在执行 batch 内先做 NVDEC/预处理再调用 ViT；可另行评估下一批预处理与当前批 ViT 的重叠。若完全隐藏约 18% 预处理且没有资源争用，理想上界约 1/(1-0.18)=1.22 倍；这只是按阶段份额计算的上界，未实现或验证收益。单纯提高请求并发或 batch 上限，在 ViT 每视频耗时近似线性时主要增加等待和显存。

本次完整性范围限于用户指定的无 gRPC 多模态引擎：真实文件到可用 embedding、正确性、无采集压测、队列/批次核对、独立 Nsys 和层归因已覆盖；LLM 完整请求、实际 ViT→LLM 传输、NCU/MTU 不在本次范围，不据此推断全链路或计算单元饱和。

## 审计与产物

- /home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8-idle-recheck-20260917-xiqhehyb：计时请求 360，错误 0；并发全输出校验 8 次；源代码一致=True，实际导入源文件一致=True，decoder_close_timeout=False。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8c16-stages-20260917-abj8dzw2：计时请求 734，错误 0；并发全输出校验 24 次；源代码一致=True，实际导入源文件一致=True，decoder_close_timeout=True。
- 两个可区分输入分别建立参考，输出有限、逐元素与各自串行参考完全一致；计时阶段检查 shape/dtype/device/生产 feature hash；embedding/hash cache 无命中、无 in-flight 去重。
- 已有生产文件改动未修改。本次只添加 .t 测试/分析程序与本文档，未 commit/push。
- NVDEC 线程池退出仍有阻塞。测试 harness 在保存所有结果后等待最多 10 秒、记录超时，再由进程退出释放资源；这不是对生产关闭问题的修复。各次 cleanup.json 记录资源释放。
- 最终选择、全轮 QPS 与资源可见性：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8c16-stages-20260917-abj8dzw2/final-selected.json；其他 GPU 忙碌的初测 C8 保留在 resource-all-phases.json。
- 原始请求、阶段与队列预算：各 run 的 load/requests.jsonl、forwards.json、baseline-latency-budget.json；诊断 stage-events.json、stage-summary.json。
- Nsight：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8c16-stages-20260917-abj8dzw2/engine.nsys-rep；SQLite：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8c16-stages-20260917-abj8dzw2/engine.sqlite；关联解析：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-c8c16-stages-20260917-abj8dzw2/nsys-analysis.json。
- Token 验证：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-frame-token-check-_ycs85st/token-count.json。

最后清理确认：所有本次引擎、采样器和 Nsight 进程均已退出；GPU0 恢复 117 MiB、0% 利用率，无 compute process。每个最终中位轮均有一次全卡采样不可用，其他可见样本未见外卡忙碌；保留该采样限制。
