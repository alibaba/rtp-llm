# ViT 指标口径与上报位置

参考外源 `upstream/feat/minimax_m3_0802`（`d13f803604c568ac3f5597cae42ffb702d3731ca`）。通用指标使用 M3 的名称；Qwen3.5 NVDEC 和 exact-grid CUDA Graph 的指标接在本分支实际执行路径上。

## 预处理和 embedding

| 指标 | 单位与口径 |
| --- | --- |
| `py_rtp_vit_preprocess_rt` | ms，单次实际 preprocess 调用的总耗时，不含进程池排队 |
| `py_rtp_vit_download_rt` | ms，preprocess 调用内的媒体读取耗时之和；覆盖 HTTP、本地文件、data URL；URL 数据缓存命中为 0 |
| `py_rtp_vit_preprocess_other_rt` | ms，preprocess 总耗时减去 download，非负；两者之和等于 preprocess_rt |
| `py_rtp_vit_preprocess_queue_size` | 已接收且尚未完成的进程池任务数，包含正在运行的任务；本地执行器初始化为 0 |
| `py_rtp_vit_image_count` | 每次逻辑请求中的 IMAGE / DEFAULT 输入数，保持 M3 口径，不把视频帧算成图片 |
| `py_rtp_vit_video_frame_count` | 每段视频实际采样帧数；重复采样位置计数，temporal padding 的补帧不计数 |
| `py_rtp_vit_embedding_length` | 成功结果的输出 token 数；同步调用和异步结果获取分别只报一次；包括缓存结果，hashes-only 用对应 feature hash 数计数 |
| `py_rtp_vit_embedding_rt` | ms，单次 GPU batch 的 batched_embedding 调用耗时，与 M3 一致 |
| `py_rtp_vit_embedding_forward_rt` | ms，保留本分支已有的同一 forward 耗时指标 |
| `py_rtp_vit_embedding_batch_rt` | ms，每次逻辑 submit_and_wait 的总耗时，包含 admission、GPU 输入准备、排队及 forward 等待 |
| `py_rtp_vit_embedding_batch_size` | 单次 forward 合并的 scheduler chunk 数；一个逻辑请求可能被拆成多个 chunk，一个 chunk 可以包含多个 media |
| `py_rtp_vit_embedding_queue_size` | 等待 forward 的 ready chunk 数，包含暂存 pending chunk；不包含正在 admission / GPU 输入准备的请求 |
| `py_rtp_vit_embedding_queue_wait_rt` | ms，从 ready chunk 入队到 forward 的等待，包含收集 batch 的窗口；每个 chunk 一次 |
| `py_rtp_vit_process_pool_restart_qps` | 实际成功重建预处理进程池后累加 1；初次创建及创建失败不计数 |

`embedding_rt` / `forward_rt` 沿用调用的 host wall-time 边界，不额外同步 CUDA，因此不能直接当成纯 GPU kernel 时间。Qwen3.5 的 NVDEC/GPU 输入准备发生在进程池 preprocess 之后、ready queue 之前，耗时由下述阶段指标和 embedding_batch_rt 覆盖，不包含在 preprocess_other_rt 中。

`py_rtp_vit_embedding_length` 是结果获取侧 token 数；已有 `rtp_llm_vit_output_token_count` 是 transport 成功输出后的 token 数，两者边界不同，应按场景使用，不能相加。

## Qwen3.5 视频阶段

GPU decode / resize / processor 路径补齐以下原有指标名，使用固定低基数标签 `model=qwen35, mm_type=video, backend=nvdec`，另用 `timing` 区分计时方法。fetch 沿用已有 `model=qwen35, mm_type=video` 标签。

| 指标 | 单位与 GPU 路径边界 |
| --- | --- |
| `rtp_llm_vit_image_decode_rt_us` | us，NVDEC 已完成的解码/采样 wall time，加上 NV12 转 RGB 的 CUDA event span |
| `rtp_llm_vit_image_resize_rt_us` | us，resize 的 CUDA event span |
| `rtp_llm_vit_image_processor_rt_us` | us，rescale/normalize/patch 排列的 CUDA event span |
| `rtp_llm_vit_image_fetch_rt_us` | us，原有媒体读取调用耗时，继续在 preprocess 侧上报 |
| `rtp_llm_vit_resized_pixel_count` | 采样帧数 × resize 后高 × 宽；补齐 NVDEC 路径，temporal padding 不计入 |

缓存结果不会重新执行视频处理，因此不会新增这些阶段样本。CUDA event 样本由仅用于监控的守护线程读取；请求线程不为计时增加同步。后台队列有界，仅持有 event 和指标数值，不保留视频或 embedding tensor。队列满时不阻塞推理，会丢弃指标样本并限频告警，使用 `py_rtp_vit_preprocess_metric_dropped_qps` 记录丢样。应检查该指标，避免把缺样当成无流量。

## CUDA Graph

使用 M3 的 `py_rtp_vit_cuda_graph_{hit,miss,capture,fallback}_qps`，同时保留本分支带 `event/model` 标签的 `vit_graph_event_qps`。

- 显式关闭 Graph、max_entries=0、多 grid、非 CUDA、超过 patch 上限、正在其他 capture、后端不支持或 capture 失败，均计 fallback。
- 首次遇到新形状的 eager 预热计 miss；capture 和后续 replay 分别计 capture 和 hit。
- 本分支采用 exact-grid，不做 padding，因此成功 capture/hit 时 `py_rtp_vit_cuda_graph_padding_ratio` 为 0。
- counters 不是互斥分类：一次请求可能同时计 miss 和 capture/fallback，不应把全部 counter 相加当作请求量。

## 部署与排查

Python 枚举在 kmonitor 初始化时统一注册，无需额外修改内源架构 select 或 requirements。修改需进入新安装包/镜像并重启对应进程才会在线生效；代码工作区变化不会改变现有线上容器。

ViT 队列、预处理、forward、Graph 指标在实际执行 ViT 的 worker 进程上查看。Prefill 的 `rtp_llm_vit_rpc_client_*` 在调用端上报；RDMA 的 response_bytes 是 receipt protobuf 大小，不是 embedding 数据大小。未执行的条件路径不会凭空产生样本，例如没有 Graph capture 就不会有 capture QPS。

## 本次验证（2026-09-22）

- 通用预处理、下载、缓存、scheduler、Graph 和已有 ViT 指标联合测试：90 项，88 通过，2 项 CUDA 测试跳过。
- GPU video 模块的 CPU/fake-event 测试：30 项，25 通过，5 项 CUDA 测试跳过。覆盖后台上报、最后空闲请求、监控阻塞时饱和队列不阻塞提交、异常隔离、帧数及像素数。
- 合计 113 项通过、7 项 GPU 测试跳过；没有启动真实模型服务，也没有在线验证监控平台收到样本。
- 12 个本次修改的 Python 文件通过 Black 检查，git diff --check 通过。

日志在父仓库 `.t/vit-metrics-complete-d5ag28a9/integration-tests.log` 和 `gpu-video-tests.log`。
