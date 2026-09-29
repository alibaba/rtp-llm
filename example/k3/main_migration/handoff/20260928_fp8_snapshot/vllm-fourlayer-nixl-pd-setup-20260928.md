# vLLM 四层 K3 NIXL PD 对照准备（2026-09-28）

固定镜像为 `mirrors-ssl.aliyuncs.com/vllm/vllm-openai@sha256:dfaab3570be5b1f66c21e60c60f1616ad3a0143f9899b8738257004f289979fd`，源码提交 `3df4ae153eb385e27b52f26c81f8edb9e20b9984`、版本 `0.29.1rc1.dev452`。官方同提交 NIXL toy proxy 保存为 `vllm-toy-proxy-3df4ae153.py`；没有改写跨机传输代码。当前选择 111 Prefill、112 Decode，均由个人 UID 19357313 启动独立任务容器，并直接从 3FS 读取相同四层 checkpoint。

四层 checkpoint 的 `text_config.quantization_config` 声明 routed MoE 为 `mxfp4-pack-quantized`；强制全局 `--quantization fp8` 会被 vLLM 拒绝。当前保留 checkpoint 原量化格式，KV cache 指定普通 `fp8_e4m3`，在实际 trace 中再核对 MLA query/KV 是否走 FP8。不能把所有模型权重统称为 FP8。

固定版本在这套模型上的启动约束由先前失败日志确认：

- `config.json` 的 `max_position_embeddings=8192`，要运行 64K 必须对本任务容器设置 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`；仍需用 64K 请求核对数值。
- E4M3 KV cache 要求 `attention-config.use_prefill_query_quantization=true`。自动选择的 `FLASH_ATTN` MLA Prefill 不支持这条 FP8 query 路径，因此显式选择 `TOKENSPEED_MLA`。两端日志已确认选中 TokenSpeed；原失败发生在权重读取前。
- NIXL 的 KDA/Mamba conv 三次读取状态要求 `VLLM_SSM_CONV_STATE_LAYOUT=DS`；上一轮在权重加载后因缺少该值退出。该值已仅加到任务容器。

111/112 的 vLLM safetensors rank 0 均从 3FS 读完七个 shard，初次约 15.5/15.7 秒，再次约 16.39/15.68 秒。日志提醒 3FS FUSE 未被自动识别为网络文件系统，因此 vLLM 没有自动预取；这几次加载未显示明显读阻塞，暂不改变 3FS 客户端并发。`/proc/<pid>/io` 的 `rchar`/`read_bytes` 对 mmap/FUSE 路径不能可靠表示真实吞吐，不据此下结论。两端服务总初始化明显长于权重 shard 阶段，应分开记录。

固定版本默认在 65,536 token 预算上做 FlashInfer 自动调优，111/112 两端超过五分钟未完成。任务容器的第一次启动日志分别保存在 `producer-autotune-65k-stall.log`、`consumer-autotune-65k-stall.log`。为先核对 PD 路径，当前脚本暂用 `--no-enable-flashinfer-autotune`；后续性能对照必须标注这个差异，不能称作充分调优的 vLLM 上限。

第一次成功启动的短请求通过官方 proxy 返回 HTTP 200；Decode 侧 8 个 rank 均记录 NIXL compatibility passed，传输指标记录 8 次成功传输。原始响应在 `vllm-pd-3df4-small-nixl-diagnostic-20260928.json`，两端日志在 `producer-nixl-diagnostic-non-nccl.log`、`consumer-nixl-diagnostic-non-nccl.log`。这是四层、4 token 的链路诊断，不能据此判断回答准确率。那轮日志还显示 TP 启用 `FLASHINFER` 和 `SYMM_MEM` AllReduce，不符合固定 BF16 NCCL 的比较口径。

当前脚本 `run_vllm_3df4_fourlayer_pd.sh` 已关闭这两种通信后端和 `allreduce_rms` 融合。新启动的 111/112 日志均确认 TP 后端只有 `PYNCCL`、`allreduce_rms` 不再列为启用融合。两端七个 shard 的 3FS 加载分别用 15.54/14.36 秒。固定 65,536 token + 8 输出 token 的官方 proxy 请求成功，Decode 八 rank NIXL 握手通过，KV Transfer 统计随请求增加，平均每次约 38.61 MB。四层输出不能作为语义正确性证据，原始响应见 `vllm-pd-3df4-64k-first-diagnostic-20260928.json`。

两次共 28 条代表性 64K 预热请求的 HTTP 完整耗时波动明显。第一次 14 条未通过脚本末三次 ±5% 的 HTTP 门槛，因此没有开始 profiler；第二次保留异常标记并在 14 条预热后捕获三次全 rank Prefill。GPU 注释区间的关键 rank 耗时 118.040、112.159、113.652 ms，相对中位数最大偏差 3.861%。八个 rank 原始 trace、请求记录、SHA256 与内核族汇总位于 `timeline-64k-vllm-3df4-fourlayer-nixl-pd-pynccl-profile-20260928/`。该区间标记 `execute_context_1(65535)_generation_0(0)`，即 Prefill 计算 65,535 个 token，把最后一个 token 留给 Decode；与 RTP 对照解释时要注明这一 token 的差别。

八个 rank 的远端与 115 本地 trace SHA256 逐文件一致。采集完成后已保存两端完整启动与运行日志，并停止本任务的 producer、consumer 和 proxy 容器，释放 111/112 GPU。再测时须重做独占检查和同路径预热。

## 114/115 复测准备与模块归因

9 月 29 日，114、115 拉取了相同 digest 的固定 vLLM 镜像；任务专用启动脚本为 `run_vllm_3df4_fourlayer_pd_114115.sh`。启动前两端读四层 target 的小型配置文件均进入 3FS 内核等待，因此没有启动服务。存储侧证据见 `3fs-mtp-first-read-stall-20260929.md`，恢复后要先复查读取，再运行脚本。

现有全 rank vLLM trace 的 `user_annotation` 只有请求级 `execute_context`，CPU `aten::linear` 等记录没有调用栈；虽然能按 CUDA correlation 找到 GEMM 的发射事件，但不能把 NVJet GEMM 可靠归到某一层的具体投影。114/115 脚本新增 `K3_VLLM_PROFILE_WITH_STACK=true` 和 `K3_VLLM_RUN_ID=<独立标识>`，供后续单独采一次带栈诊断 trace；默认仍关闭调用栈，沿用原来的计时配置。带栈 trace 是否足以归因须实际检查，采集开销可能影响耗时，不能直接拿它与默认设置的正式性能 trace 比较。
