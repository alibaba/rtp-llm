# MLA K 合并核的 12-head 候选

固定 `feat/k3_dev` 四层 FP8 trace 的 MLA pipeline 中，`Tensor.copy` 累计约 0.097 ms；集成版 `d733278ac` 的 `attention.mla.core` 中约 0.307 ms。两个范围的融合边界不同，这些累计值只用于定位候选，不能直接证明整体耗时差异。源码显示集成版的 `concat_and_cast_mha_k_triton` 要求 head 数为 2 的幂；四层 K3 的 12 个 head 因而走两次 PyTorch 切片拷贝。本候选让原核按 2 的幂向上取整 head 范围，并对越界 head 的读写加掩码；K/V 数值布局和通信路径不变。

在 110 的个人 `lhc_GPU` 中、GPU 1 上运行 Bazel `--config=cuda13 --config=sm10x` 用例，源码和构建缓存都在个人 ext4 数据盘。测试输入为 257 token、12 head、128 维 NoPE、64 维 RoPE，NoPE 是从 256 维 KV 投影切出的非连续 BF16 view；与独立的 PyTorch 拼接结果做逐元素精确比较。

- 旧核（文件 SHA256 `1a53157ce9df809f404a5e91ce868331817d2b77a5d61576ee2e273dae6a50de`）的有效红灯在归档内的 `mla-kmerge-red-bazel-20260929-r2.log`：用例实际运行，Triton 在 `tl.arange(0, 12)` 报 `arange's range must be a power of 2`。构建 21,904 个动作，测试 1/1 失败，符合预期。
- 新核（文件 SHA256 `b4f9c4818e6c9e130b148d0aae658291d417a0f0d494cf8163eecc2b753e3f11`）的绿灯在归档内的 `mla-kmerge-green-bazel-20260929-r2.log`：相同用例 1/1 通过，Bazel 退出码 0。新增 `CC=/usr/bin/gcc` 只供 Triton 运行时编译使用；此前一次因容器中不存在 Bazel 传入的 GCC 路径而失败，不能当作代码红灯。

两个原始日志保存在 `raw-logs.tar.gz`，归档校验值见 `raw-logs.sha256`。

用相同的 65,536-token、12-head、BF16 非连续 KV 投影 view，预先分配输出，在 110 的 GPU 1 上做两次拷贝与单次 Triton 合并核的 A/B/B/A 交错测量。两种输出先逐元素精确比对。每组先完成一次惰性编译、至少 10 次同路径预热，末三次 CUDA event 时间落在中位数 ±5% 内；每组再采 30 次。原始样本在 `benchmark-64k.json`，运行日志在 `benchmark-64k-raw.tar.gz`，归档哈希在 `benchmark-64k-raw.sha256`。

| 64K 局部算子 | 第一组中位数 | 第二组中位数 |
|---|---:|---:|
| 两次 PyTorch 拷贝 | 0.313296 ms | 0.313344 ms |
| 单次 Triton 合并 | 0.096768 ms | 0.096400 ms |

这个形状的局部 GPU 时间缩短约 0.217 ms。完整 110–115 探测记录在 `host-selection-benchmark.json`：110 和 113 都无外部 GPU 计算，数值排序选了 113；因为候选已在 110 独立编译，随后对 110 的 GPU 1 单独复查为独占可用，见 `host-selection-benchmark-110.json`。本次只测算子，没有加载模型权重或改动 3FS 并发。

这仍不是四层 FP8+MTP PD flow 或 64K timeline 的结果，不能据此将候选并入性能锚点。下一步需在双机服务里核对新核实际被调用、答案链路正确，再比较全 rank target Prefill 关键路径。

## 111/112 双机候选编译与启动前检查

候选源码固定在 `3d7fe34a7f025fd06ebc731fa0b39323bd5484c8`。111 和 112 分别在个人 ext4 数据盘上的独立工作树、Bazel 输出目录编译了 `//rtp_llm:rtp_llm_server`；容器镜像相同，容器内用户均为 `luohaocheng.lhc`，命令包含 `--config=cuda13 --config=sm10x`。两端各完成 23,714 个动作，Bazel 均报告 `Build completed successfully`。原始日志及哈希保存在 `build-logs-111112.tar.gz` 和 `build-logs-111112.sha256`。

同一套启动参数的 `--print-config` 显示 target 从 `/mnt/hf3fs/3fs/models/kimi/kimi-k3-4layers` 直接读取，`LOAD_METHOD=fastsafetensors`；MTP 视图在各自主机个人数据盘，shard 指向已核实的 3FS 权重。启动前 guard 结果见四份 `weight-*.txt`：每端 target 为 7 个 shard、16,402 个 tensor、54.47 GiB；MTP 为 9 个 shard、5,404 个 tensor、20.00 GiB；索引和 Safetensors header 均通过。111/112 各 8 个 RDMA bond 都是 `ACTIVE`，两端互 ping 无丢包。3FS 挂载可读。这里仅证明启动前条件，服务实际加载日志仍需在切换后核查。

`host-selection-build-complete.json` 是旧 `d733` 服务仍占用 111/112 显存时的只读快照：110 可用，113 显存不足，114/115 没有运行中、同镜像的个人容器，因此当时没有可直接再加载一套 TP8/EP8 PD 的双机组合。旧服务属于本任务；完成证据留存并确认不再承接请求后，才能按准确进程组停止它们，重新执行双机选择。

## 四层 FP8+MTP 双机 flow

停止旧服务后，111/112 的本任务进程组分别启动固定候选提交 `3d7fe34a7`。两端 8 个 rank 的实际日志均显示 `finally choose load method: fastsafetensors`，guard 的 `verify-log` 全部通过；111 Prefill 和 112 Decode 的 `/health` 均返回 200。启动参数为 target FP8 per-block、FP8 KV cache、draft BF16、NCCL TP8/EP8，Decode 启用 CUDA Graph；Prefill MLA 日志显示 TokenSpeed 路径。`host-selection-flow.json` 是请求前的双机占用快照：只豁免这两个已核实的本任务服务进程组，没有外部 GPU 计算进程。

11 条 flow 的 runner 全部通过、没有跳过；独立复核结果在 `flow-11-independent-audit.json`。复核逐条检查 HTTP 200、非空 UTF-8 响应、PD 状态交接长度、Decode 路由、Native MTP draft 实际轮次及 300 秒上限。最慢的首个 65,537-token 请求为 71.18 秒，其余请求均在 17 秒内；11 条响应没有 Unicode replacement character。含逐请求原始响应和 token fixture 的归档为 `flow-11-raw.tar.gz`，校验值在 `flow-11-raw.sha256`。四层权重是随机裁剪模型，这些结果只证明运行链路和格式，不能作为完整模型语义准确性的证据。

## 64K 热态 Prefill 对照

`host-selection-timeline.json` 证明测量时 111/112 的 GPU 只有本任务服务进程。三组 RTP 请求使用相同的 65,536 个输入 token（SHA256 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`）、四层权重、FP8 target、Native MTP、TP8/EP8、PD 路径、8 个输出 token，以及不复用前缀的 10 次预热和 16 次正式请求。候选请求的独立复核见 `independent-request-audit.json`，10 次预热和 16 次请求均通过，未见替换字符。八个 rank 的原始 trace 在 `allrank-traces.tar.gz`；请求原文和响应在 `requests-64k-kmerge-3d7.tar.gz`，哈希在 `timeline-archives.sha256`。`module-target.json` 和 `module-draft.json` 保存逐 rank、逐请求的 CUDA launch 到模块归属；`startup-111.tar.gz`、`startup-112.tar.gz` 保存本轮服务全 rank 加载日志及预检，哈希在 `startup-archives.sha256`。

| 四层版本 | 全 rank 匹配请求数 | target Prefill 中位数 | draft Prefill 中位数 | 完整 Prefill GPU span 中位数 |
|---|---:|---:|---:|---:|
| 集成版 `d733278ac` | 7 | 72.617 ms | 63.480 ms | 135.722 ms |
| 本候选 `3d7fe34a7` | 6 | 72.255 ms | 62.956 ms | 135.148 ms |
| 固定 `feat/k3_dev` `a9bf762e8` | 7 | 71.556 ms | 62.564 ms | 135.266 ms |

数值来自同一版全 rank GPU 时间关联脚本，HTTP 往返不用于 Prefill 结论；每组可匹配的请求数不同，因此 0.1 ms 量级的差异需要复测。集成版 L3 `attention.mla.core` 的累计核时间从 6.359 ms 降到 6.084 ms；旧版 `direct_copy_kernel_cuda` 在该范围的最慢 rank 中位数为 0.307 ms，新版 `concat_and_cast_mha_k_kernel` 为 0.078 ms。固定 feat 的 `native_mla_and_cache_pipeline` 为 6.087 ms，但范围边界不同，只能作为定位线索。候选 target 仍比固定 feat 慢约 0.70 ms，尚未满足性能锚点条件。

固定 feat 的 `kimi_k3.all_gather_gemm.fp8_fused` 将 E4M3 激活值和 UE8M0 scale 作为通信数据交给 PyTorch symmetric-memory pipeline；`kimi_k3.gemm_reduce_scatter.fp8_fused` 调用 DeepGEMM `fp8_gemm_rs_nt` 自带的归约/散发 workspace。这两个原实现的通信路径不符合本任务的 BF16 NCCL 约束，不能直接迁入。其 fused scope 的累计时间较短，但与集成版分开的 AllGather、投影和 ReduceScatter 范围也不是同一测量边界。下一步应在 BF16 NCCL 约束下定位剩余约 0.70 ms，再做针对性候选，而不是整包迁入 feat 的通信实现。

111 的 target loader 选择 FastSafetensors 后约 25 秒进入 MTP loader，112 约 41 秒；这段时间同时包含权重处理，不能当作纯 3FS 吞吐。112 服务总启动约 342 秒，后续还进行了模型初始化和 Decode CUDA Graph 相关工作。实际使用的是本任务进程的 64 线程 3FS 预读辅助库，没有改动机器或集群级 3FS 配置；本轮没有观察到需要先调整并发才能继续测试的权重读取阻塞。
