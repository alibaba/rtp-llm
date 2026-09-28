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
