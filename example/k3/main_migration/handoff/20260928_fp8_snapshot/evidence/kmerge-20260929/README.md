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

为检查 BF16 NCCL AllGather 是否有简单的环境配置收益，我在独占的 110 上用 `bench_bf16_nccl_allgather.py` 单独测了 TP8、每 rank 8,192×7,168 的 BF16 输入；这与四层 64K attention 输入形状相同。每组单独建进程，先做一次初始化，再完成 20–40 次预热，末三次所有 rank 均在各自中位数 ±5% 内，随后测 30 次。输出的 rank 标记也逐个核对。结果是默认 NCCL 配置 1.410 ms、强制 `NCCL_ALGO=NVLS` 1.626 ms、强制 `NCCL_PROTO=LL128` 1.486 ms，均为每次最慢 rank 的 CUDA event 中位数。原始逐 rank 数据、110 的独占选择快照及运行日志在本目录的 `bf16-nccl-ag-*` 文件中。这是同形状通信筛选，不包含模型计算或 PD 链路；所测两种覆盖配置都没有收益，因此本轮没有改服务的 NCCL 参数。

## Paged convolution 元数据候选

我用 `analyze_prefill_gpu_idle.py` 对上面的两组八 rank trace 逐个匹配 target 请求，并把 CUDA launch 关联到 GPU kernel，再求每个 rank 的 kernel 区间并集。候选版最慢 rank 的 GPU 空隙中位数为 4.880 ms，固定 feat 为 2.369 ms；相应的 GPU 忙碌并集为 67.372 ms 和 69.018 ms。这些数值只描述被 target CPU scope 发射的 kernel，不能把空隙直接当作 CPU 等待或某个算子耗时。逐请求、逐 rank 数据在 `target-gpu-idle.json.gz` 和 `feat-a9bf-target-gpu-idle.json.gz`。

候选版的稳定大空隙集中在 embedding 前及首层 AttnRes 前。trace 中该阶段有一次 `aten::copy_`，输入为 8,192 个 `int32`，其中一个匹配请求耗时 2.460 ms。源码的 `prepare_causal_conv1d_metadata` 正好为 65,536 token 生成 8,192 项传统 convolution 索引；当前 FP8 cuLA paged convolution 的对齐无前缀路径读取的是另一份 1,024 项 paged 索引。因此我增加了一个保守的条件：所有 KDA 层都走 paged convolution、每层有 cache 且每层 prefix 按 64 token 对齐时，不再准备传统索引；任一条件不满足仍保留原来的 fallback。这个源码与 trace 的对应关系是性能假设，需由重启后的双机 flow 和热态 timeline 确认收益。

111 的 CPU Bazel 测试先按旧代码运行，3 条中对齐 paged 路径的 1 条按预期失败；修改后 3 条通过。测试覆盖无前缀 paged 路径、未对齐 prefix 的传统 fallback 和缺失 cache 的 fallback。命令使用个人账号、容器内本地 ext4 源码和输出目录，以及 `--config=cuda13 --config=sm10x`。原始 red/green 日志在 `test-paged-conv-bazel-111.tar.gz`；`host-selection-unit-111.json` 记录测试前服务占用。此时尚未重启服务，也没有新候选的 GPU 性能或完整模型正确性结论。

## Paged metadata 候选的双机构建与权重预检

111/112 各自在新建的个人 ext4 工作树上使用同一 `b24b6cd88` modeling 源码和同一远端开发分支；旧的 `3d7` 工作树与 trace 原文件保留。两端容器内均以 `luohaocheng.lhc`、`--config=cuda13 --config=sm10x` 编译，Bazel 各完成 23,714 个动作并报告成功。构建日志和哈希保存在 `pagedmeta-b24-build-logs-111112.*`。`host-selection-pagedmeta-before-restart.json` 记录旧服务仍在时的占用，`host-selection-pagedmeta-launch.json` 记录新服务启动前 111/112 的 GPU 选择和占用。

两端启动前的 FastSafetensors guard 均通过：target 从 3FS 读取，7 个 shard、16,402 个 tensor、54.47 GiB；MTP 视图在个人数据盘，9 个 shard、5,404 个 tensor、20.00 GiB，实际 shard 指向已核实的 3FS。启动配置显式为 `fastsafetensors`。四份 guard 输出及两份启动配置在 `pagedmeta-b24-preflight-111.tar.gz`、`pagedmeta-b24-preflight-112.tar.gz`，哈希见 `pagedmeta-b24-preflight-111112.sha256`。111/112 的 8 个 RDMA bond 都是 ACTIVE，互 ping 各 3 包、0% 丢包。这些预检和构建结果本身不证明新服务的 smoke 或性能。

## Paged metadata 候选的四层双机复核

`b24b6cd88` 在 111 Prefill、112 Decode 的独立个人工作树编译并启动；两端各 8 个 rank 的 FastSafetensors 日志通过校验。`host-selection-pagedmeta-flow.json` 和 `host-selection-pagedmeta-timeline.json` 记录了各次运行前的 GPU 占用检查。

四层 FP8、Native MTP、TP8/EP8 双机 PD flow 的 11 条请求全部返回；独立审计 `pagedmeta-b24-flow-independent-audit.json` 逐条检查了 PD 状态长度、Decode 路由、MTP draft 轮次、UTF-8 和 300 秒上限，11/11 通过，0 条跳过，最慢一条 72.284 秒。请求原文、响应和执行记录保存在 `pagedmeta-b24-flow-raw.tar.gz`，校验值在同名前缀的 `.sha256`。四层随机裁剪权重不能证明答案语义正确。

相同 65,536-token 输入和双机服务上，完成 10 次无前缀复用的同路径预热及 16 次正式采样，独立请求审计 `pagedmeta-b24-timeline-independent-audit.json` 全部通过。八个 rank 的 trace、原始请求和校验值分别保存在 `pagedmeta-b24-allrank-traces.tar.gz`、`pagedmeta-b24-timeline-requests.tar.gz` 和 `pagedmeta-b24-timeline-archives.sha256`。本轮脚本沿用了旧的 trace 文件名前缀 `k3_64k_integrated_kmerge_3d7_r1`；**文件内容属于新工作树中的 `b24b6cd88` 服务**，不能与旧版 `3d7` trace 混用。

`pagedmeta-b24-aligned-phase-audit.json` 匹配 6 条全 rank 请求：target Prefill GPU span 中位数 71.607 ms，draft 63.249 ms，完整 Prefill 134.950 ms。旧 `3d7` 分别为 72.255、62.956、135.148 ms；固定 `feat/k3_dev` `a9bf762e8` 分别为 71.556、62.564、135.266 ms。target trace 中原先每个 rank、每条请求出现的 8,192 项 `aten::copy_` 已消失；`pagedmeta-b24-target-gpu-idle.json.gz` 中最慢 rank 的 GPU 空隙中位数从 4.880 降到 4.100 ms。target 与固定 feat 相差约 0.051 ms，仍在样本波动范围内；本轮不据此宣称集成版 target 更快，也不作为完整模型性能结论。

同一独占双机服务上的第二轮采样使用相同输入、10 次同路径预热、16 次正式请求。请求独立审计通过 10/10 预热和 16/16 正式请求，Native MTP 均执行；全 rank trace 匹配 7 条。`pagedmeta-b24-r2-aligned-phase-audit.json` 的 target、draft、完整 Prefill GPU span 中位数分别为 71.914、63.330、135.262 ms。最慢 rank 的 target GPU 空隙中位数为 4.284 ms。第二轮 trace 文件名使用正确的 `pagedmeta_b24_r2` 前缀和 `_2` 后缀；原始 trace、请求、哈希、逐模块归属、空隙分析及运行前占用快照均以 `pagedmeta-b24-r2-*` 和 `host-selection-pagedmeta-r2-timeline.json` 保存。两轮集成版 target 中位数一快一慢于固定 feat 的 71.556 ms；完整 Prefill 接近持平。固定 feat 只有既有的一轮 7 条匹配请求，且其中含明显慢样本；这些数据不足以声称集成版每模块或完整模型更快。

为了检查准备阶段的一个具体候选，我在 110 的独占 GPU 1 上测量 `prepare_paged_short_conv_metadata`，包含 CPU 规划、两份 device metadata 传输及末尾 CUDA 同步。8192 token 单序列先完成 10 次稳定预热、再测 100 次，中位 0.044861 ms；两个 4096-token 序列预热 16 次、测 100 次，中位 0.045854 ms。原始逐次结果在 `bench-paged-conv-metadata-before-110.json`，选择快照在 `host-selection-single-seq-meta-bench.json`，脚本为 `bench_paged_conv_metadata.py`。这些耗时远小于 trace 中 0.7–1.4 ms 的较大 GPU 空隙，不能把整个空隙归因于此函数，因此没有据此改动算子或缓存接口。固定 feat trace 的 target 起点还包含约 1.16 ms 的 `buildAttentionInputMetadataKernel`；两版 GPU 忙碌与空隙的组成不同，需对照相同阶段再判断性能。

切换服务前，将 111 Prefill 和 112 Decode 的本轮完整启动目录分别保存为 `pagedmeta-b24-startup-111.tar.gz` 和 `pagedmeta-b24-startup-112.tar.gz`，哈希见 `pagedmeta-b24-startup-archives.sha256`。归档包含启动配置、FastSafetensors guard、RDMA 预检、主进程与八个 rank 的日志；活跃 Unix socket 不是可归档文件，tar 按预期跳过它。

为了在同一 111/112 双机上复测固定 feat `a9bf762e8`，核查发现 112 的旧 Bazel 可执行产物已缺失。直接重建先因旧版 `rules_python`、`rules_cc` 指向无 SSH 授权的内部 GitLab 地址而失败；随后使用 112 个人数据盘上原有、对应 feat 的 17 份外部依赖源码快照和 RDMA 构建 overlay，保持源码提交、CUDA13/SM10x、个人账号和本机独立 Bazel 输出目录不变。`build_feat_a9bf_112_20260929.sh` 的构建完成 30,181 个动作，退出码 0，`//rtp_llm:rtp_llm_server` 可执行入口存在。完整成功日志在 `feat-a9bf-rebuild-112-with-sources-20260929.log.gz`（SHA256 `6f0f95da23dfa55776a854acf1b110228c3ec26e35c8774ea5500193c00d8dba`）。111 的固定 feat 构建产物也仍存在。以上只证明双机重启前的构建条件，还不代表 feat r8 服务或性能结果。

## 固定 feat 的同机复测 r8

111 Prefill、112 Decode 运行固定源码 `a9bf762e8`，四层 target 为 FP8、draft 为原生 MTP，TP8/EP8 双机 PD，Decode 开启 CUDA Graph。两端八个 rank 的启动日志均选择 FastSafetensors，直接读取已核实的 3FS shard。`host-selection-feat-r8-launch.json` 和 `host-selection-feat-r8-timeline-immediate.json` 分别记录启动前和测量前的 GPU 状态；测量前只有本任务服务进程占用 GPU。四层 flow 的 10 条请求全部返回；`feat-a9bf-r8-flow-independent-audit.json` 对原始响应独立复核为 10/10。旧 feat 的 HTTP aux 不暴露 draft 轮次，也把 Decode 复用长度报为 0，因此不能靠 flow 单独证明 MTP 执行和 KV 交接字节数。

64K 测量使用与集成版相同的固定输入，token SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`。禁用前缀复用，先完成 10 次同路径预热；末三次首 token 时间为 135.866、142.930、136.749 ms，均在中位数 ±5% 内。随后 16 次正式请求均为 HTTP 200，独立原始响应复核无错误或 Unicode 替换字符。八 rank trace 匹配到 6 条共同请求，每条都有 target 和 draft Prefill 范围，证实原生 MTP draft 实际运行。按每条请求最慢 rank 的 GPU span 取中位，target 为 **71.033 ms**、draft 为 **62.609 ms**、完整 Prefill 为 **134.763 ms**。逐模块归属见 `feat-a9bf-r8-module-target.json` 和 `feat-a9bf-r8-module-draft.json`；累计核时间可能跨 stream 重叠，不能相加成关键路径。

本轮相较于上一轮固定 feat r7 的 target 71.556 ms、完整 Prefill 135.266 ms 略快。集成版 `b24b6cd88` 两轮 target 为 71.607 / 71.914 ms，完整 Prefill 为 134.950 / 135.262 ms。当前差距在 0.2–0.9 ms，且各轮匹配请求数不同；这些数据仍不足以证明集成版四层 Prefill 稳定优于 feat。feat 的 FP8 融合通信路径也不满足最终 BF16 NCCL 契约，不能直接迁入。此处只记录可比现状，不据此提交性能锚点。

原始数据：`feat-a9bf-r8-allrank-traces.tar.gz` SHA256 `ef7d06639be74a7f1c340d2ded494150f62da3bdd9474caac4c78987c919d4fd`；`feat-a9bf-r8-timeline-requests.tar.gz` SHA256 `c32bf83962d7a736daed1a8f7fb946674ae9ca7773dbe8ecde1fd403d7dfabc0`；Prefill 与 Decode 启动记录归档的 SHA256 分别为 `90835b5a5fc3ce1c8f8a5ed227bd261b19a51e2c0a3dfdecb0b77d450e9134c1`、`6599459bcc9f24b2ff40ee7d46d1cd3ef407f972bf547c408c04199faffe46b4`。`feat-a9bf-r8-timeline-independent-audit.json` 保存逐请求复核和旧 aux 的证据边界。启动阶段没有观察到明显的 3FS 读取阻塞，本轮未改变机器级并发配置。

### KV 写入核的配置差异

r8 的 `concat_and_cache_mla_kernel` 在固定 feat trace 中每 rank 约 0.074 ms，集成版 b24 r2 约 0.212 ms；两边 kernel 符号、65,536 个 CTA、512 线程配置相同。启动日志显示一个不能忽略的差别：feat Prefill 的 `kv_cache_sharded=true`、`local_kv_page_rr_shard_count=8`、`enable_sp=0`；集成版 Prefill 的 `kv_cache_sharded=0`、`enable_sp=1`、`ffn_sp_size=8`。因此两边虽同为四层、TP8/EP8、FP8+MTP PD，KV 写入和 SP 的实际执行配置并不相同。现有每算子时间不能直接归因于 kernel 实现优劣。

在独占的 112 GPU 1 上，用集成版 b24 构建产物单独测同一个普通 E4M3 写入核。每组先完成惰性初始化及至少 10 次稳定预热，再采 50 组、每组 10 次调用。64K token 全部有有效槽位时，128-token page、4096-token page 和四倍物理 page 间距分别为 0.2031、0.2031、0.2006 ms；仅 1/8 token 有有效槽位的诊断组为 0.0557 ms。原始样本、预热和输入布局见 `bench-mla-cache-writer-layout-112.json`，脚本为 `bench_mla_cache_writer_layout.py`，GPU 筛选记录在 `host-selection-mla-cache-writer-*.json`。该结果说明 page 大小和物理间距没有解释 0.14 ms 差距；有效槽位比例能明显改变耗时。它**不能**证明固定 feat 的真实 slot mapping 恰好只有 1/8 有效，需要结合 CP 映射及 PD 正确性继续核查。当前不能把同名 kernel 当作“feat 更快的实现”直接迁移。

保留 r8 原始数据后，我只停止了本任务在 111/112 的两个已核实进程组；服务管理器逐个退出 rank 后还有本任务孤儿 rank，复查 PID/UID/进程组后清理完毕。两端端口已关闭，GPU 无活动计算进程。保留服务模式的外层控制器因这次主动退出返回 1，原始输出在 `feat-a9bf-r8-controller-after-intentional-teardown.log.gz`；flow runner 自身及 64K 采集 runner 均退出 0，控制器返回值不能用于否定已归档的请求结果。
