**CRC copy 的手动正确性与性能测试**

从仓库根目录执行。需要 CUDA 13 构建环境、一张空闲 NVIDIA GPU，以及可用的 `nvidia-smi` 和 GPU NUMA sysfs 信息；不需要模型 checkpoint。

```bash
export CRC_BENCH_SOURCE_COMMIT=$(git rev-parse HEAD)
export CRC_BENCH_BASE_COMMIT=$(git merge-base HEAD origin/main)
CUDA_VISIBLE_DEVICES=0 bazelisk test \
  //rtp_llm/models_py/bindings/cuda/test:crc_copy_benchmark_test \
  --config=cuda13 \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --test_env=CUDA_VISIBLE_DEVICES \
  --test_env=GPU_COUNT=1 --test_env=WORLD_SIZE=1 \
  --test_env=CRC_BENCH_SOURCE_COMMIT --test_env=CRC_BENCH_BASE_COMMIT \
  --cache_test_results=no --test_output=streamed
```

该目标标记为 `manual`、`local`、`exclusive`，不会随 `bazelisk test //...` 自动运行。显式调用时不要省略 `--cache_test_results=no`，否则可能读到 Bazel 缓存的旧测试结果。GPU lock 协调本仓库的测试任务；运行前仍需确认 GPU 没有其他工作负载。已有 `crc_block_copy_batch_test` 保持常规回归 UT，不受该目标影响。

默认做两次独立进程运行，每次每个点预热 30 轮、采样 100 轮。配置为 DeepSeek V4 Flash-0731 FP8、TP4、PREFILL CP_RR/CP4、`kv_cache_sharded=true`、`seq_size_per_block=kernel_seq_size_per_block=128`、`gen_num_per_cycle=0`，分别测 FULL 与 SWA。BS 是一次调用中的本地 backing 数，覆盖每个整数 1–32。五条比较路径，共 128 个配置、640 个路径点、128000 个耗时样本；这些数据衡量单 GPU copy，不代表四 rank 并发或整模型吞吐。

模型采用 Flash-0731 的 43 个 target layer；前两层压缩比为 0，随后交替为 4 和 128；本测试通过 `gen_num_per_cycle=0` 关闭 speculative decoding，仅测这 43 层，FP8 specs 通过 `test::setDsv4KvCacheSpecs` 构造，再交给当前 main API 的 `CacheConfigCreator::createWarmupConfig` 和 `DeviceBlockPoolConfigHelper::createConfigForGroup` 解析。Benchmark 从物理 pool 的层数、KV/scale stride、offset 生成 tile，不再用硬编码的 tile 长度列表构造输入。只选择允许 prefix reuse 的 FULL/SWA group，并保留 topology 的 tag 顺序及每个 tag 的模型层映射。统计器另有独立的 shape 预期，模型配置变化时会拒绝混用数据。

| 路径 | 实现与计时边界 |
|---|---|
| `integrated_crc` | 直接链接当前仓库的 `CrcBlockCopyBatch`，每个 backing 的完整 payload 对应一条 CRC 消息。包含 metadata、GPU CRC、数据传输、同步和 verdict。 |
| `copy1d_batch` | 调用真实 `CudaBatchDeviceHostCopyStrategy`，包含每次 descriptor 构造和 binding 内同步；禁止自动 fallback。 |
| `copy3d_batch` | 从实际 tiles 推导连续的 affine span；只有宽度相同、host 连续、device pitch 恒定的 tiles 才合并。当前 shape 预期每 backing 三个操作，代码不写死这个数量。构造和检查 descriptor、提交、等待完成都计时；使用独立 Torch 非默认 stream 和测试局部提交锁。 |
| `staged_no_crc` | 调用真实的 `StagedSmDeviceHostCopyStrategy`，保留与 CRC 统一的 `CopyTileKernel`。GPU staging 与最终 HOST segment 直接 DMA，不再通过额外 pinned host scratch 做 CPU payload pack/unpack。 |
| `main_staged` | main `e73826e27a6005640bb75ce2a73f235bfa51c55b` 的 staged strategy、scratch 和 executor 原样放入测试 namespace；沿用该版本的 SM gather/scatter kernel，包含三组 metadata 上传、pinned scratch、CPU pack/unpack 和同步。 |

CRC、1D 和优化 staged 直接链接当前生产实现。main staged 单独保留固定提交的测试快照，用于避免与优化后的实现混用；3D 是测试内从实际 tiles 推导的对照。各路径与 CRC 的耗时差都不代表单个 CRC kernel 的时间。

框架在 CUDA13 x86/ARM 构建中默认对 HOST/DISK cache 的传输启用 CRC，没有运行时开关；不支持 CRC 的构建沿用原 copy 路径。本测试显式调用各条实现做对照，`staged_no_crc` 等无 CRC 路径不是框架的配置选项。

| backing | 按 pool/layer 的 tile 字节数 | payload/backing | tiles/backing |
|---|---|---:|---:|
| FULL | 21×19008 + 21×4224 + 20×1152 | 510912 B | 62 |
| SWA | 43×18720 + 21×4096 + 21×16384 | 1235040 B | 85 |

CRC 每个 backing 传 `E=align16(P+4)=P+16`；其他路径传 payload P。CRC 另有每 backing 12 B Result。Host backing 使用当前普通 HOST pool 的 16 字节对齐 stride，即 `E`，均为 pinned memory；4 KiB 对齐只用于 direct disk I/O，不用于本次 HOST↔DEVICE 对照。FULL page 保留完整大小；SWA 使用 CP_RR 本地 slice。Device-only HCA state 和额外 speculative 状态不进入测试。

每个进程先检查 CRC 已知向量和 mixed/ragged 输入。每个 shape 采样前，所有 BS/路径执行两种物理 block rotation 的独立 CPU oracle/guard 校验；CRC 还检查损坏后的拒写及恢复。H2D 的每次预热和采样还会在计时前清空整个目标 pool，在计时后核对所选 block 的 payload 和未选 block 的 guard，避免前一条成功路径掩盖漏写，也覆盖实际测量使用的地址组合。这些清理和核验都不计时，不读取 HOST payload；目标清理后仍执行统一的 L2 驱逐。D2H 保留采样前的 oracle 校验，不在计时轮后增加 CPU 读取。任何校验失败、不支持或 CUDA 错误都使测试失败，保留当时日志，不生成成功的性能报告。v5 metadata 声明 `measured_h2d_verified=true`，并区分优化 staged 的 `direct_dma` 与 main 快照的 `cpu_pack_unpack`。统计器拒绝缺少这些契约的旧数据，也拒绝把插桩结果当作普通性能数据。

每个计时调用前驱逐独立的 8×L2 buffer，源池/目标池中可轮换的有效容量也至少为 8×L2。保留 block 0 作为 sentinel，另增一个 block，避免排除 sentinel 后容量不足。同轮各路径使用相同数据、随机执行顺序，CPU descriptor metadata 统一预热。驱逐不计时，路径内部不插入驱逐；没有硬件 counter 保证 100% L2 miss。H2D host 输入每轮由 CPU 准备，不声称 host cache 冷。默认将进程绑到 GPU 附近的一个允许使用的 CPU；可用 `--test_arg=--cpu=63` 指定，或 `--test_arg=--cpu=none` 明确关闭。显存需容纳多个至少 8×L2 的池及 CUDA/CRC workspace，host 内存还需容纳 oracle 和轮换计划。

计时是预构造原生输入后的一次完整同步调用的 host wall time，包括实现内部的参数构造、锁、拷贝、kernel 和同步。没有 executor/RPC/模型计算。CSV 的 GB/s 使用有效 payload/完整调用耗时，不能当成物理链路带宽。

**CPU payload memcpy 归因诊断**

将主命令的目标替换为 `//rtp_llm/models_py/bindings/cuda/test:staged_memcpy_profile_test`，即可运行独立诊断。它保留五路随机配对、双向 FULL/SWA、两进程各 30 轮预热和 100 轮采样，仅选择 BS=1、8、32，共 12000 个完整调用耗时样本。普通 benchmark 不链接 memcpy wrapper，不受诊断计时插桩影响。

诊断只在两条 staged 调用期间包装实际的 libc `memcpy`，仅记录大小等于当前 backing payload 的调用。优化 staged 必须是 0 次、0 字节、0 耗时；main 快照必须是 BS 次、BS×payload 字节，作为拦截有效的正对照。任一计数不符会使测试失败。两条路径共输出 4800 条 profile，与完整调用样本按方向、layout、BS、repeat、round、variant、执行位置和 source plan 对应，五路样本不会重复计数。

诊断使用 `staged_memcpy_profile_v2`，报告总同步调用时间、CPU payload memcpy 时间及剩余时间。剩余部分仍包括 CPU 参数构造、CUDA 提交和等待，不能称为纯 GPU 时间；插桩开销也保留在总时间内。零计数只说明原 payload 大小的 libc memcpy 已消失，仍需结合源码确认没有改用其他 CPU 拷贝形式。诊断结果用于归因，正式五路性能表使用未插桩目标的数据。优化前的 v1 结果单独保留，不与 v2 合并。

**已知环境失败与显式排除**

2026-09-24 在 CUDA runtime 13.2、驱动 580.105.08、设备报告 SM103 的环境，历史脚本的 1D H2D 在 FULL BS=11 出现数据不一致，纯 CUDA 复现也在 BS=11/14 返回 719。本 Bazel 目标直接调用生产 strategy，在 FULL BS=8、rotation=1 检出了 payload/guard 不一致。失败点并不固定，根因未定位；不能泛化为每台机器、每个 BS 都有缺陷。因此默认仍测完整矩阵，以便其他环境复现或确认问题是否已修复。

仅需复现该次报告的有效部分时，在主命令末尾显式追加：

```bash
--test_arg=--exclude-1d-h2d
```

此时 D2H 五路、H2D 四路，共 115200 个耗时样本。Metadata、每个排除格和 CSV 都明确标记 1D H2D 不可用，不把它计为通过，也不会自动重试成别的 copy 实现。

只做全 BS 正确性检查或快速验证构建，可分别追加：

```bash
--test_arg=--correctness-only
# 或：保留全部正确性检查，但减少性能样本；不要与默认样本数混用。
--test_arg=--iterations=4 --test_arg=--warmup=1
```

两种模式都支持显式排除。默认性能比较使用 200 样本/有效点；快速验证模式不应作为正式性能结论。

**输出与结果检查**

输出写入 Bazel 的 `TEST_UNDECLARED_OUTPUTS_DIR`。通常在：

```text
bazel-testlogs/rtp_llm/models_py/bindings/cuda/test/crc_copy_benchmark_test/
  test.log
  test.outputs/outputs.zip
```

归档包含 `copy_80.jsonl`、`copy_81.jsonl`、各进程 stdout/stderr/exitcode、`run_manifest.json`、`analysis.json`、`latency.csv` 和 `summary.md`。Bazel 版本或配置不同也可能保留未压缩输出目录。失败时已产生的原始记录仍保留；退出码失败不能解释为成功结果。需要固定输出位置时，可追加 `--test_arg=--output-dir=/absolute/empty/directory`；已有结果的目录会被拒绝，避免旧报告混入新运行。

统计入口严格检查完整矩阵、同轮 source/执行位置、有效耗时、GPU/运行参数一致性和显式排除，也逐 tile 检查 tag、member、局部层号、模型层号和物理 offset。所有样本保留，p50 为中位数，p95 为 nearest rank；CRC 对比使用同轮差值及比值的中位数。`analysis.json` 给出重复运行、四个时间分段和执行位置的漂移告警，超过 10% 时应先检查环境。告警不被当作固定的性能 pass/fail 阈值。

运行前应提交待测源码，保证两项 commit 环境变量对应构建输入。Runner 要求提供完整 SHA，将 source commit、main base commit 和实际 binary SHA256 写入 manifest 及每次进程的原始 metadata；缺失这些信息时直接失败。两次进程的数据只有来源一致才能合并。请同时保留完整命令和测试产物。

框架默认 `memory_cache_max_descriptors_per_transfer_batch=8`。本测试直接提交至多 32 个 backing，测量单次 copy 的批量扩展情况，不代表默认框架配置的请求耗时。

统计器本身有不占 GPU 的回归测试，检查缺失/重复样本、错误配对、非法耗时、损坏检查失败和错误排除声明：

```bash
bazelisk test //rtp_llm/models_py/bindings/cuda/test:crc_copy_benchmark_stats_test \
  --config=cuda13 --cache_test_results=no --test_output=errors
```
