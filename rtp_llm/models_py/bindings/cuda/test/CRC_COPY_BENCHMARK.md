**CRC copy 的手动正确性与性能测试**

从仓库根目录执行。需要 CUDA 13 构建环境、一张空闲 NVIDIA GPU，以及可用的 `nvidia-smi` 和 GPU NUMA sysfs 信息；不需要模型 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 bazelisk test \
  //rtp_llm/models_py/bindings/cuda/test:crc_copy_benchmark_test \
  --config=cuda13 \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --test_env=CUDA_VISIBLE_DEVICES \
  --test_env=GPU_COUNT=1 --test_env=WORLD_SIZE=1 \
  --cache_test_results=no --test_output=streamed
```

该目标标记为 `manual`、`local`、`exclusive`，不会随 `bazelisk test //...` 自动运行。显式调用时不要省略 `--cache_test_results=no`，否则可能读到 Bazel 缓存的旧测试结果。GPU lock 协调本仓库的测试任务；运行前仍需确认 GPU 没有其他工作负载。已有 `crc_block_copy_batch_test` 保持常规回归 UT，不受该目标影响。

默认做两次独立进程运行，每次每个点预热 30 轮、采样 100 轮。配置为 DeepSeek V4 Pro FP8、TP8、PREFILL CP_RR/CP8、`seq_size_per_block=128`、`gen_num_per_cycle=0`，分别测 FULL 与 SWA。BS 是一次调用中的本地 backing 数，覆盖每个整数 1–32。总计 128 个配置、640 个路径点、128000 个耗时样本；这不是八 rank 并发或整模型吞吐测试。

| 路径 | 实现与计时边界 |
|---|---|
| `integrated_crc` | 直接链接当前仓库的 `CrcBlockCopyBatch`。包含 metadata、GPU CRC、数据传输、同步和 verdict。 |
| `copy1d_batch` | 调用真实 `CudaBatchDeviceHostCopyStrategy`，包含每次 descriptor 构造和 binding 内同步；禁止自动 fallback。 |
| `copy3d_batch` | 测试中的 3D 实验实现，每 backing 三个 affine pool 操作；构造和检查 descriptor、提交、等待完成都计时。生产 stream/锁是私有实现，因此使用独立的 Torch 非默认 stream 和测试局部提交锁。 |
| `main_staged` | 调用真实 `StagedSmDeviceHostCopyStrategy`。包含 GPU staging、pinned scratch 和 CPU pack/unpack。 |
| `gather_control` | 测试局部的无 CRC 对照：gather/scatter、GPU staging、最终 pinned host，传输与 CRC 相同的编码长度；无 CRC/footer sealing/Result。它不等同于 main staged。 |

生产实现没有复制到测试中，修改仓库实现后重新构建即可测到新代码。3D 与无 CRC control 是诊断用实现。它们和 CRC 的差值不代表单个 CRC kernel 的时间。

框架在 CUDA13 x86/ARM 构建中默认对 HOST/DISK cache 的传输启用 CRC，没有运行时开关；不支持 CRC 的构建沿用原 copy 路径。本测试显式调用各条实现做对照，`main_staged` 等无 CRC 路径不是框架的配置选项。

| backing | 按 pool/layer 的 tile 字节数 | payload/backing | tiles/backing |
|---|---|---:|---:|
| FULL | 31×1152 + 30×19008 + 30×4224 | 732672 B | 91 |
| SWA | 61×9360 + 30×2048 + 30×8192 | 878160 B | 121 |

CRC/control 每个 backing 传 `E=align16(P+4)=P+16`；其他路径传 payload P。CRC 另有每 backing 12 B Result。Host backing 使用 `align4096(E)` stride，均为 pinned memory。FULL page 保留完整大小；SWA 使用 CP_RR 本地 slice。Device-only HCA state 和额外 speculative 状态不进入测试。

每个进程先检查 CRC 已知向量和 mixed/ragged 输入。每个 shape 采样前，所有 BS/路径执行两种物理 block rotation 的独立 CPU oracle/guard 校验；CRC 还检查损坏后的拒写及恢复。任何校验失败、不支持或 CUDA 错误都使测试失败，保留当时日志，不生成成功的性能报告。

每个计时调用前驱逐独立的 8×L2 buffer，源池/目标池也至少为 8×L2，并轮换物理 block。同轮各路径使用相同数据、随机执行顺序，CPU descriptor metadata 统一预热。驱逐不计时，路径内部不插入驱逐；没有硬件 counter 保证 100% L2 miss。H2D host 输入每轮由 CPU 准备，不声称 host cache 冷。默认将进程绑到 GPU 附近的一个允许使用的 CPU；可用 `--test_arg=--cpu=63` 指定，或 `--test_arg=--cpu=none` 明确关闭。显存需容纳多个至少 8×L2 的池及 CUDA/CRC workspace，host 内存还需容纳 oracle 和轮换计划。

计时是预构造原生输入后的一次完整同步调用的 host wall time，包括实现内部的参数构造、锁、拷贝、kernel 和同步。没有 executor/RPC/模型计算。CSV 的 GB/s 使用有效 payload/完整调用耗时，不能当成物理链路带宽。

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

统计入口严格检查完整矩阵、同轮 source/执行位置、有效耗时、GPU/运行参数一致性和显式排除。所有样本保留，p50 为中位数，p95 为 nearest rank；CRC 对比使用同轮差值及比值的中位数。`analysis.json` 给出重复运行、四个时间分段和执行位置的漂移告警，超过 10% 时应先检查环境。告警不被当作固定的性能 pass/fail 阈值。

记录所用提交、上述完整命令和测试产物，才能与其他机器比较：

```bash
git rev-parse HEAD
```

框架默认 `memory_cache_max_descriptors_per_transfer_batch=8`。本测试直接提交至多 32 个 backing；保持默认上限时，当前 CRC 的 BS≥16 分段分支不会触发。

统计器本身有不占 GPU 的回归测试，检查缺失/重复样本、错误配对、非法耗时、损坏检查失败和错误排除声明：

```bash
bazelisk test //rtp_llm/models_py/bindings/cuda/test:crc_copy_benchmark_stats_test \
  --config=cuda13 --cache_test_results=no --test_output=errors
```
