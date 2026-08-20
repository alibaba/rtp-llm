# BlockTreeCache Benchmark Nsight Systems 采集指南

本文说明如何为 BlockTreeCache benchmark 采集 Nsight Systems（`nsys`）数据，以及打开报告后通常需要关注什么。本文不提供特定机器或 case 的性能结论。

## 基本原则

- 先裸跑 benchmark，保存 `result.json`；Nsight Systems 会引入额外开销，profile run 不作为吞吐基线。
- 普通 CUDA timeline 和 GPU Metrics 建议分两次采集。前者用于 API/GPU 时序和耗时，后者用于判断 Copy Engine、SM、PCIe、BAR1 等硬件活动。
- 一次只采一个明确的 case、方向和 strategy。正式 suite 的混合方向和 30 秒以上 measured window 会生成较大的报告，也不利于归因。
- `.nsys-rep` 是原始报告，供 `nsys-ui` 打开；`.sqlite` 是从 report 导出的派生数据，不需要一起采集。

## 前置准备

先按 benchmark README 完成构建，然后进入开源目录：

```bash
cd github-opensource

export LD_LIBRARY_PATH=/opt/conda310/lib:${LD_LIBRARY_PATH:-}

BENCH_DIR=rtp_llm/cpp/cache/block_tree_cache/benchmark
BINARY=./bazel-bin/$BENCH_DIR/block_tree_cache_gpu_benchmark
TRANSFER_PROFILE=$BENCH_DIR/profiles/deepseek_v4_pro_fp8_descriptor_sizes.json
TREE_PROFILE=$BENCH_DIR/profiles/deepseek_v4_pro_fp8_tp1_cp1.json
OUT_DIR=/tmp/block_tree_cache_nsys

mkdir -p "$OUT_DIR"
test -x "$BINARY"
nsys --version
```

Transfer binary 的公共参数仍名为 `--model-profile`，但应传入 descriptor-size profile：

```text
deepseek_v4_pro_fp8_descriptor_sizes.json
```

Tree binary 才使用完整 model profile：

```text
deepseek_v4_pro_fp8_tp1_cp1.json
```

建议把以下信息与报告一起记录：代码 commit、binary SHA256、profile SHA256、GPU/Driver/CUDA/Nsight Systems 版本，以及完整 benchmark 命令。

## 先裸跑目标 workload

下面以 `full_context`、单方向 H2D、CUDA batch strategy 为例。`--min-measured-seconds=5` 足以采到稳定重复事件，同时控制报告大小：

```bash
"$BINARY" transfer \
  --cuda-device 0 \
  --max-device-memory-fraction 0.8 \
  --model-profile "$TRANSFER_PROFILE" \
  --seed 42 \
  --repetition-id 0 \
  --output-json "$OUT_DIR/full_context_h2d_baseline.json" \
  --group-set full_context \
  --transfer-directions h2d \
  --transfer-operation-count 4096 \
  --transfer-concurrency 8 \
  --transfer-descriptor-batch-size 8 \
  --copy-strategy batch \
  --host-memory pinned \
  --min-measured-seconds 5
```

采集前确认 `result.json` 中状态为 completed、失败操作数为 0，并保存 resolved configuration。切换 workload 时只修改需要比较的维度：

- GroupSet：`full_context` 或 `swa`。
- 方向：`d2h`、`h2d`、`d2disk`、`disk2d`、`h2disk`、`disk2h`。
- Device/Host strategy：`batch` 或 `staged-sm`。
- Disk case 还需要设置 `--disk-path`、`--disk-io-mode` 和 `--disk-access-pattern`。

## 采集普通 CUDA timeline

普通 timeline 用于观察 CUDA Runtime/Driver API、GPU memcpy、kernel、stream synchronization 和 OS Runtime 等事件：

```bash
TRACE_PREFIX=$OUT_DIR/full_context_h2d_timeline

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=process-tree \
  --osrt-threshold=10000 \
  --stats=false \
  --force-overwrite=true \
  --output="$TRACE_PREFIX" \
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "$BINARY" transfer \
    --cuda-device 0 \
    --max-device-memory-fraction 0.8 \
    --model-profile "$TRANSFER_PROFILE" \
    --seed 42 \
    --repetition-id 0 \
    --output-json "$OUT_DIR/full_context_h2d_timeline_result.json" \
    --group-set full_context \
    --transfer-directions h2d \
    --transfer-operation-count 4096 \
    --transfer-concurrency 8 \
    --transfer-descriptor-batch-size 8 \
    --copy-strategy batch \
    --host-memory pinned \
    --min-measured-seconds 5
```

输出文件为：

```text
/tmp/block_tree_cache_nsys/full_context_h2d_timeline.nsys-rep
```

采集磁盘方向时，在 `nsys profile` 参数中额外增加：

```bash
--osrt-file-access=true
```

这样可以在 report 中查看 `pread`、`pwrite` 等文件访问事件。磁盘目录应位于真正要测试的文件系统，而不是未确认后端的容器 overlay。

Tree 采集使用相同的 `nsys profile` 前缀，将 target command 换为：

```bash
"$BINARY" tree \
  --cuda-device 0 \
  --max-device-memory-fraction 0.8 \
  --model-profile "$TREE_PROFILE" \
  --seed 42 \
  --repetition-id 0 \
  --task-pool-size 4 \
  --output-json "$OUT_DIR/tree_timeline_result.json"
```

Tree workload 的 measured window 固定较长，报告通常明显大于单方向 Transfer 报告。CPU on-CPU/off-CPU 分析仍优先使用 README 中的 perf/BCC 流程；nsys 主要用于 CUDA 和跨 CPU/GPU 时间线。

## 采集 GPU Metrics

GPU Metrics 是周期采样的设备级指标。先确认当前 Nsight Systems 能识别目标 GPU 和 metric set：

```bash
nsys profile --gpu-metrics-devices=help
nsys profile --gpu-metrics-set=help
```

Nsight Systems 2025.6 使用 `--gpu-metrics-devices`；如果其他版本提示参数不存在，以 `nsys profile --help` 中显示的参数名为准。

下面仍使用同一个 workload，采集 device 0 的 10 kHz GPU Metrics。省略 `--gpu-metrics-set` 时，Nsight Systems 会自动选择适用于当前 GPU 的默认集合；需要严格复现时可显式补上 help 输出中的 alias。

```bash
METRICS_PREFIX=$OUT_DIR/full_context_h2d_gpu_metrics

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=10000 \
  --stats=false \
  --force-overwrite=true \
  --output="$METRICS_PREFIX" \
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "$BINARY" transfer \
    --cuda-device 0 \
    --max-device-memory-fraction 0.8 \
    --model-profile "$TRANSFER_PROFILE" \
    --seed 42 \
    --repetition-id 0 \
    --output-json "$OUT_DIR/full_context_h2d_gpu_metrics_result.json" \
    --group-set full_context \
    --transfer-directions h2d \
    --transfer-operation-count 4096 \
    --transfer-concurrency 8 \
    --transfer-descriptor-batch-size 8 \
    --copy-strategy batch \
    --host-memory pinned \
    --min-measured-seconds 5
```

### ERR_NVGPUCTRPERM

如果出现 `ERR_NVGPUCTRPERM`，说明 collector 没有读取 GPU performance counters 的权限。推荐的一次性处理方式是让 `nsys` collector 以 root 运行，同时通过 `--run-as` 让 benchmark 仍以普通用户运行：

```bash
BENCH_USER=$(id -un)
NSYS_BIN=$(command -v nsys)

sudo "$NSYS_BIN" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=10000 \
  --stats=false \
  --force-overwrite=true \
  --output="$METRICS_PREFIX" \
  --run-as="$BENCH_USER" \
  env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  "$BINARY" transfer \
    --cuda-device 0 \
    --max-device-memory-fraction 0.8 \
    --model-profile "$TRANSFER_PROFILE" \
    --seed 42 \
    --repetition-id 0 \
    --output-json "$OUT_DIR/full_context_h2d_gpu_metrics_result.json" \
    --group-set full_context \
    --transfer-directions h2d \
    --transfer-operation-count 4096 \
    --transfer-concurrency 8 \
    --transfer-descriptor-batch-size 8 \
    --copy-strategy batch \
    --host-memory pinned \
    --min-measured-seconds 5

sudo chown "$(id -u):$(id -g)" "$METRICS_PREFIX.nsys-rep"
```

在容器中可用相同原则：通过 `docker exec --user root` 启动 collector，再把 `--run-as=<普通用户>` 传给 nsys。仅仅以 privileged 模式创建容器，并不代表 `docker exec --user <普通用户>` 后仍保有读取 counters 的 capability。

采集完成后，如果 report 归 root 所有，应将文件 owner 改回工作用户。不要为了单次采集直接修改全机 Driver 权限策略。NVIDIA 的权限说明见 [ERR_NVGPUCTRPERM](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)。

## 打开和导出报告

远程服务器只负责采集。把 `.nsys-rep` 复制到安装了相同或更高版本 `nsys-ui` 的工作站即可打开；查看 UI 不需要 GPU，也不需要复制 `.sqlite`。

命令行摘要：

```bash
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,osrt_sum \
  --force-export=true \
  "$TRACE_PREFIX.nsys-rep"
```

导出 SQLite 做自定义查询：

```bash
nsys export \
  --type=sqlite \
  --force-overwrite=true \
  --output="$TRACE_PREFIX.sqlite" \
  "$TRACE_PREFIX.nsys-rep"
```

如果 `.nsys-rep` 更新过，而同名 `.sqlite` 更旧，必须重新强制导出。`nsys stats --report` 的 report list 中不要加入空格。

## 打开后通常关注什么

### CUDA API 与 GPU workload

- 用 correlation arrow 或 correlation ID 对齐 CUDA API 与对应的 GPU activity。
- 分开记录 API duration、GPU activity duration 和端到端 wrapper/wall time；重叠区间不能相加。
- 查看 memcpy 的方向、bytes、stream，以及一次 API 是否对应多个 GPU activities。
- 查看 `cudaStreamSynchronize`、event wait 和 GPU idle gap，判断 CPU 提交、GPU 执行还是同步等待占主导。

### GPU engine 与数据链路

把以下 GPU Metrics 行与目标 memcpy/kernel 时间窗对齐，而不是对整个 report 做笼统平均：

- `Async Copy Engine Active 0/1/...`、`Sync Copy Engine Active`：Copy Engine 活动。
- `SMs Active`、`Compute in Flight`、`GR Active`：计算/graphics engine 域活动。
- `PCIe RX/TX Throughput`：GPU 接收/发出的 PCIe 流量。
- `PCIe Read/Write Requests to BAR1`：BAR1 aperture 请求。
- `DRAM Read/Write Bandwidth`：显存侧读写压力。

单一指标通常不足以证明实际路径。例如看到 `GR Active` 不能直接等同于 BAR1；应同时核对 Copy Engine、SM、PCIe、BAR1、CUDA activity 和 CPU API 时序。

### Kernel 和 staged copy

- 分开查看 metadata memcpy、bulk memcpy、gather/scatter kernel 和 stream synchronization。
- 检查 kernel launch 数、duration、相邻 memcpy/kernel gap，以及是否与其他 GPU workload 竞争。
- 计算有效带宽时使用 activity 的实际 bytes 和 duration，并注明是单 activity、单 API 聚合还是端到端 wall time。

### Disk transfer

- 在 OS Runtime/File Access 中查看 `pread`/`pwrite` 的调用数、size、duration 和线程并发。
- 对齐 disk IO 与前后的 D2H/H2D staging，区分串行阶段和流水重叠阶段。
- 同时保留磁盘路径、文件系统、direct/buffered mode 和 working-set 配置；否则结果不可复现。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| report 很大、UI 打开很慢 | 缩短 `--min-measured-seconds`，使用单方向、单 case，不要直接 profile 整个 suite |
| GPU Metrics 行不存在 | 确认采集命令包含 `--gpu-metrics-devices`，并检查 counters 权限 |
| UI 中一次 batch 显示成多个 memcpy | 先核对 correlation ID、各 activity 的 bytes/copyCount 总和；batch API 可以被 Runtime/Driver 分成多个 activity |
| UI 缩放后相邻事件看成一整块 | 放大到微秒级，或在 Events View 中逐条查看 |
| `nsys stats` 提示 SQLite 比 report 旧 | 加 `--force-export=true`，或重新运行 `nsys export --force-overwrite=true` |
| profile 吞吐低于裸跑 | profiler 有开销；用裸跑结果做性能基线，用 report 做归因 |
| 本机 nsys-ui 打不开远端 report | 安装与 collector 相同或更高版本的 Nsight Systems UI |

更多 CLI 和指标定义见 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
