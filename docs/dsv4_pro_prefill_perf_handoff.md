# DeepSeek-V4-Pro Prefill 性能测试交接

这份手册给接手测试的同事使用。它描述的是当前仓库里的单机 DSV4-Pro prefill 测试，不是线上压测脚本。测试前先确认模型目录、GPU 和代码版本，三者有一个不对就不要启动。

## 0. 最短执行路径

先准备一个**本地、容器内可见**的模型目录，再按下面顺序操作：

```bash
# 1) 确认代码版本（工作树有未提交改动时不要强行切分支）
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git branch --show-current
git status --short

# 2) 确认模型和 8 张 GPU
export MODEL_DIR=/data1/serina.wzq/DeepSeek-V4-Pro
test -r "$MODEL_DIR/config.json" && test -r "$MODEL_DIR/tokenizer_config.json"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# 3) 先只跑单元测试，再启动手工 perf target
bazelisk test //rtp_llm/test/perf_test:batch_decode_test_test \
  --config=cuda13 --config=sm10x --test_output=errors
# 具体的 DSV4 target 见第 6 节；target 名 dsv4_pro_prefill_handoff
```

`dsv4_pro_prefill_handoff` 是交接手册建议的新 target 名称，仓库现有 BUILD 不会自动生成它。需要先复制现有 DSV4-Pro prefill stanza，并按第 5 节替换参数；不要直接运行旧的 64K profile target 当作 1M 基线。

## 1. 测试口径

当前标准入口是 `rtp_llm/test/perf_test/batch_decode_test.py`，通过 `GridRunner` 逐个执行 `batch_size × input_len`。本手册的基线口径是：

| 项目 | 基线 |
|---|---|
| 模型 | DeepSeek-V4-Pro |
| 阶段 | prefill only，`--partial=2` |
| GPU | 8 卡，CUDA 13.2，SM100/L20D 运行环境 |
| 并行 | `tp=8, dp=1, ep=8, world=8` |
| CP | `ALL_GATHER` |
| KV cache | FP8，复用开启 |
| batch | `1`（本地 context batch 也设为 `1`） |
| 输出 | 每个请求 1 token；prefill 时间就是首 token 前的计算时间 |
| 测量 | 建议每个 geometry 预热 1 次、正式测量 3 次；正式测量不要开启 profiler |
| 1M 边界 | `input_len=1,048,575`、`decode_test_length=1`、`max_seq_len=1,048,576` |

这里的 `input_len` 是完整输入长度，不是新计算 token 数。带前缀缓存的场景需要额外记录 `observed_cache_len`，新计算量是 `input_len - observed_cache_len`。

标准 `GridRunner` 只扫描 batch 和 input length，本身没有 `--cache_len` 参数。因此，不能把普通 grid 的结果当成 cache-hit 结果；cache-hit 测试必须由能执行“seed → hit request → 校验 reuse_len”的专用 runner 完成，并把实际 `reuse_len` 写入结果。

## 2. 代码版本

当前代码分成两个仓库：

```text
GitHub inner repo: /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
branch: feat/dsv4_on_dev

GitLab outer repo: /data0/luoli.hn/work/rtp_llm_4
branch: develop/wangyin_ds_v4_20260424
```

常用检查：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git status --short
git branch --show-current
git rev-parse HEAD
```

不要在两个仓库之间复制源码，也不要用 `git clean -fdx` 清理外层工作区；外层目录可能有其他任务留下的构建缓存和工作树。

如果两个仓库都没有本地改动，切换版本可以这样做：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
git switch feat/dsv4_on_dev

cd /data0/luoli.hn/work/rtp_llm_4
git switch develop/wangyin_ds_v4_20260424
```

切换前后都记录 `git rev-parse HEAD`。如果 `git status --short` 有输出，先让负责人确认这些改动是否属于本次测试；不要为了切分支删除它们。

## 3. 运行环境

在容器内确认以下命令可用：

```bash
/opt/conda310/bin/python --version       # Python 3.10
/opt/conda310/bin/python - <<'PY'
import torch
print(torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_mode --format=csv
bazelisk --version
/opt/rh/gcc-toolset-12/root/usr/bin/g++ --version
/usr/local/cuda/bin/nvcc --version
```

L20D/CUDA 13 的构建配置是：

```bash
--config=cuda13 --config=sm10x
```

`.bazelrc` 已为 `cuda13` 设置 GCC 12、CUDA 13.2 和 `TF_CUDA_COMPUTE_CAPABILITIES`。如果是从一个没有加载 `.bazelrc` 的 shell 启动，至少补上：

```bash
export DG_JIT_CPP_STANDARD=20
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export CUDAHOSTCXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export NVCC_PREPEND_FLAGS=-ccbin=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:/opt/conda310/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/cuda-13.2/lib64:/usr/local/cuda/lib64:/opt/conda310/lib:/usr/lib64:/usr/local/lib64:/usr/lib:/lib64
```

`cuda13` 分支使用 Triton 3.6；除非目标镜像明确要求，不要从旧 CUDA 12 测试复制 `TRITON_PTXAS_PATH`。

DSV4 的常用运行时开关（都能在仓库的 server args 或 DSV4 attention 代码中找到）如下。它们是实验配置，不要在不同批次之间悄悄改变：

| 环境变量 | 建议值 | 用途 |
|---|---:|---|
| `WORLD_SIZE` | `8` | 8 个 rank 的进程拓扑 |
| `CP_ROTATE_METHOD` | `ALL_GATHER` | CP token 旋转方式 |
| `PREFILL_CP_KV_CACHE_SHARDED` | `1` | CP prefill KV 分片 |
| `FP8_KV_CACHE` | `1` | FP8 KV cache |
| `REUSE_CACHE` | `1` | 允许请求复用前缀 KV |
| `ENABLE_DEVICE_CACHE` | `1` | 启用设备侧 cache |
| `DSV4_USE_MEGA_MOE` | `1` | DSV4 MegaMoE 路径 |
| `DSV4_CHUNK_TOKENS` | `8192` | DSV4 prefill 分块大小 |
| `DSV4_PREFILL_CP_OVERLAP` | `0/1` | 是否启用 CP prefill overlap；基线必须固定 |
| `WARM_UP` | `1` | 开启服务 warm-up |

`MAX_SEQ_LEN`、`MAX_BATCH_TOKENS_SIZE`、`MAX_CONTEXT_BATCH_SIZE`、`FP8_KV_CACHE` 等也有对应 server args。为了避免环境变量和 CLI 产生两套口径，长度、拓扑、batch、block size 优先写在 target 的 `args` 中；缓存和 JIT 开关再通过 `env` 传入。

## 4. 模型和缓存目录

模型和 tokenizer 必须在**运行测试的容器内**可读。推荐把两者设成同一个本地模型目录：

```bash
export MODEL_DIR=/data1/serina.wzq/DeepSeek-V4-Pro
test -r "$MODEL_DIR/config.json"
test -r "$MODEL_DIR/tokenizer_config.json"
find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' | sort | head
```

如果模型目录在宿主机而容器内不存在，先做只读 bind mount 或使用容器已有的 NAS 路径；不要让 `start_server` 在测试过程中临时下载 1M 模型权重。使用 Hub repo id 只有 perf 入口会尝试 ModelScope/HuggingFace 解析，普通 `start_server` 不会自动下载。

JIT/cache 建议放在持久化工作目录：

```bash
export HIPPO_APP_WORKDIR=/ssd/1/dsv4_perf_work
mkdir -p "$HIPPO_APP_WORKDIR"/{jit_cache,triton_cache,tilelang_cache,results,logs}
export DG_JIT_CACHE_DIR=$HIPPO_APP_WORKDIR/jit_cache
export TRITON_CACHE_DIR=$HIPPO_APP_WORKDIR/triton_cache
export TILELANG_CACHE_DIR=$HIPPO_APP_WORKDIR/tilelang_cache
```

远端 JIT/OSS 凭证只能由 secret manager 或当前 shell 注入，不能写进 BUILD、脚本、日志或提交记录。交接文件不包含任何 access key。

如果模型还没有落盘，先由存储或模型平台管理员把权重同步到 `MODEL_DIR`，并在容器内做文件可读性检查。不要在 Bazel 测试过程中执行下载；权重下载失败和服务启动失败要分开处理。

## 5. 推荐的 DSV4-Pro prefill 参数

这些参数是服务参数，不是 `batch_decode_test.py` 自己实现的逻辑；perf 入口会把未消费的参数转发给 `start_server`。

```text
--model_type deepseek_v4
--checkpoint_path $MODEL_DIR
--tokenizer_path $MODEL_DIR
--batch_size 1
--input_len 256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,786432,1048575
--partial 2
--decode_test_length 1
--max_seq_len 1048576
--max_batch_tokens_size 1048576
--max_context_batch_size 1
--concurrency_limit 1
--tp_size 8
--dp_size 1
--ep_size 8
--world_size 8
--cp_rotate_method ALL_GATHER
--seq_size_per_block 512
--kernel_seq_size_per_block 128
--fp8_kv_cache 1
--reuse_cache 1
--enable_device_cache 1
--use_deepep_moe 1
--use_deepep_low_latency 0
--act_type BF16
--load_method fastsafetensors
--enable_cuda_graph 0
--reserver_runtime_mem_mb 81920
```

`decode_test_length` 虽然在 prefill 请求中仍只发 1 个 token，但它会参与服务最大长度计算：

```text
effective_max_seq_len = max(max_seq_len, max(input_len) + decode_test_length)
```

因此 1M 边界使用 `1,048,575 + 1 = 1,048,576`。如果把 `decode_test_length` 写成 2，服务实际会申请到 1,048,577，不应再把它称为严格 1M 配置。

同一个参数不要同时在 CLI 和环境变量里写两份。建议模型、并行度、block size 等放 CLI；缓存开关和 JIT 开关放环境变量。`GridRunner` 在 grid 模式会用 `max(batch_size)` 设置服务并发，`--concurrency_limit` 主要对 distribution 模式生效。

本手册的基线是 batch=1，所以 `--concurrency_limit=1` 是为了保持“一个 geometry 一次 forward”的可比性。如果要测并发吞吐，另建 target，显式记录 `batch_size`、`max_context_batch_size` 和实际 server concurrency；不要把两类结果合并拟合。

## 6. 可复现的 Bazel 命令

先做无 GPU 的 Python 检查：

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
python3 -m py_compile \
  rtp_llm/test/perf_test/batch_decode_test.py \
  rtp_llm/test/perf_test/batch_perf_impl.py \
  rtp_llm/test/perf_test/grid_runner.py
bazelisk test //rtp_llm/test/perf_test:batch_decode_test_test \
  --config=cuda13 --config=sm10x \
  --test_output=errors
```

正式运行时，建议从 BUILD 中复制 `v4_pro_cp8_ep8_prefill_64k_perf` 成一个新的手工 target，并把第 5 节的参数写成该 target 的唯一 `args`。这样不会遇到重复参数被旧值抢先解析的问题。当前仓库已有的 target 使用的是 6 个示例长度、`decode_test_length=2`、`seq_size_per_block=256`，适合 profile smoke，不是严格 1M handoff 基线。

运行新 target 的命令形状如下：

```bash
export BAZELISK_HOME=/ssd/1/dsv4_perf_work/bazelisk-home
export RESULT_DIR=/ssd/1/dsv4_perf_work/results/prefill_$(date +%Y%m%d_%H%M%S)

bazelisk test //rtp_llm/test/perf_test:dsv4_pro_prefill_handoff \
  --config=cuda13 --config=sm10x \
  --test_timeout=345600 \
  --test_output=streamed \
  --nocache_test_results \
  --test_arg=--result_dir="$RESULT_DIR" \
  --test_env=PERF_GRID_WARMUP_RUNS=1 \
  --test_env=PERF_FORMAL_WARMUP_RUNS=1 \
  --test_env=PERF_MEASURE_RUNS=3 \
  --test_env=PERF_PROFILE_RUNS=0 \
  --test_env=TOKENIZERS_PARALLELISM=false
```

注意：`--test_arg=--checkpoint_path=...` 不要直接追加到已有 target。`batch_decode_test.py` 会把 checkpoint/tokenizer 留在转发参数中，路径提取函数按第一次出现的值解析，重复写法可能仍然使用 BUILD 里的旧路径。需要换模型时，修改新 target 的唯一 `--checkpoint_path` 和 `--tokenizer_path`，再重新 build/test。

启动前把最终 `args` 保存到结果目录旁的 `argv.txt`，并检查以下值只出现一次：`checkpoint_path`、`tokenizer_path`、`max_seq_len`、`max_batch_tokens_size`、`decode_test_length`、`tp_size`、`dp_size`、`ep_size`、`world_size`。这一步能避免“日志里写的是 1M，服务实际申请了 1,048,577”之类的隐性偏差。

## 7. 测量次数和结果文件

`GridRunner` 的每个 case 大致经过以下阶段：

1. 预热：处理 JIT 或首次分配的抖动；
2. 正式测量：由 `PERF_MEASURE_RUNS` 控制，建议 3 次；
3. profiler：默认存在，但会影响 RT。做性能拟合时用 `PERF_PROFILE_RUNS=0` 关闭。

结果目录通常包含：

```text
$RESULT_DIR/
├── Prefill_Result.json       # 逐 case 指标
├── test_info.json             # 本次配置
└── timelines/                 # 只有开启 profiler 时才有

$TEST_UNDECLARED_OUTPUTS_DIR/main_logs/process.log  # 引擎日志
```

标准 grid JSON 的核心字段是 `input_len`、`batch_size`、`success_rate`、`avg_wait_time`、`avg_prefill_time`。cache matrix 应额外保存 `cache_len_requested`、`cache_len_observed`、`runs[]`、`success_runs` 和 `status`。

cache 结果必须以 `cache_len_observed` 为准。请求的 cache 长度可能因物理 block 对齐而被向下取整；请求了正 cache 但实际 `reuse_len=0` 时，不得把它标成 cache hit。

建议每个结果 JSON 顶层同时记录：`model_id`、代码 commit、完整 argv、关键 env 的脱敏 hash、GPU 型号、`max_seq_len`、`max_batch_tokens_size`、测量轮数、grid 文件 SHA。没有这些 provenance，结果只能用于临时排查，不能拿去拟合或横向比较。

### 专用 cache runner 的实际位置

这一点不能靠默认入口推断：标准 `GridRunner` 没有 cache 维度。本轮带 cache 的临时 runner 实际放在：

```text
/tmp/cache_grid_runner.remote.py
```

它提供 `PrefixPromptFactory` 和 `CacheGridRunner`，流程是“写入唯一 prefix → 发起带相同 prefix 的 continuation → 读取 `aux_info.reuse_len` → 保存三轮 RT”。配套的临时入口是：

```text
/tmp/batch_decode_test.remote.py
```

其中 `--cache_grid_json` 指向显式的 `input_len × cache_len` case 文件，`--cache_measure_runs=3` 控制每个 geometry 的测量次数，结果文件为：

```text
$RESULT_DIR/cache_grid_results.json
```

需要特别说明：这两个 `/tmp` 文件不是当前 Git 分支里的受版本控制文件，机器重启或清理临时目录后可能消失。因此它们只能解释历史结果，不能作为正式交接依赖。正式交接前应把 runner 和入口纳入目标分支，并为其补一个 Bazel target；在此之前，接手人不能只按本手册第 6 节的标准 target 完成 cache-hit 测试。

## 8. 运行时监控和停机

启动后至少每 30 秒看一次：

```bash
tail -F "$TEST_UNDECLARED_OUTPUTS_DIR/main_logs/process.log"
nvidia-smi
```

重点看：

- `All backend ranks started` / health ready；
- 8 个 rank 是否都存在；
- GPU 显存、util 是否在变化；
- `OOM`, `Traceback`, `FATAL`, `PROCESS_EXIT`, `GET_HOST_FAILED`；
- 结果 JSON 的 case 数是否增长。

不要因为某一个 case 很慢就重启服务。服务应覆盖整个 grid，case 之间只切换 scheduler 配置。只有明确确认属于自己的测试进程、并且服务已经无法自行退出时，才按进程树做有范围的清理；不要使用无目标的 `kill -9`。

## 9. 拟合和画图

### 9.1 输入校验和拟合

拟合脚本只会使用通过校验的 geometry：三次 measurement 都成功、output length 为 1、prefill RT 有限、observed reuse 在三次中一致。它会把同一物理 geometry 的重复记录合并成 median。

```bash
cd /data0/luoli.hn/work/rtp_llm_4/dsv4-cache-affinity-1OIKUM
FIT=rtp_llm/test/perf_test/deepseek_v4_prefill_formula_fit.py
DATA=/path/to/cache_grid_results.json
OUT=/path/to/formula_output

python3 "$FIT" validate-inputs \
  --inputs "$DATA" \
  --batch-size 1 \
  --report "$OUT/input_validation.json"

python3 "$FIT" fit \
  --inputs "$DATA" \
  --batch-size 1 \
  --objective mae \
  --output-dir "$OUT"
```

输出：

```text
$OUT/input_audit.json
$OUT/fit_report.json
$OUT/deepseek_v4_prefill_formula.txt
$OUT/predictions.csv
```

`--objective mae` 是绝对误差目标。脚本返回码为 0 表示生产门禁通过，3 表示公式已经生成但 MAPE/p95/max 门禁没有通过，2 表示输入无效或样本不足。返回 3 不能当作命令失败，也不能把未通过的公式直接上线。

当前 fitter 导出的公式只使用 `tokens`、`hitCacheTokens` 和 `+ - * / ( )`。公式适用于固定 batch 和实际测量范围，不自动外推 batch scaling。

### 9.2 三维图

```bash
CHART=rtp_llm/test/perf_test/generate_prefill_3d_chart.py
python3 "$CHART" \
  --input "$DATA" \
  --output "$OUT/deepseek_v4_prefill_3d.svg" \
  --batch-size 1
```

当前图的坐标约定固定为：

```text
X = measured TTFT / prefill RT (ms)
Y = observed cached tokens
Z = uncached compute tokens = input_len - observed_cache_len
```

浅灰点是全部可用 geometry；实线是近固定 cache 的中位数趋势，虚线是近固定 compute 的中位数趋势。颜色只区分趋势线，不表示 RT 数值。

## 10. 出问题时先查这几项

| 现象 | 先查什么 |
|---|---|
| 服务启动后立刻退出 | `process.log`、模型目录、`model_type`、CUDA/GCC 版本 |
| 1M case 被拒绝 | `max(input_len)+decode_test_length` 是否超过 `max_seq_len` |
| cache 命中为 0 | seed 是否先成功、prefix 是否完全相同、block 对齐是否改变 observed reuse |
| 结果行数少 | 是否有失败请求、重复物理 geometry、正 cache 实际 reuse=0 |
| 每个 case 很慢 | 是否误开 profiler、是否每个 case 都重启 server、JIT cache 是否持久化 |
| 公式在 FlexLB 解析失败 | 是否混入 `sum`、`max`、`batchSize`、`computeTokens` 或 Python 语法 |

交接时至少提供：代码 commit、模型目录、完整 argv/env、GPU 型号、结果 JSON、`process.log`、拟合报告和 SHA256。不要只交一张截图或一行平均 RT。
