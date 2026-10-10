# Cache Grid 性能工具

本目录提供 prefill/cache 测试、断点续测、重测、trace 补采、公式拟合和图表。
日常操作统一使用仓库根目录的 `./tools/cache_perf`；也可使用 `python3 tools/cache_perf`。
测试通过 Bazel target `//rtp_llm/test/perf_test:cache_grid_perf_test` 启动模型服务。

- 本文：配置、生成 grid、运行、分析与产物。
- [格式与执行参考](docs/reference.md)：必填字段、batch 调度、容量规则、计时口径、profiling 和迁移。

## 配置 profile

复制 [通用模板](config/local.example.jsonc) 或 [DSV4 模板](config/dsv4_local.example.jsonc)，
填写测试机器实际配置。模板数值只是示例，不代表其他模型的容量或缓存粒度。

| 配置位置 | 用途 |
|---|---|
| 顶层 `model_label` | 图表名称及公式文件名前缀 |
| `engine` | 模型类型、模型/tokenizer 路径、TP/DP/EP/world、引擎容量 |
| `engine.max_context_batch_size`、`engine.concurrency_limit` | 由 profile 配置，启动器不随 grid/batch/workspace 自动改写 |
| `engine_args` | 其他引擎参数 |
| `engine_env` | 模型/引擎环境变量 |
| `runtime_env` | 编译器、CUDA、动态库和 JIT 缓存路径 |
| `cache_grid.measure_runs`、`cache_grid.commit_tail_tokens` | 正式测量轮数、seed 提交尾部长度 |
| `cache_grid.cache_alignment` | 普通 grid 生成器的输入参数；运行时以 grid 的 `generator.cache_alignment` 为准 |
| `bazel` | executable、configs、output_base、test_timeout 等构建配置 |

profile 必须有 `schema_version: 1`。`.jsonc` 支持行注释和块注释，不支持尾逗号；
路径不展开 `$VAR`、`${VAR}` 或 `~`，请填写明确值。注释不影响指纹。
缓存复用粒度必须匹配实际模型和 CP 配置；例如 physical block512、CP8、sharded KV 的可见粒度是 4096。

环境优先级为 `--env KEY=VALUE > profile engine_env/runtime_env > 允许继承的外部环境`。
两个 profile 环境段不能重复定义同一变量。`runtime_env` 在启动 Bazel 前生效，
两类环境都通过 `--test_env` 传给测试进程；最终环境写入快照。
例如临时覆盖编译器：

```bash
./tools/cache_perf run --profile /path/local.jsonc --grid /path/grid.json \
  --result-dir /path/new-results --env "CC=${SMOKE_CC}" --env "CXX=${SMOKE_CXX}" \
  --env "CUDAHOSTCXX=${SMOKE_CXX}" --env "NVCC_PREPEND_FLAGS=-ccbin=${SMOKE_CXX}" --dry-run
```

Bazel 选项可用 `--config`（可重复）、`--output-base`、`--bazel` 覆盖。
直接使用 `bazelisk test --test_arg=--profile=...` 时，profile 只能影响测试进程，不能改变已启动 Bazel 的编译环境。

## 准备 grid

runner 只接受 `schema_version: 2` 的显式 `cases`，不展开采样模板或补齐必填字段。
格式及分组示例见[参考文档](docs/reference.md#grid-格式)。
[smoke grid](examples/dsv4_pro_prefill_smoke.json) 可作 DSV4 示例；运行前检查其长度和缓存粒度。

生成普通分层采样 grid：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid \
  --output /tmp/cache_grid.json --profile /path/local.jsonc \
  --min-input-len 8192 --max-input-len 65536 --cache-alignment 4096
```

固定 cache、递增 compute；每个 case 满足 `input_len = cache_len + compute_len`：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid \
  --output /tmp/fixed-cache-grid.json --grid-mode fixed-cache-sweep \
  --fixed-cache-len 32768,65536 --min-compute-len 4096 \
  --compute-step 32768,8192,4096 --max-input-len 262144 --cache-alignment 4096
```

步长必须唯一、从粗到细排列，每级只补新点。也可将 `--fixed-cache-len` 换为
`--random-cache-count 16 --min-cache-len 4096 --max-cache-len 131072 --seed 104729`。
最小 compute 必须容纳运行时的 commit tail。

生成多个 batch 文件；以下只是可见复用粒度为 1024、整批预算 65536 的示例：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid \
  --output-dir /tmp/batch_cache_grid --num-cases 100 --batch-sizes 1 2 4 \
  --workspace-tokens 65536 --max-batch-tokens 65536 \
  --min-input-tokens 2048 --max-input-tokens 49152 \
  --input-alignment 256 --cache-alignment 1024 --commit-tail-tokens 1024 --seed 20260917
```

每个 batch size 输出一个 JSON；改用 `--output /tmp/mixed.json` 可生成混合 batch 文件。
不指定 `--workspace-tokens` 则只使用普通 token 预算。
`--batch-input-limit B:TOKENS` 可单独约束某个 batch；`--kv-budget-tokens` 约束活跃请求与 seed 尾部的估算。
同参数和 seed 生成相同计划，不覆盖已有文件。采样和容量细节见[容量规则](docs/reference.md#容量与随机采样)。

## 运行与恢复

首次运行必须使用空/新结果目录。`--dry-run` 打印计划，不写结果、不启动 Bazel 或 GPU。
`--runs` 指每个 case 的正式测量轮数，不改变 batch size。

```bash
# 单 grid：测试完成后统一拟合和绘图。
./tools/cache_perf pipeline --profile /path/local.jsonc --grid /path/grid.json \
  --result-dir /path/new-results --runs 3

# 多 grid：顺序启动各 JSON 对应的服务，全部完成后统一分析。
./tools/cache_perf pipeline --profile /path/local.jsonc --grid-dir /tmp/batch_cache_grid \
  --result-root /path/new-batch-results --runs 3

# 只测试，不做后处理。
./tools/cache_perf run --profile /path/local.jsonc --grid /path/grid.json \
  --result-dir /path/test-only --runs 3

# 单 grid 续测；多 grid 则用 --result-root。
./tools/cache_perf pipeline --test-mode resume --result-dir /path/new-results
./tools/cache_perf pipeline --test-mode resume --result-root /path/new-batch-results

# 只续测、不分析。
./tools/cache_perf resume --result-dir /path/test-only

# 已有完整结果，只重新拟合和绘图。
./tools/cache_perf pipeline --skip-test --result-root /path/new-batch-results

# 部分结果仅预览，不拟合或声明可发布。
./tools/cache_perf pipeline --skip-test --partial --result-root /path/new-batch-results

# 完整后处理结束后，将整个结果目录归档并上传到指定 OSS 对象。
./tools/cache_perf pipeline --skip-test --result-root /path/new-batch-results \
  --oss-destination oss://bucket/path/new-batch-results.tar.gz
```

恢复读取冻结的 profile/grid/环境，不接受这些启动配置的覆盖。
多 grid 恢复会跳过已完成运行，有 checkpoint 的续测，未开始的按冻结配置启动。
运行失败但没有 checkpoint 时，应使用新目录重试。
`--skip-test` 不接受 profile/grid/runs/env 等启动覆盖项。

后处理默认包含所有 batch；`--batch-size` 是可选过滤器。
可用 `--estimator median|min|trimmed` 改变跨轮聚合方式，`--skip-fit` 只绘图，
`--cards` 覆盖 TPM 卡数；输出路径可用 `--formula-output-dir`、`--svg-output`、
`--cold-svg-output`、`--html-output` 指定。
测试失败或结果不完整时停止后处理；拟合门禁返回 3 时仍绘图，最终返回 3。
`--oss-destination` 也可用于首次运行和续测，必须给出以 `.tar.gz` 结尾的完整 OSS 对象路径。
测试与图表完成后，pipeline 在结果目录旁生成唯一命名的 tar.gz（包内以结果目录名为顶层），
用 `ossutil cp` 上传；拟合门禁返回 3 时仍上传拒绝报告，测试或后处理失败、部分结果预览时不上传。
上传失败会令 pipeline 报错，本地压缩包保留以便重试。上传状态和压缩包 SHA-256 记录在本地
`pipeline_summary.json`；压缩包内的该文件是上传前的后处理完成快照。

## 重测与 trace

```bash
./tools/cache_perf retest --result-dir /path/results --cases 0,1 --runs 3
./tools/cache_perf profile --result-dir /path/results --cases 0 --runs 1 --trace-timeout 180

# Nsight Systems；不指定 backend 时使用 Kineto。
./tools/cache_perf profile --result-dir /path/results --cases 0 --runs 1 \
  --profile-backend nsys --nsys-path /usr/local/cuda-12.6/bin/nsys
```

case ID 必须来自原 grid。重测和补采输出到 `cache_perf_replays/retest_<id>/` 或
`cache_perf_replays/profile_<id>/`，不覆盖原 checkpoint 或正式报告。
输入按原几何重新构造，并非原 token 序列逐字回放。
profile 只支持未分组 batch=1，不支持共享 seed；trace 耗时不用于正式拟合。
Kineto 输出 `timelines/`，nsys 输出 `nsys/*.nsys-rep`，`manifest.json` 记录采集状态。
采集边界和排障见[Nsight Systems 参考](docs/reference.md#nsight-systems)。

## 结果、公式与图表

所有公式和基准图使用服务端 `aux_info.first_token_cost_time`，结果字段为
`prefill_time_ms`，单位 ms；包含引擎等待，不包含输入分词和客户端通信。
每轮 batch 延迟取批内请求服务端耗时最大值，再跨轮聚合。
客户端 wall time 仅供诊断，不能替代服务端时间。

拟合仅使用 **restricted-symbolic**，需要 CPU PyTorch。保留每个请求的长度列表，
候选包括逐请求 `sum(...)`、batchSize 和 maxComputeTokens 等；
按请求分布划分训练/验证/测试，测试集不参与选式和重拟合。
逐 batch 验收、复用校验和 TPM 定义见[分析契约](docs/reference.md#分析契约)。

独立分析入口也直接读取原始结果文件，无需先转换为 CSV：

```bash
python3 -m rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit validate-inputs \
  --inputs /path/results/cache_grid_results.json
python3 -m rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit fit \
  --inputs /path/results/cache_grid_results.json --output-dir /path/formula
python3 -m rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit analyze-anomalies \
  --inputs /path/results/cache_grid_results.json --output-dir /path/anomalies
```

| 产物 | 内容 |
|---|---|
| `profile.snapshot.json`、`grid.snapshot.json` | 冻结输入 |
| `cache_perf_launch.json`、`test_info.json` | 启动配置、指纹与运行状态 |
| `cache_grid_results.json` | 原始测量结果和实际 run_config |
| `pipeline_summary.json` | 阶段状态、退出码及产物路径 |
| 结果目录旁的 `<结果目录名>-*.tar.gz` | 可选的完整归档，供 OSS 上传与本地复查 |
| `report/sources.json`、`report/audit.json` | 来源、配置一致性、排除原因、各 batch 数量 |
| `report/observations.json` | 请求分布、逐轮延迟、case 和来源，作为可审计产物 |
| 单 grid `formula/`；多 grid `report/formula/` | model.json、fit_report.json、predictions.csv、fit_gap.svg、公式文本 |
| `report/latency.interactive.html`、`report/tpm.interactive.html` | 离线交互图 |
| `prefill_3d.svg`、`prefill_cold_miss.svg` | 按 batch 着色的静态投影、零 cache 切片 |

交互图支持 batch 过滤、总长度/每请求均值、case/逐轮数据、延迟/输入 TPM/计算 TPM。
全部 batch 视图以 batch 为 Z 轴、指标着色；指定 batch 时以指标为 Z 轴。点选可查看请求分布与来源。
数据不足时拟合只输出审计及拒绝报告。默认公式名为 `<model_label>_prefill_formula.txt`，
key 为 `PREFILL_TIME_FORMULA`；profile 的可选 `chart` 字段和工具 CLI 可覆盖展示配置。

## 代码入口与维护

| 位置 | 职责 |
|---|---|
| [runner/](runner/) | grid 生成、启动、快照、观测读取和 pipeline |
| [formula/prefill_formula_fit.py](formula/prefill_formula_fit.py) | 数据校验、拟合、误差报告、异常分析 |
| [formula/restricted_symbolic_fit.py](formula/restricted_symbolic_fit.py) | 候选库、选式和系数拟合 |
| [plot/](plot/) | 统一交互图与独立静态/batch 图 |
| [config/](config/) | profile 解析、拓扑、配置模板 |
| [examples/](examples/) | 显式 grid 示例 |
| [tests/](tests/) | Python 与 HTML 交互回归 |
| [../batch_decode_test.py](../batch_decode_test.py) | 共享测试入口、容量配置、模型服务生命周期 |

Bazel 定义集中在 [../BUILD](../BUILD)，不要为本目录新增子包边界。

```bash
./tools/cache_perf --help
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid --help
python3 -m rtp_llm.test.perf_test.cache_grid.runner.generate_random_batch_grid --help
python3 -m unittest discover -s rtp_llm/test/perf_test/cache_grid/tests -p '*_test.py'
python3 -m unittest rtp_llm.test.perf_test.batch_decode_test_test
```
