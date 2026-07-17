# FlexLB Master + Mock Engine 性能测试 Handoff 手册

本文用于交接 FlexLB Master 的 batch 调度性能测试。目标是让接手人能够复现测试、逐级寻找容量拐点，并判断瓶颈在发压端、FlexLB Master 还是 mock engine。

本次基准结果见 [FlexLB Master + Mock Engine Batch 性能报告](flexlb-master-mock-engine-performance-20260717.md)。

## 1. 测试目标和边界

测试链路：

`真实 trace 到达时间 -> load client -> Schedule RPC -> Master 路由 -> fixed-window 凑批 -> EnqueueBatch -> engine ACK -> Schedule 返回`

本手册只测 Master 调度能力：

- 必须使用 `SCHEDULE_MODE=batch`、`SCHEDULE_ONLY=1`、`FLEXLB_BATCH_ALGORITHM=fixed_window` 和 `FLEXLB_BATCH_FIXED_WAIT_MS=10`。
- 不调用 `FetchResponse`。Fetch 是 frontend 的后续动作，不属于 Master Schedule 性能。
- 吞吐以 Master 服务端的 `server_arrival_qps` 为准。
- 延迟以 Master 服务端的 `schedule_latency_ms` 为准，不以 client RTT 作为最终报告口径。
- mock engine 必须有足够余量，不能让 mock 的 CPU、线程或队列先成为瓶颈。

一次有效档位必须同时满足：

1. `error_count == 0`。
2. `server_latency.json` 中 `arrival_count == completion_count`。
3. FlexLB、mock engine 和 load client 全程存活，无 OOM、拒绝执行或无可用 worker 错误。
4. 使用同一份 trace、Master 配置、mock 性能模型、engine 数量和代码版本进行横向比较。

## 2. 三类 worker 不要混淆

| 参数 | 所属组件 | 作用 | 测试原则 |
|---|---|---|---|
| `LOAD_CLIENT_WORKERS` | 发压端 | load client 进程数，影响 channel、timer 并行度和微突发形态 | 发压不足时增加；比较 Master 配置时固定 |
| `SCHEDULE_WORKER_SIZE` | FlexLB Master | Master 内部 Schedule worker 数 | 本轮基线固定为 16 |
| `JAVA_MOCK_EVENT_LOOP_THREADS` | Java mock engine | mock gRPC event loop 数 | 本轮基线固定为 32 |

报告中的“worker 数”如果没有特别说明，指 `LOAD_CLIENT_WORKERS`，不是 Master 内部 worker。

## 3. 环境准备

### 3.1 运行位置

在具备 Java 21、Python 3、Maven wrapper 和足够 CPU/内存的 Linux 测试机或容器中执行。参考环境是在仓库的 `github-opensource` 子仓中运行：

```bash
export RTP_LLM_OPEN_SOURCE=/path/to/RTP-LLM/github-opensource
source ~/.bashrc
cd "$RTP_LLM_OPEN_SOURCE"
export JAVA21_HOME="${JAVA21_HOME:-$HOME/java21}"
export JAVA_HOME="$JAVA21_HOME"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
python3 --version
```

注意：部分机器的 `~/.bashrc` 会修改当前目录，所以 `source` 后必须重新 `cd` 到 `github-opensource`。

### 3.2 资源和端口

750 个 prefill 加 500 个 decode engine 会使用从 `MOCK_BASE_GRPC_PORT` 开始的 1250 个连续 gRPC 端口。默认基址为 61000，范围为 61000 至 62249。FlexLB 默认使用：

- HTTP：7001
- management：7002
- gRPC：7003，即 HTTP 端口加 2

开始前检查：

```bash
ulimit -n
lsof -i:7001 -i:7002 -i:7003
lsof -nP -iTCP:61000-62249 -sTCP:LISTEN | head
```

建议文件描述符上限至少为 65535。正式记录结果前还应记录机器 CPU 型号、逻辑核数、内存、容器 CPU/memory limit 和代码 commit。

### 3.3 输入文件

```bash
EVAL_DIR="$PWD/rtp_llm/flexlb/tools/online_eval"
test -s "$EVAL_DIR/data/online_logs/trace_30min.jsonl"
test -s "$EVAL_DIR/data/config/master_fixed_window.json"
test -s "$EVAL_DIR/data/performance/dsv4_flash_performance.fast_ab.json"
```

三份文件的作用：

- `trace_30min.jsonl`：请求到达时间、输入/输出 token 长度和 cache key。
- `master_fixed_window.json`：Master 进程配置，也是 `PREFILL_TIME_FORMULA` 的唯一来源。
- `dsv4_flash_performance.fast_ab.json`：mock 的 decode batch 曲线和 `sleep_scale`。

## 4. 编译和快速校验

```bash
cd "$RTP_LLM_OPEN_SOURCE/rtp_llm/flexlb"
./mvnw clean package -DskipTests -P '!internal'

test -s flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar
test -s flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar
bash -n tools/online_eval/run_online_eval.sh
python3 -m py_compile tools/online_eval/flexlb_load_client.py
```

重新编译后再压测，避免代码和旧 jar 不一致。完整验证可以执行：

```bash
./mvnw test -P '!internal'
python3 -m unittest discover -s tools/online_eval/tests
```

## 5. mock engine 的模拟方式

基准必须使用 `MOCK_ENGINE_IMPL=java`，不使用 Python mock 做高 QPS 容量结论。

Java mock 不是固定延迟返回。它从每个请求读取输入 token、输出 token 和 cache key，并维护每个 engine 的运行任务、等待任务、KV 使用量和 cache 命中。Prefill 时间优先使用 `master_fixed_window.json` 中和 Master 相同的 `PREFILL_TIME_FORMULA`：

```text
prefill_ms = formula(batchSize, inputTokens, computeTokens,
                     hitCacheTokens, hasHitCache) * prefill.scale * sleep_scale
```

Decode 时间由输出长度和实时 active batch 决定：

```text
decode_ms = output_len * interpolate(step_ms_by_batch, active_batch)
            * decode.scale * sleep_scale
```

`fast_ab` 模型的 `sleep_scale=0.1` 用于给 mock 留出吞吐余量，避免下面的模拟 engine 先卡住 Master。它适合隔离 Master 调度上限，不代表真实 LLM engine 的绝对 TTFT 或端到端容量。需要评估真实 engine 容量时，必须换成真实测得的 performance model，并重新确认 mock 不成为瓶颈。

## 6. 流量发送方式

`REPLAY_SPEED` 不会把一整秒请求瞬间打完。load client 按 trace 的毫秒时间戳发送，每条请求的目标时间为：

```text
send_due = (request_ts - first_trace_ts) / REPLAY_SPEED
```

因此它保留原始请求间隔和微突发，只对时间轴等比例压缩。`LOOP=1` 时会重复 trace，直到 `DURATION_S` 到期。

### 6.1 先校准，再阶梯加压

不要直接从 10K 开始。先用约 100 QPS 校准。当前 trace 的经验起点是 `REPLAY_SPEED=13`，但不同 trace 必须按实测修正：

```text
next_speed = current_speed * target_master_qps / measured_server_arrival_qps
```

建议容量阶梯：100、250、500、1000、2000、3000、5000、8000、10000 QPS。每档至少运行 60 秒；临近拐点时延长到 180 至 300 秒并重复 3 次。

当前 trace 可用的初始 speed 估值如下，最终以 `server_arrival_qps` 校准：

| 目标 Master QPS | 初始 Replay speed | 建议 load client workers |
|---:|---:|---:|
| 100 | 13 | 1 |
| 250 | 32 | 1 |
| 500 | 64 | 1 |
| 1000 | 125 | 1 |
| 2000 | 250 | 1 或 2 |
| 3000 | 375 | 2 |
| 5000 | 650 | 4 |
| 8000 | 1000 | 8 |
| 10000 | 1250 | 8 |

load worker 变化会改变发压端微突发形态。要比较不同 Master 代码或 `SCHEDULE_WORKER_SIZE`，必须固定 `LOAD_CLIENT_WORKERS`；要证明发压端不是瓶颈，可以在相同目标 QPS 下补做不同 load worker 的对照组。

## 7. 运行一个基准档位

下面命令运行约 100 QPS 的 Master-only batch 测试：

```bash
cd "$RTP_LLM_OPEN_SOURCE/rtp_llm/flexlb/tools/online_eval"

RUN_ID="handoff_$(date +%Y%m%d_%H%M%S)_w1_s13" \
N_PREFILL=750 \
N_DECODE=500 \
MOCK_ENGINE_IMPL=java \
MOCK_BASE_GRPC_PORT=61000 \
JAVA_MOCK_EVENT_LOOP_THREADS=32 \
PERFORMANCE_FILE="$PWD/data/performance/dsv4_flash_performance.fast_ab.json" \
PROCESS_CONFIG_FILE="$PWD/data/config/master_fixed_window.json" \
SCHEDULE_MODE=batch \
SCHEDULE_ONLY=1 \
FLEXLB_BATCH_ALGORITHM=fixed_window \
FLEXLB_BATCH_FIXED_WAIT_MS=10 \
SCHEDULE_WORKER_SIZE=16 \
LOAD_CLIENT_WORKERS=1 \
REPLAY_SPEED=13 \
DURATION_S=60 \
LOOP=1 \
LIMIT=999999999 \
MAX_CONCURRENCY=131072 \
FLEXLB_WARMUP_SECONDS=10 \
FLEXLB_JVM_HEAP_SIZE=32g \
FLEXLB_JVM_XMS=32g \
FLEXLB_JVM_XMX=32g \
JFR_DURATION=120s \
bash run_online_eval.sh
```

同时设置 `FLEXLB_JVM_HEAP_SIZE` 和 `FLEXLB_JVM_XMS/XMX`：前者进入 FlexLB 运行配置，后两者确保启动命令实际使用 32 GiB 堆。

脚本会按顺序启动 Java mock、等待全部 endpoint 被 Master 发现、启动 FlexLB、等待 10 秒预热、执行发压、采集服务端延迟和 JFR，最后清理本次进程。

## 8. 批量执行容量矩阵

先完成 100 QPS 校准。确认无错误后，可以使用下面的矩阵。任何一档出现错误或 P95/P99 突增，都应暂停继续加压，先分析该档及前一档。

```bash
cd "$RTP_LLM_OPEN_SOURCE/rtp_llm/flexlb/tools/online_eval"

BASE_ENV=(
  N_PREFILL=750
  N_DECODE=500
  MOCK_ENGINE_IMPL=java
  MOCK_BASE_GRPC_PORT=61000
  JAVA_MOCK_EVENT_LOOP_THREADS=32
  "PERFORMANCE_FILE=$PWD/data/performance/dsv4_flash_performance.fast_ab.json"
  "PROCESS_CONFIG_FILE=$PWD/data/config/master_fixed_window.json"
  SCHEDULE_MODE=batch
  SCHEDULE_ONLY=1
  FLEXLB_BATCH_ALGORITHM=fixed_window
  FLEXLB_BATCH_FIXED_WAIT_MS=10
  SCHEDULE_WORKER_SIZE=16
  DURATION_S=60
  LOOP=1
  LIMIT=999999999
  MAX_CONCURRENCY=131072
  FLEXLB_WARMUP_SECONDS=10
  FLEXLB_JVM_HEAP_SIZE=32g
  FLEXLB_JVM_XMS=32g
  FLEXLB_JVM_XMX=32g
  JFR_DURATION=120s
)

run_case() {
  local workers="$1"
  local speed="$2"
  local target="$3"
  local run_id="handoff_$(date +%Y%m%d_%H%M%S)_qps${target}_w${workers}_s${speed}"
  env "${BASE_ENV[@]}" \
    "RUN_ID=$run_id" \
    "LOAD_CLIENT_WORKERS=$workers" \
    "REPLAY_SPEED=$speed" \
    bash run_online_eval.sh
}

run_case 1 13 100
run_case 1 32 250
run_case 1 64 500
run_case 1 125 1000
run_case 2 250 2000
run_case 2 375 3000
run_case 4 650 5000
run_case 8 1000 8000
run_case 8 1250 10000
```

不要同时并行跑多个 case。每个 case 都会占用相同端口，并且并行运行会污染 CPU、网络和延迟数据。

## 9. 读取和校验结果

每个 run 的关键文件：

| 文件 | 用途 |
|---|---|
| `load_client/summary.json` | 合并后的 QPS、错误数和 Master 服务端延迟 |
| `load_client/server_latency.json` | Master arrival/completion 计数和各阶段延迟原始值 |
| `load_client/shard_*/summary.json` | 多 load worker 时每个发压分片的数据 |
| `flexlb_profile.jfr` | FlexLB JVM profile |
| `flexlb.log` | Master 日志、拒绝执行、无可用 worker、GC/OOM 线索 |
| `mock_engine.log` | mock 的 RPC 数、prefill pending 和 decode running |
| `flexlb_env.txt` | 本次 engine endpoint 和启动环境 |

用以下命令打印单个 run 的正式报告字段并做计数校验：

```bash
RUN_DIR="$PWD/run/<run_id>"
python3 - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
summary = json.loads((run / "load_client/summary.json").read_text())
server = json.loads((run / "load_client/server_latency.json").read_text())
latency = summary["schedule_latency_ms"]

print(f"load_workers={summary.get('load_client_workers', 1)}")
print(f"actual_send_qps={summary.get('actual_send_qps', summary.get('send_qps'))}")
print(f"master_arrival_qps={server['arrival_qps']}")
print(f"master_completion_qps={server['completion_qps']}")
print(f"requests={server['arrival_count']} completed={server['completion_count']}")
print(f"errors={summary.get('error_count', summary.get('errors', 0))}")
print("avg={mean} p50={p50} p90={p90} p95={p95} p99={p99} max={max}".format(**latency))

if server["arrival_count"] != server["completion_count"]:
    raise SystemExit("INVALID: Master arrival/completion count mismatch")
if summary.get("error_count", summary.get("errors", 0)) != 0:
    raise SystemExit("INVALID: Schedule errors are not zero")
PY
```

`server_completion_qps` 可能略高于 `server_arrival_qps`，因为两者使用各自首尾事件作为统计窗口。完整性判断应比较 count，不应要求两个 QPS 完全相等。

## 10. Master 服务端延迟分段

`schedule_latency_ms` 是 RPC 到达 Master 到 Schedule 完成的总耗时。`server_stage_latency_ms` 包含：

| 阶段 | 含义 | 上升时优先排查 |
|---|---|---|
| `grpc_queue_ms` | 请求进入 gRPC 后到业务处理开始 | gRPC executor、event loop、CPU 饱和、GC |
| `route_submit_ms` | 路由计算和提交 batcher | 调度算法、cache index、锁竞争、对象分配 |
| `batch_wait_ms` | fixed-window 等待和凑批 | batch window、batcher 队列和调度线程 |
| `dispatch_ack_ms` | Master 发 EnqueueBatch 到 engine ACK | outbound gRPC executor、连接、mock ACK 能力 |
| `ack_response_ms` | ACK 后完成 Schedule response | response executor、event loop、GC |

可以直接读取或重置服务端采样：

```bash
curl -s http://127.0.0.1:7001/rtp_llm/server_latency | python3 -m json.tool
curl -s -X POST http://127.0.0.1:7001/rtp_llm/server_latency/reset
```

## 11. 如何判定瓶颈属于谁

### 11.1 发压端不足

典型现象：

- `actual_send_qps` 明显低于按当前 speed 预期的 QPS。
- Master `arrival_qps` 跟随 `actual_send_qps`，Master 各阶段延迟仍低。
- 增加 `LOAD_CLIENT_WORKERS` 后 Master QPS 上升，Master 配置不变。

处理：先增加 load worker 或拆到独立发压机，再讨论 Master 上限。不要把 load client 的单进程 timer/channel 上限算成 FlexLB 上限。

### 11.2 mock engine 或下游链路不足

典型现象：

- `dispatch_ack_ms` 明显上升，而 `route_submit_ms` 和 `batch_wait_ms` 正常。
- `mock_engine.log` 中 `prefill_pending` 持续增长，或 mock CPU/event loop 饱和。
- 出现 `engine-grpc-client-executor` 拒绝执行、连接失败或 ACK timeout。

处理：先确认使用 Java mock 和 `fast_ab`，检查 mock CPU、`JAVA_MOCK_EVENT_LOOP_THREADS`、outbound gRPC executor 和连接稳定性。若只有冷启动出现，增加预热并复测；若稳态持续出现，才属于真实容量问题。

### 11.3 FlexLB Master 自身不足

典型现象：

- `actual_send_qps` 足够，mock pending 有界，但 Master arrival 或 completion 不再随压力增长。
- `grpc_queue_ms`、`route_submit_ms` 或 `batch_wait_ms` 在拐点显著上升。
- JFR 显示 Master 热线程、锁竞争、分配或 GC 与对应阶段一致。

处理：在问题档位保留现场并 profile，先用阶段打点定位模块，再改代码或配置。不要只根据火焰图中 `NioEventLoop.select` 的占比下结论；select 可能只是线程等待，必须结合 CPU 利用率、runnable 线程和队列延迟判断。

### 11.4 engine 容量或状态同步不足

典型现象：

- `NO_AVAILABLE_WORKER`、available concurrency 为 0，或 endpoint 未全部 alive。
- mock `prefill_pending`/`decode_running` 随时间持续增长。
- 降低 `sleep_scale` 或增加 engine 后问题消失。

这类结果不能用于宣称 Master 到达上限，应先恢复 engine 余量。

## 12. 冷启动和预热

1250 个 endpoint 在启动阶段会集中建 channel、执行 callback 和同步状态。必须看到日志中的 endpoint ready，再等待至少 10 秒开始发压。

如果只在启动后第一档出现以下错误，而同配置预热后零错误，应标记为冷启动污染并重测，不纳入稳态数据：

- `RejectedExecutionException`，尤其是 `engine-grpc-client-executor`。
- 短暂的 `NO_AVAILABLE_WORKER`。
- endpoint discovered 数量不足 750/500。

如果预热后仍复现，就不能按冷启动忽略，必须保留日志和 JFR 分析。

## 13. Profile 方法

`run_online_eval.sh` 默认给 FlexLB 开启 JFR，并将结果写到 `flexlb_profile.jfr`。对容量拐点至少保留一份 JFR：

```bash
jfr summary run/<run_id>/flexlb_profile.jfr
jfr print \
  --events jdk.CPULoad,jdk.GarbageCollection,jdk.GCHeapSummary,jdk.JavaMonitorEnter,jdk.ThreadPark \
  run/<run_id>/flexlb_profile.jfr \
  > run/<run_id>/jfr_key_events.txt
```

用 JDK Mission Control 或已部署的 async-profiler 生成火焰图时，采样窗口必须覆盖稳态问题段。分析顺序：

1. 先看 `server_latency.json` 判断是哪个阶段上涨。
2. 再在 JFR/火焰图中找该阶段对应的 CPU、锁、分配或 executor 排队。
3. 同时检查 mock pending 和 load client 实际发送 QPS，排除上下游。
4. 修改后用同一档位和同一 worker 数做 A/B，并至少重复 3 次。

## 14. 结果记录模板

每次测试至少记录：

```text
date:
host/container:
git commit:
CPU/memory/container limits:
trace:
master config:
performance model:
n_prefill/n_decode:
schedule mode:
schedule_only:
batch algorithm:
fixed window wait ms:
schedule workers:
load client workers:
replay speed:
duration:
JVM Xms/Xmx:
actual send QPS:
Master arrival/completion QPS:
arrival/completion count:
errors:
Master avg/p50/p90/p95/p99/max:
grpc queue avg/p95/p99:
route submit avg/p95/p99:
batch wait avg/p95/p99:
dispatch ACK avg/p95/p99:
ACK response avg/p95/p99:
mock max prefill pending/decode running:
verdict:
artifact directory:
```

原始 run 目录不要只保留汇总表。至少保留 `summary.json`、`server_latency.json`、JFR、FlexLB 日志、mock 日志和启动环境，才能在结果异常时回到现场。

## 15. 常见错误

- 只看 client RTT：会把发压端排队和网络抖动算到 Master。正式值必须取服务端打点。
- 忘记 `SCHEDULE_ONLY=1`：会执行 FetchResponse，测试目标变成端到端链路。
- 使用 Python mock 冲高 QPS：mock 可能先成为瓶颈，无法证明 Master 容量。
- 使用固定 completion delay：丢失输入长度、cache 命中和 active batch 对执行时间的影响。
- 一次直接打 10K：无法定位拐点，还可能用冷启动失败污染结论。
- 只改 `REPLAY_SPEED` 不看实际 QPS：speed 是 trace 时间倍率，不是 QPS 本身。
- 同时改变 Master worker 和 load worker：无法判断收益来自服务端还是发压端。
- 不做预热：1250 个 endpoint 的建连和状态同步会污染第一档。
- 只看 `NioEventLoop.select` 占比：等待线程占比高不等于 event loop 是吞吐瓶颈。
- 忽略 mock pending：下游模拟容量不足时，Master 长尾结论无效。

## 16. 交接完成检查表

- [ ] Java 21 和两个 jar 校验通过。
- [ ] 使用 Java mock、750 prefill、500 decode 和 `fast_ab`。
- [ ] 使用 batch、`SCHEDULE_ONLY=1`、fixed-window 10 ms，没有 FetchResponse。
- [ ] endpoint 全部 ready 后预热至少 10 秒。
- [ ] 从约 100 QPS 开始，按阶梯逐级加压。
- [ ] QPS 和延迟均取 Master 服务端指标。
- [ ] 每档 error 为 0，arrival/completion count 相等。
- [ ] load client 和 mock engine 均有余量。
- [ ] 拐点档位保存 JFR、阶段延迟和完整日志。
- [ ] A/B 测试固定 trace、engine 数、load worker 和机器环境。
- [ ] 结果与原始产物路径写入 `rtp_llm/flexlb/docs/` 报告。
