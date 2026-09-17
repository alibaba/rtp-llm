# Batch 与请求流量录制

当前范围（2026-09-16）：只保留流量录制，不新增 profiler 打点，不自动开启 profiler，不输出 captures.jsonl 或采集 kernel trace。原仓库已有 profiler 功能不受影响。离线工具保留对历史 trace 的兼容，不代表当前录制仍支持 batch→kernel 自动关联。

本功能默认关闭。当前代码接入普通 `NormalExecutor`，请求时间嵌入输入 owner 输出的 batch 日志，batch 长度来自实际模型输入，不再记录独立生命周期日志。完整设计见 [设计文档](../batch_execution_record_replay_design.md)；按调用路径阅读实现见 [新增代码阅读指南](../batch_recording_code_walkthrough.md)。

## 开启录制

在启动各个引擎进程前设置相同的会话 ID。请求事件/manifest 等使用独立 owner 子目录，batch 文件由 alog.conf 决定：

```bash
export RTP_LLM_RECORD_DIR=/tmp/rtp-recordings
export RTP_LLM_RECORD_SESSION=experiment-001
export RTP_LLM_RECORD_REPLICA=replica-001
export RTP_LLM_RECORD_DECODE=0
export RTP_LLM_RECORD_QUEUE_SIZE=4096
export RTP_LLM_RECORD_FILE_BYTES=0
export RTP_LLM_RECORD_TOTAL_BYTES=1073741824
```

`RTP_LLM_RECORD_DIR` 和 `RTP_LLM_RECORD_SESSION` 必须同时设置；启动后不支持动态修改。会话 ID 应在每次新实验时更换。`RTP_LLM_RECORD_FILE_BYTES` 默认 0，是旧 owner JSONL 的兼容配置，当前仅输出 batch 时不生效；batch 的轮转由 alog.conf 的 max_file_size 管理。其他数值参数要求为正整数，无效值回退默认值。队列单位为记录条数，总字节预算按 owner 独立计算，包含提交给 batch logger 的数据。

所有 batch 逐行追加到 alog 配置的固定文件 `logs/batch_schedule.log`，路径相对于服务工作目录；不添加 PID 后缀，不按 batch 建文件。TP 输入 owner 才提交 batch 快照，默认不轮转，大小和轮转由 alog.conf 管理。多个实例共享该路径时采用 best-effort 语义，不额外实现跨进程写入/轮转协调；损坏或丢失记录按缺失数据处理。请求组成、长度和已知 TTFT 嵌入 batch 的 requests 数组；不再写 request_events.jsonl、executions.jsonl 或 requests.jsonl。owner 目录只保留 manifest.json 元数据。达到总提交预算（默认 1 GiB）或 snapshot worker 出错时停止接收新记录。

batch 通过 `RTP_LLM_BATCH_SCHEDULE_LOG_INFO(json)` 写入配置名为 batch_schedule 的 logger，使用不截断的原始消息接口，不加源码/时间前缀，不受 USE_CONSOLE_APPENDER/LOG_LEVEL 包装层重定向。路径、级别、格式、异步 flush 和轮转完全由 [alog.conf](../../rtp_llm/config/alog.conf) 的 batchScheduleAppender 控制，录制器不覆盖这些配置。初始化 alog 时可能创建空 batch 文件，即使录制尚未启用；没有 batch 配置则该专用 logger 默认不向 root 输出。其他 JSONL 暂保留动态 appender；manifest 仍用临时文件 + rename 更新。

```properties
alog.logger.batch_schedule=INFO, batchScheduleAppender
inherit.batch_schedule=false
alog.appender.batchScheduleAppender=FileAppender
alog.appender.batchScheduleAppender.fileName=logs/batch_schedule.log
alog.appender.batchScheduleAppender.layout=PatternLayout
alog.appender.batchScheduleAppender.layout.LogPattern=%%m
alog.appender.batchScheduleAppender.async_flush=true
alog.appender.batchScheduleAppender.flush=false
alog.appender.batchScheduleAppender.flush_threshold=64
alog.appender.batchScheduleAppender.flush_interval=100
alog.appender.batchScheduleAppender.max_file_size=0
alog.appender.batchScheduleAppender.compress=false
```

flush_threshold 单位 KiB，flush_interval 单位毫秒，max_file_size 单位 MiB（0 禁止轮转）。请保留纯消息布局和关闭继承，否则输出不再是独立 JSONL。修改配置后重启进程使其生效。宏统一由 Logger.h 引入：

```cpp
RTP_LLM_BATCH_SCHEDULE_LOG_INFO(json_line);
RTP_LLM_BATCH_SCHEDULE_LOG(alog::LOG_LEVEL_INFO, json_line);
```

参数是已序列化的完整 JSON 字符串，不是 printf 格式串；宏自身不负责录制开关/请求身份/队列，正常录制仍先进入 ExecutionRecorder，待快照准备好后再调用宏。

交付语义为 **best effort**：缺日志就是缺失数据，不承诺无损。manifest 的 `submitted_to_alog` / `bytes_submitted` 表示提交量，不是实际落盘量；`written`、`bytes_written`、`sink_dropped` 为 null，`dropped` / `errors` 仅覆盖录制器能观察的丢弃/错误。`storage_backend=alog`、`delivery_policy=best_effort`，`complete` 始终 false（不能证明全量无损）。正常关闭 drain 上层队列并尝试 flush 本录制器的 alog 文件，`closed=true` 也不是无损保证。

分析器对 best-effort 录制应只使用实际观察到的数据。精简格式从 batch 聚合 TTFT，缺失 TTFT 保持未知，不推断请求是否结束。损坏 JSONL 行警告后跳过；完全未被记录的请求不可统计。旧版事件格式保留原完整性检查。

输出示例：

```text
/tmp/rtp-recordings/owner-<pid>-<timestamp>/
  manifest.json
<服务工作目录>/logs/batch_schedule.log
```

每个被采集 batch 一行，包含完整 requests 数组，`schema_version` 保持为 `1`。精简后的字段约定：

- batch 层仅输出 `schema_version`、`owner_id`、`exec_id` 和 `requests`。
- 每个请求显式输出 `phase`、`input_len`、`reuse_len`、`q_len`、`kv_len`，长度为 0 时也保留。
- `request_id` 仅在已知时输出；`ttft_us` 仅在已知时输出，单位微秒，未知不能当成 0。后续 batch 继续携带已知 TTFT，统计时按 owner + request 去重，排除 fake 请求。
- `sequence_id` 缺省为 0，仅非零时输出；`is_fake` 缺省为 false，仅为 true 时输出。数组下标表示 batch slot。
- 不再输出 `scheduler_step_id`、入队/首次调度时间、`first_token_produced`。请求顺序不表示调度时间，不据此推断请求终态。
- session、replica、DP/world rank 从匹配 `owner_id` 的 manifest 查询；固定的 `model_role=target` 和 `length_source=cpu_input_snapshot` 也保存在 manifest 中。迁移日志时应一起保存 manifest。

`input_len`（原 `prompt_tokens`）是请求原始输入 token 数，不包含本次生成的输出；`reuse_len`（原 `prefix_cache_hit_tokens`）是初始命中的前缀 token 数，不随 decode 增长，也不从 input_len 中扣除。

普通单 token decode 时，`q_len=1`，`kv_len=seqLength()-1`（异步 device-state 路径使用等价的 CPU 快照 `next_real_seq_len-1`）。若输入长度为 P，本轮开始前已生成 G 个 token，则已有 KV 长度为 P+G-1：最后一个已生成 token 是本轮输入，其 KV 将在本轮计算。例如 P=100，prefill 产出首 token 后第一次 decode 的 input_len=100、q_len=1、kv_len=100；下一轮分别为 100、1、101。这一关系针对普通逐 token decode，不用于推断 MTP 或特殊执行路径。

`exec_id`、`owner_id`、`q_len`、`kv_len` 分别替代旧 key execution_id、owner_instance_id、q_tokens、kv_tokens_before；仅改日志名称，内部变量名不变。q_len 和 kv_len 的单位均为 token，kv_len 始终表示本轮执行前已有 KV 的长度，不包含本轮 q_len。读取历史日志时可回退读取旧 key。

例如一个占位 decode batch：

```json
{"schema_version":1,"owner_id":"1623917-1789647165853528931","exec_id":1789647165854016833,"requests":[{"is_fake":true,"phase":"decode","input_len":1,"reuse_len":0,"q_len":1,"kv_len":1}]}
```

`RTP_LLM_RECORD_DECODE` 默认关闭，设置为 `1` 才采集 decode。关闭时跳过所有包含 decode 的 batch（包括混合 batch），在 CPU 快照分配和 JSON 构造之前过滤。长度仍来自输入组装阶段的 CPU 快照，录制不进行长度 D2H 或 CUDA event 等待。manifest 的 engine.record_decode 和 decode_filter_policy 标明采集范围，各 TP rank 应使用相同配置。

TTFT 复用 aux_info 的请求起点至首 token 更新耗时，不依赖流式发布或 aux_info 开关。日志延后到输出更新后提交；未产生首 token 的 chunk 或更新前失败的请求省略 TTFT，不回写历史行。未进入被采集 batch 的请求不会留下记录。

## Profiler 已从本期录制中移除

同一服务副本的各 TP rank 使用相同 `RTP_LLM_RECORD_REPLICA`；不同副本使用不同值。未设置时默认按 DP rank 命名，仅适合单副本录制。

`RTP_LLM_RECORD_PROFILE_START/STEPS` 已不再接入；移除新增的 `recorder.snapshot`、`rtp.execution`、`rtp.graph_replay` scope，以及 captures.jsonl 写入。exec_id 仍标识每轮 batch，但不会由录制器写入 profiler trace。原有 timeline 控制接口仍可独立使用，不能依靠它自动关联本期 batch ID。

## 离线分析

以下工具说明为历史接口；当前工作区不包含 `batch_trace_analyze` / `batch_replay` 脚本。外部读取工具使用精简日志时，需按 owner 关联 manifest、支持缺省字段，不能继续依赖每行的 session/DP 等字段或计算已移除的排队时间。

```bash
python -m rtp_llm.test.perf_test.batch_trace_analyze \
  --record-dir /tmp/rtp-recordings/owner-<pid>-<timestamp> \
  --batch-file /path/to/logs/batch_schedule.log \
  --output /tmp/record-report
```

当前按上面的命令分析流量长度和请求延迟，不传 trace。新 manifest 标记 batch_log_location=alog_config，分析时必须用 --batch-file 指定配置文件实际输出位置（可重复指定轮转分片），或 --batch-dir 指定已导出的旧格式 batch 目录。分析器按 session_id/replica_id/dp_rank 过滤文件中其他录制数据。输出 report.json、report.html，以及空的 kernel_events.jsonl（保留工具格式兼容）。以下 trace 分析选项只适用于旧版本历史 trace。

分析 TP 非输入 owner 的 trace 时，增加 `--batch-dir /path/to/input-owner` 指向同会话、同副本、同 DP 的 batch 目录。报告中的 `missing_batch_execution_ids` 表示 trace 执行缺少对应 batch，不能将此类执行视为已完整关联。

逐请求输出总耗时、首次排队、首 token 延迟和 `post_first_schedule_latency_ns`（首次调度到终态）。最后一项包含后续等待和输出开销，不是单请求独占 GPU 时间；缺少事件时相应耗时为 null。

## 执行形状回放

native 回放当前要求 DP=1，可使用 TP 多卡。多 DP 各组不同流量的联合回放尚未接入，因此明确拒绝 DP>1，避免把一个 DP 的快照复制给全部 DP 后误报为真实分布回放。

回放必须使用独立引擎进程和相同模型/拓扑配置，不能复用在线服务。支持单个 FP16/BF16 full-KV group 的普通融合文本模型，包括普通 prefill、decode 和混合 batch；MTP、多模态、特殊/量化 KV、PD 分离明确拒绝。每条快照独立执行，KV 以零值初始化，token ID 确定性生成；不复现内容、专家路由和 cache 共享。

先生成计划，不启动模型：

```bash
python -m rtp_llm.test.perf_test.batch_replay \
  --batches /path/to/batch_schedule.log \
  --record-dir /tmp/rtp-recordings/owner-<pid>-<timestamp> \
  --execution-ids 18231 \
  --warmup 5 --repeat 20 --output /tmp/replay-plan
```

指定引擎命令可自动启动独立进程并等待每个 rank 完成：

```bash
python -m rtp_llm.test.perf_test.batch_replay \
  --batches /path/to/batch_schedule.log \
  --record-dir /tmp/rtp-recordings/owner-<pid>-<timestamp> \
  --output /tmp/replay-run --world-size 1 \
  -- python -m rtp_llm.start_server <模型及引擎参数>
```

manifest 查找顺序为：显式 `--record-dir`、batch 文件同目录的 `manifest.json`、环境变量 `RTP_LLM_RECORD_DIR`。`--record-dir` 和环境变量既支持具体 owner 目录，也支持包含 `owner-*` 的录制根目录；指定根目录时根据 batch 中的 session、replica、DP/world rank 和 owner_id 匹配唯一 manifest，可用 `--execution-ids` 缩小范围。若日志包含多个 owner，需指定具体 owner 目录或执行 ID，脚本不会任意选取。选定 manifest 后只回放该 owner 的 batch，按其 best-effort 策略跳过损坏行，并将元数据复制为输出目录中的 `source_manifest.json`。迁移日志到其他机器时需一并复制对应 manifest 并指定目录；没有指定录制目录且找不到相邻 manifest 时，仍兼容仅含 batch 的输入，但会严格检查 JSONL。

输出目录必须是新目录，防止读取历史完成标记。launcher 设置 `RTP_LLM_REPLAY_PLAN`，引擎直接运行计划并拒绝在线请求；完成后 launcher 终止自己创建的进程组。多机启动需由现有分布式启动器分发相同计划并收集结果，当前 launcher 只管理本地子进程。

形状回放代码保留供后续拓展，当前不属于本期录制验收范围。运行时只输出同步执行耗时（包含输入准备），已移除自动采集 forward trace；不能将该耗时直接等同于纯 kernel 耗时。分析器的历史 trace 比较选项仅用于已有历史数据。

## 当前验证边界与待完善项

- 已通过 CPU recorder 测试、7 项离线工具单测和 mock 引擎录制/回放测试；mock 测试覆盖同步与 `RTP_LLM_STREAM_ASYNC=1`、`RTP_LLM_DROP_BROAD_SYNC=1` 模式。测试 JSONL 已用于分析和计划生成的命令行检查。
- 小矩阵验证已确认本机 PyTorch 2.11/CUDA 13 的 eager 和 graph replay 可关联；这不替代实际模型和多卡覆盖验收。
- 当前 manifest 提供模型/拓扑/cache 配置，报告比较配置差异；完整权重/软件指纹尚待补齐，不能把 config_match 解释为相同权重。普通引擎的 scheduler_step_id 随输入传播；离线回放没有在线调度 step。
- native 回放恢复 q/KV 配对和原始 prompt 长度元数据，真实内容和 MoE 路由不受控。graph bucket 由同配置的实际执行路径选择，报告标记 bucket 差异，尚未强制复现原 bucket。
- 尚未完成线上吞吐/TTFT/TPOT 开销门槛、多卡完整性、动态启停、有时限 shutdown drain、按事件类型 metrics 与特殊 KV 回放验收。当前实现不可宣称已达到设计文档的全部生产验收项。

测试命令：

```bash
python -m unittest rtp_llm.test.perf_test.batch_recording_test
bazelisk test //rtp_llm/cpp/observability:execution_recorder_test --config=cuda13
bazelisk test //rtp_llm/cpp/normal_engine/test:recording_engine_test --config=cuda13
CUDA_VISIBLE_DEVICES=<空闲GPU> python -m rtp_llm.test.perf_test.batch_trace_gpu_test --output /tmp/new-gpu-check
```
