# 如何添加新 case

新增 case 先选择 functional 或 workload，并在 `suites.yaml` 登记分类；复杂场景的独立检查使用 Python `case.observe`。具体执行策略和产物见 [两类测试](test-suites.md)。

当前入口是 **YAML 提供数据，Python 定义测试行为**。
术语、profile、资源所有权与结果状态见 [框架设计](framework-design.md)。
以下命令从仓库根目录运行；真实 Java 测试使用已分配的远端端口与运行目录。

## 1. 只改变配置：修改 YAML

复制最接近的 `scenarios/` 配置文件，修改顶层 `id`，保留程序所需的参数节，
再按目标测试修改环境、输入、时间预算和断言数据。Python 不提供缺失参数的默认值。

每行 `variants` 通过 `program` 选择 Python 函数；同一函数可以配出多种规模。
例如在现有 `request_completion.yaml` 中增加：

```yaml
- id: completion_4p8d
  program: immediate
  profiles: [batch-window, single-batch]
  environment:
    n_prefill: 4
    n_decode: 8
  parameters:
    count: 4
    input_len: 4096
    output_len: 8
    completion:
      setup_timeout_s: 240
```

该行生成两个实例。根 `parameters` 与本行参数递归合并，列表和标量整体替换；
本行只覆盖 `completion.setup_timeout_s`，其余 completion 配置仍继承根配置。
每个实例取得独立副本，不会污染其他变体。

所有用例配置写在 YAML：`environment`、`profiles`、`execution`、`metadata`、
`parameters` 和数值约束 `parameter_schema`。Python 不再维护 `VARIANTS`、
`PROFILES`、`METADATA`，也不通过 `case.number()` 提供默认值或上下界。
改变阈值会改变测试含义，应说明依据；不能为了得到 PASS 放宽期望。

## 2. 需要新的业务流程：添加 Python 程序

在 `flexlb_test_framework/case_programs/` 创建 `my_completion.py`：

```python
from ..case_config import output


def normal(case):
    case.step("setup", "setup", timeout_s=case.value("setup_timeout_s"))
    case.step("submit", "request", params=case.value("request"))
    case.step("terminal", "wait", params={
        "requests": output("submit", "requests"),
    })
    case.step("completed", "check", params=case.params("completed", {
        "actual": output("terminal", "completed"),
    }))
    case.step("no_errors", "check", params=case.params("no_errors", {
        "actual": output("terminal", "error_count"),
    }))
    case.step("cleanup", "teardown")
```

在 `case_programs/__init__.py` 的 `PROGRAMS` 中注册模块入口：

```python
"my_completion": "flexlb_test_framework.case_programs.my_completion",
```

配套 YAML 定义全部用例数据：

```yaml
schema_version: 2
case: my_completion
metadata:
  description: All submitted requests complete without stream errors.
  category: status
  tags: [smoke]
profiles: [batch-window, single-batch]
environment: {backend: java_mock, n_prefill: 2, n_decode: 2}
execution: {timeout_s: 300, stage_timeout_s: 60, cleanup_timeout_s: 120}
parameters:
  setup_timeout_s: 180
  request: {input_len: 2048, output_len: 2, count: 2}
  completed: {op: eq, expected: true}
  no_errors: {op: eq, expected: 0}
variants:
- id: normal
  program: normal
```

`case.value("path.to.value")` 读取必填参数；`case.number("count")` 额外按 YAML
`parameter_schema.count` 中的 `minimum/maximum/integer` 校验数值。
`case.params("path", dynamic)` 把 YAML 数据与 Python 运行结果引用递归合并；
动态引用由 Python 绑定，不能在 YAML 中填写 `$ref`。

`profiles` 必须在根或变体行给出。根列表约束整份配置的运行范围；变体行可从中选择，省略时继承根列表。没有根列表时，各变体独立声明；没有 Python 隐藏的用例 profile 列表。
元数据、能力要求和 finding 标记也在 YAML 的 `metadata` 中配置，可按变体覆盖。
实际能力仍由编译器验证，例如需要 batch 接口的流程不能在 non_batch 环境运行。

Python 构造器只决定操作顺序、分支、循环、结果绑定与计算，此时不发请求或启动服务。
运行中的动态观测放在 action 中处理；不要把 `steps`、`action`、`$ref` 等编排字段放进 YAML。

## 3. 检查配置并执行

先列出实例，核对程序哈希、配置哈希和资源预算：

```bash
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py   --source rtp_llm/flexlb/tools/online_eval/scenarios/core/my_completion.yaml   --list-json
```

再预览父进程分配：

```bash
python3 rtp_llm/flexlb/tools/online_eval/parallel_runner.py   --source yaml   --case-dir rtp_llm/flexlb/tools/online_eval/scenarios/core/my_completion.yaml   --instances 'my_completion::normal::batch-window'   --profile batch-window --grade normal --parallel 1 --shard case   --out-dir /tmp/completion-large-review --dry-run
```

在已分配的远端开发容器中先构建同一源码版本的 Java API 与 Mock JAR，
设置租约范围内的 `FLEXLB_FT_PARALLEL_MASTER_BASE` 和 `FLEXLB_FT_PARALLEL_MOCK_BASE`，
换一个新的输出目录并去掉 `--dry-run` 执行。通过父入口运行，它负责端口锁、资源分配和汇总。
`--parallel` 是同时运行的独立实例上限；实际通道数还受实例数、端口窗口与资源预算限制。

`--source yaml` 选择“YAML 配置驱动的 Python case”。
父入口默认选择新版和内置 `scenarios/`，日常命令可以省略 `--source yaml --case-dir scenarios`。
性能压测另见 `stress/`。
列举、编译、dry-run 都不能当成实际 Java 通过。

## 4. 已有 action 不够时

先查 `flexlb_test_framework/scenario/catalog.py` 和 `scenario/actions/`。
若已有动作能表达请求、故障和观测，直接由 Python case 组合；只有新增底层能力才添加 handler。

扩展 action 包含四部分：参数校验、运行函数、公开输出类型、检查 ID 集合，
以 `StageHandler` 注册到模块 `HANDLERS`，再由 catalog 显式导入。
可参考 `scenario/actions/environment.py` 和 `status_protocol.py`。

- 编译校验禁止 I/O；运行阶段通过 `ctx` 使用环境、资源和操作接口。
- 先保存原始证据，再判断实际值；缺失或无法解析不能填零当成功。
- 断言不满足返回 FAIL；协议/执行异常保留 ERROR，真实超时保留 TIMEOUT。
- RPC、HTTP、轮询及线程 join 必须遵守剩余 deadline。
- 活动资源取得后立即登记清理回调；动态 worker 走 `ctx.ops.add_engine` 并声明新增上限。
- 返回的检查 ID 必须与声明匹配；cleanup 失败不能得到 PASS。

## 5. Finding 和验证

已确认的 finding 由 YAML 的 `metadata` 声明，例如 `findings: [completed.comparison]`。
它仅承接该检查的普通 FAIL；新失败不能自动标为 finding，ERROR/TIMEOUT/清理错误仍阻断交付。

验证应覆盖新增行为或错误边界，并运行选中的真实 Java 实例。
检查父 `aggregate.json`、child `scenarios.json`、实例 `result.json`、原始证据及 cleanup。
报告同时写清源码/JAR、配置、grade、实例 ID 和结果路径；不能仅凭测试数量判断业务通过。


### Master Schema 3 配置

在 YAML 的 `environment.config_overrides` 中配置 `max_inflight_per_prefill_worker`、
`request_timeout_ms`、`decision_lifetime` 等字段。inflight 上限在 BATCH 下计批次，
在 NON_BATCH 下计请求；构造三个并存请求的容量用例应显式给够上限，不能依赖默认值。
`request_timeout_ms` 是 Master 的请求非活动预算：引擎侧请求结束不代表 Master 的记账立即消失。
恢复阶段先验证两侧清场；等待预算也放在 YAML 中。

全局 outstanding/等待位上限、旧 delivered-not-accepted 配置和独立 ack timeout 已删除。
PRIORITY 省略 preemption 会启用默认抢占，不能用省略字段构造“关闭抢占”用例。
完整迁移与退役范围见 [Schema 3 对齐记录](validation/config-alignment-20260909.md)。

## 持续负载用例的证据接入

配置中的请求数量、发射间隔、并发上限、变更次数和时间预算都在 YAML 声明。Python 负责有界发射和收尾；不能通过等待每个请求完成再发下一笔来模拟并发负载。慢请求达到并发上限时，记录发射延迟，避免无界排队。

请求优先使用公共 `RecordedRequests`，并把持有它的资源注册到运行上下文。自定义生产器公开 `snapshot_records()`；需要同时提供自定义完整性信息时公开 `evidence_snapshot()`，返回 `records`、`complete`、`errors`。Python 生产器注明 `producer_kind: python`。不能只在最终汇总中写一个成功数；每笔请求须保留 ID、发出时间、终态、协议结果和错误，异常路径也如此。直接构造前置状态的请求标记 `purpose: preconditioning`，仍进入统一关联表，但单独写入聚合目录的 `preconditioning-requests.json`。

读取 Mock 指标时使用公共 `online_eval.telemetry.http_text` 或共享序列 API。不要直接请求会清空计数的指标接口。内存保留条数由 `suites.yaml` 的 `sample_history_limit` 决定；较早样本从原始日志流式回放。每个环境、每个 master 都有独立采集日志和采集生命周期；首尾缺采、断采、缺失数据源或失败收尾都不能当作有效运行。

一个功能合同可以只发送几笔请求验证边界，不需要为了生成压测图而延长等待。一个复杂场景则必须提供足够的持续负载与观测窗口。参考 `balance_distribution` 的 `sustained_mix` 和 `elastic_concurrent_mutation`，同时检查请求完成、资源清理、时间序列与恢复质量。

## 添加由 Java 发流量的场景

参考 `scenarios/workload/trace_scale_out.yaml` 与 `case_programs/trace_scale_out.py`。YAML 中 `background`、`formal` 是配置块，调用顺序完全由 Python 决定。

1. 在 YAML 为每组声明 `group_id`、`phase_id`、`poll_s`、JVM 内存、`trace` 和 `client`。`trace` 要提供种子、数量、毫秒间隔、block size，以及各 family 的精确 prefix token、随机 suffix 长度与 token 上界、输出长度和优先级。
2. Python 调用 `java_flow_start`，取得 `java_flow` 句柄。Java 使用现有 ClientOps 启动，环境地址与运行目录由框架绑定，不能在 YAML 写死远端端口。
3. 使用 `java_flow_checkpoint` 确认真实发送、完成和 Decode 在途数量。扩容后可绑定 `elastic_add` 返回的 engine，确认它已有接收记录。其他前提仍由相应 Python action 按真实证据判断。
4. 构造忙碌状态的背景组跨阶段继续运行；不要在每个 checkpoint 排空。有限准备组是否排空由 Python 用例明确安排。
5. `java_flow_stop` 停止接纳新请求，`java_flow_drain` 等待终态和进程退出，`java_flow_check` 分别断言完整性与 YAML 指定的成功率。
6. 在 `suites.yaml` 登记为 workload；独立结果检查使用 `case.observe`，前提检查使用 `case.step`。缺前提时应中止依赖步骤并保留部分证据。

场景客户端必须显式配置 `REPLAY_UNIQUE_PREFIX: 'false'`、`FETCH_OUTPUT_STREAM: 'true'`、正的 `DURATION_S` 和 `MAX_CONCURRENCY`。未知环境变量名会被拒绝，例如响应收尾预算的正式名称是 `RESPONSE_TIMEOUT`。正常停发不使用 `ClientOps.stop_async` 的终止进程路径。

`flow-input.json` 与 trace manifest 保留配置和摘要；`client_lifecycle.jsonl` 支持运行中定位，`client_events.jsonl` 是自然退出的完整请求产物。根据 run/group/rid 和实际时间选择请求集合，保留失败原因与未完成请求；同一请求跨阶段存在时不要重复算入全场分母。
