# 如何添加新 case

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
