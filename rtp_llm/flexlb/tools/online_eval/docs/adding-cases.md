# 如何添加新 case

当前入口是 **YAML 提供数据，Python 定义测试行为**。
术语、profile、资源所有权与结果状态见 [框架设计](framework-design.md)。
以下命令从仓库根目录运行；真实 Java 测试使用已分配的远端端口与运行目录。

## 1. 只改变 P/D 规模或输入：添加 YAML 配置

复用现有 `request_completion` 程序，无须新增 Python 文件。例如保存为
`rtp_llm/flexlb/tools/online_eval/scenarios/core/completion_large.yaml`：

```yaml
schema_version: 2
case: request_completion
id: completion_large
profiles: [batch-window, single-batch]
environment:
  backend: java_mock
  n_prefill: 4
  n_decode: 8
parameters:
  input_len: 4096
  output_len: 8
  count: 4
execution:
  timeout_s: 300
  stage_timeout_s: 60
  cleanup_timeout_s: 120
variants:
  - id: immediate
  - id: delayed
    use: deferred_fetch
    environment:
      n_prefill: 2
      n_decode: 4
    parameters:
      count: 2
```

这里生成 4 个实例。`immediate` 调用已有 Python 函数，`delayed` 调用 `deferred_fetch`；
后者在 wait 时才开始 Fetch。各行的环境和参数互相独立。
配置中没有请求步骤、wait、分支或断言；完成且零错误的契约仍在 Python 中。

改变输入前先查看目标 Python 程序支持哪些 `case.number(...)` 参数。
目前 completion 支持上述三个参数，其他程序按各自声明接受参数。
`environment.n_prefill/n_decode` 属于公共环境配置；并非每条业务逻辑都适合任意规模，
例如“恰好两个 worker 的 90:10 分布”还依赖 Python 中的固定构造，需要一起审查。
拼错、未使用或越界参数会在编译时直接失败。

## 2. 需要新的业务流程：添加 Python 程序

在 `flexlb_test_framework/case_programs/` 创建 `my_completion.py`：

```python
from ..case_config import output

METADATA = {
    "description": "All submitted requests reach terminal without errors.",
    "category": "status",
    "tags": ["smoke"],
}
PROFILES = ["batch-window", "single-batch"]


def normal(case):
    count = case.number("count", 2, maximum=1000)
    case.step("setup", "setup", timeout_s=180)
    case.step("submit", "request", params={
        "input_len": 2048, "output_len": 2, "count": count,
    })
    case.step("terminal", "wait", params={
        "requests": output("submit", "requests"),
    })
    for name, field, expected in (
        ("completed", "completed", True),
        ("no_errors", "error_count", 0),
    ):
        case.step(name, "check", params={
            "actual": output("terminal", field), "op": "eq", "expected": expected,
        })
    case.step("cleanup", "teardown")


VARIANTS = {
    "normal": {"build": normal, "profiles": PROFILES, "metadata": {}},
}
```

然后在 `case_programs/__init__.py` 的 `PROGRAMS` 中显式注册：

```python
"my_completion": "flexlb_test_framework.case_programs.my_completion",
```

配套 YAML 只需给出程序名、环境与参数：

```yaml
schema_version: 2
case: my_completion
environment: {backend: java_mock, n_prefill: 2, n_decode: 2}
parameters: {count: 2}
variants: [{id: normal}]
```

没有根 profiles 时，默认采用该 Python 变体声明的 profiles。
Python 构造器负责产生计划，此时不要发请求或启动服务。根据配置数据选择 Python 分支，
或用 Python 循环构造重复步骤；运行中的动态观测放在 action 中处理。
不要把 `steps`、`action`、`$ref` 等编排字段重新塞进 YAML。

## 3. 检查配置并执行

先列出实例，核对程序哈希、配置哈希和资源预算：

```bash
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py   --source rtp_llm/flexlb/tools/online_eval/scenarios/core/completion_large.yaml   --list-json
```

再预览父进程分配：

```bash
python3 rtp_llm/flexlb/tools/online_eval/parallel_runner.py   --source yaml   --case-dir rtp_llm/flexlb/tools/online_eval/scenarios/core/completion_large.yaml   --instances 'completion_large::immediate::batch-window'   --profile batch-window --grade normal --parallel 1 --shard case   --out-dir /tmp/completion-large-review --dry-run
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

已确认的 finding 也只能由 Python 元数据声明，例如 `findings: [completed.comparison]`。
它仅承接该检查的普通 FAIL；新失败不能自动标为 finding，ERROR/TIMEOUT/清理错误仍阻断交付。

验证应覆盖新增行为或错误边界，并运行选中的真实 Java 实例。
检查父 `aggregate.json`、child `scenarios.json`、实例 `result.json`、原始证据及 cleanup。
报告同时写清源码/JAR、配置、grade、实例 ID 和结果路径；不能仅凭测试数量判断业务通过。
