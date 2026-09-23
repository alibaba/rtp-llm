# Python case 配置目录

每份 YAML 声明环境、profile、Python 变体选择、参数和测试元数据，格式版本为 `schema_version: 2`。
编排与断言位于 [`case_programs/`](../../src/cases/programs)；
公共 action 位于 `src/scenario/actions/`。

先读 [框架结构](../../docs/architecture/framework.md) 了解执行和资源模型；
新增用例按 [如何添加新 case](../../docs/development/adding-cases.md) 操作。

从仓库根目录列出当前 CI 必跑实例：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scripts/commands/list_cases.py \
  --source rtp_llm/flexlb/tools/online_eval/config/scenarios \
  --profile batch-window --suite core --list-json
```

`run_cases.py` 默认读取 `config/suites.yaml` 的 `default_suite`；当前 `core` 清单包含 5 个实例。
`--suite functional` / `workload` 按实例自己的 `test.kind` 筛选，`--suite all` 选择全部实例。
文件顶层 `test` 提供公共默认值；`variants[].test` 可以覆盖 kind、description、collection 和 monitoring。

所有 YAML 直接放在本目录，文件名采用稳定的 case 名；目录不参与分类或 CI 选例。`perf_preset` 选择模型采集档案。与采集值不同的时延、内存树容量、KV 池容量、PD 规模或公式必须作为测试派生覆盖，使用 `environment.model_override: {baseline: <perf_preset>, reason: <原因>}` 标明基线与原因。合成基线上的参数属于测试输入，不得当成真实部署观测值。块数口径见 [数据规则](../../data/README.md)。

更多配置及资源语义见 [执行器说明](../../src/scenario/README.md)。
