# 新增 case

## 只改变数据或阈值

在 `config/scenarios/` 下按 case 名新增或修改 schema v2 YAML，不创建分类子目录。YAML 保存拓扑、profile、请求数据、时间预算、阈值和参数约束，不写步骤或条件分支。

## 新增业务流程

在 `src/cases/programs/` 增加 Python program，并在显式目录中注册。Python 通过 `CaseBuilder` 读取 YAML 数据，声明步骤、输出引用和检查；公共动作不足时才在 `src/scenario/actions/` 增加有类型的 handler。

## Analysis 与 gate 报告

需要接受顶层 `analysis` 参数时，program 模块声明可调用的 `ANALYSIS_POLICY_VALIDATOR`。校验函数接收参数映射，返回校验后的映射，非法输入抛出 `ValueError`；未声明的 program 不接受 `analysis`。场景加载和报告策略加载共用此声明，YAML 不能声明或开启能力。

缓存 A/B 策略只接受 `alignment_event`（非空事件名或 null），不接受 `comparison` 字段。独立策略文件直接保存这些参数；从场景文件提取时，还会检查注册 program 的能力声明。

Gate 报告由 Python 生产者调用 `write_bundle(..., role="gate")` 发布。汇总器按 manifest 的角色发现并校验当前运行目录下的全部 gate bundle，不依赖 case 名或 bundle 目录名。没有角色声明的旧 bundle 仍可直接打开，但不会自动列入 gate 链接。

Runtime mode 的 `default_profile` 和 `default_master_mode` 必须成对声明。Profile 必须在 `flexlb_profile_data.REGISTERED_PROFILE_SPECS` 中注册，decision / dispatcher 轴必须与所选 master mode 一致。

## 分类

- 少量确定请求验证返回码、状态或边界：`functional`。
- 持续负载、故障、扩缩容、恢复或阶段曲线：`workload`。

在 YAML 顶层 `test` 中声明公共默认值：

```yaml
test:
  kind: functional
  description: 请求完成和状态恢复
  collection: diagnostic
```

`kind` 可选 functional / workload；`collection` 可选 aggregate / request / diagnostic。持续负载可以通过 `test.monitoring` 覆盖采样间隔等监控参数。每个实例必须获得完整的性质、说明和采集档位，缺失会报错。

只有需要列入 CI 必跑时，才在 `config/suites.yaml` 的 `ci_suites` 中增加 `case::variant`。文件位置、业务 category 和 kind 均不隐含 CI 必跑。

## 验证

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios --profile batch-window --suite functional --list-json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --instances '<exact-instance-id>' --parallel 1 --dry-run

cd tools/online_eval
PYTHONPATH="$PWD/src:$PWD" python3 -m unittest discover -s tests -p 'test_*.py'
```

新增检查必须保存实际值、期望值和原始证据。扩大拓扑或负载后需重新验证阈值，不能复制旧 band 后直接宣称场景有效。

## 配置层次与 profile 身份

本节是三层配置和变体身份的唯一规范。`render_env` 按 L1 → L2 → L3 单向渲染；新增字段先按下面的判据归类，再扩展已有 schema，不另建配置通道。

| 归属 | 判据 | 字段 |
| --- | --- | --- |
| L1：全局默认 | 单位和含义不随 decision / dispatcher 改变的通用测试基线 | `default_priority`、`preemption`、`prefill_expression`、`request_timeout_ms`、`decision_lifetime`、`status_rpc_ms`、`status_stale_after_ms`、`cleanup_interval_ms`、`decode_max_engine_requests`、`decode_max_kv_usage_percent` |
| L2：profile 模板 | 定义 profile 身份，或依赖调度模式解释、决定窗口与准入预算 | `decision`、`dispatcher`、`ordering`、`queue_timeout_ms`、`max_requests`、`max_collection_wait_ms`、`max_predicted_execution_ms`、`max_inflight_per_prefill_worker` |
| L3：case 数据 | 用例明确偏离测试基线的输入、阈值、公式或容量策略 | 所有允许覆盖的非身份字段，以及默认不启用的 `cache_affinity_max_extra_ttft_ms`、`cache_affinity_min_prefix_hit_percent` |

L1 是 `FUNCTIONAL_DEFAULTS`；L2 是 `FUNCTIONAL_PROFILE_KWARGS`。L2 的窗口字段只在 FIXED_WINDOW 渲染，SINGLE 不输出窗口字段。`GENERATOR_DEFAULTS` 仅用于低层 `build_flexlb_config` 的直接 schema 构造，不参与 profile 合并。低层构造器和 render-only stress 配置可用于 schema 测试；场景加载器和 functional 渲染入口均执行闭集 profile 身份保护。

Profile 身份只由 decision × dispatcher 决定，四个组合以 `PROFILE_SPECS` 为准。L3 允许重述相同身份值，禁止改变任一轴；`ordering`、优先级、抢占策略、队列期限是可覆盖策略，不定义新的 profile。有效能力从最终配置计算，不能从 profile 名推断 priority 或 preemption。`queue_timeout_ms: {omit: true}` 表示使用 Java 自身的队列期限，不表示无限等待；故障用例需要在配置旁说明省略的测试意图。

配置编译器使用一份环境字段目录：简单布尔/整数开关在 `OPTIONAL_SCALARS` 声明类型、缺省落值和数值下界；其余字段由专门校验分支处理。变体补丁从同一目录派生可用字段，但 `backend` 只允许在 case 顶层声明。功能 profile 的身份冲突由 `validate_profile_identity` 统一校验。调度配置进入 JSON 前由 `FifoOrdering` / `PriorityOrdering`、`SingleDecision` / `FixedWindowDecision`、`DispatcherPolicy` 和 `PreemptionPolicy` 表达；对象负责各自的字段和校验，`to_json()` 是序列化边界。新增 decision 或 dispatcher 类型时，需扩展对应值对象的选择逻辑、profile 注册、Java 严格 schema 及契约测试。YAML、冻结报告和外部 FLEXLB_CONFIG 仍使用既有数据格式。

准入 cap 按 dispatcher 计数：BATCH 的单位是 batch，NON_BATCH 的单位是请求；SINGLE 决策不把 BATCH 的 cap 改成请求计数。L2 明示 BATCH 为 2、NON_BATCH 为 64。FIXED_WINDOW 的默认 `max_requests=32` 表示每批最多 32 个请求，2 个满批的请求数上界是 64；这不是吞吐等价保证，SINGLE+BATCH 也不保证形成满批。修改批大小不会隐式换算 cap，需要同时调整时在 YAML 明示两者。

Case 的 `environment.config_overrides` 给出公共 L3 值，`environment.profile_overrides` 以完整 profile 名给出同层更具体的值：

```yaml
environment:
  config_overrides:
    request_timeout_ms: 120000
  profile_overrides:
    single-nonbatch:
      max_inflight_per_prefill_worker: 64
    batch-window:
      max_inflight_per_prefill_worker: 2
```

合并次序是模型 preset 的配对覆盖 → 公共覆盖 → 所选 profile 覆盖，再作为 L3 传给渲染器。null 不覆盖已有值；`{omit: true}` 只适用于可省略字段。不同 profile 的映射互不继承。所有键控条目在加载时校验，即使 CLI 未选择该 profile；未知 profile、未知字段、非法值、身份轴修改均失败。模型基线偏离继续使用 `model_override` 给出原因。最终 `resolved_config` 和原始配置都进入计划，可审计实际生效值。

## 变体维度、参数与命名

Case 与 variant id 均匹配 `[a-z][a-z0-9_]*`。Case 名描述测试目标；variant 名描述该 case 内可枚举的完整测试点，用描述性名词，避免裸机制缩写。实例身份固定为 `case::variant::profile`，CI 使用前两段；报告与冻结 bundle 使用完整三元组。同一文档禁止重复 variant id，同一配置集合禁止重复 case id。

只有一个测试点时，在顶层声明 `program`，省略 `variants` 和 `variant_axis`；加载器生成字面量 `default`，CI 与报告也使用 `default`。显式空列表与 null 都是配置错误。兜底不增加参数、环境或 profile 继承层。数据集身份记录在参数及冻结来源中，不让唯一的数据集名字充当样板变体。

有差异测试点时，声明一个 `variant_axis`，用 `kind` 标识维度，用 `fields` 明列允许变化的路径：

| kind | 允许字段 | program 选择 |
| --- | --- | --- |
| `data` | `parameters.<路径>`，如请求数量、输入数据集、阈值 | 只在 case 顶层声明 |
| `scale` | `environment.n_prefill`、`environment.n_decode`、`environment.prefill_cache_blocks`、`environment.decode_cache_blocks` | 只在 case 顶层声明 |
| `flow` | `program` | 每行 id 必须等于 program，且属于模块的 `FLOW_PROGRAMS` |

例如为同一流程增加第二个请求数据点：

```yaml
program: immediate
variant_axis:
  kind: data
  fields: [parameters.count]
variants:
- id: one_request
  parameters: {count: 1}
- id: three_requests
  parameters: {count: 3}
```

每个变体独立深拷贝 case 参数后合并自己的补丁，不与兄弟变体合并。相同参数路径在不同变体中合法且隔离；同一映射重复键、重复身份、未声明维度的字段、程序未读取的变体参数均失败。声明路径不得重叠。共享的 `profiles`、`execution`、`test`、`reports`、`metadata`、`parameter_schema` 和调度配置只放顶层；变体不能用 profile 筛选或配置覆盖冒充一个数据点。需要同时改变不属于同一维度的条件时，拆为不同 case。

`program` 是注册模块内的 Python 函数名；参数路径由程序的 `case.value(...)` 定义，与 variant id 无关。参数根可以表达业务结构或流程结构，但不能由 variant id 动态拼出；新增变体不复制 id 作为参数前缀。修改 variant id 需要同步 CI 选例、实例清单测试和外部选择命令，不修改程序参数路径。冻结 bundle 按其内置身份加载展示，不重解释成新配置，也不要求旧身份可重算。

流程分叉按影响量级归位：阈值、计数、超时只改变同一验证契约时留在 parameters；单一枚举完整决定流程和检查集合、共用编排骨架时使用 flow 变体；多开关组合或根本不同的步骤/验证结构使用不同 case。Flow 入口固定分支值，被吸收的旧开关必须在加载期拒绝，不能独立于身份修改。

公共程序的分支审查以运行结构为准：输入格式、QPS 一致性和预算充足性检查属于参数校验，不创建流程身份；依据可选字段插入中间阶段或切换故障注入方式属于流程差异，不得作为同一测试点的隐藏开关。单步缩容入口只接受一次 graceful 缩容；中间缩容或不同故障序列需要独立 case。可复用 action 支持更多路径不代表所有 case 都允许选择这些路径。
