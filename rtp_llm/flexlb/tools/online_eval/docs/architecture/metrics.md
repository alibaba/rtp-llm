# 指标身份与采集契约

case 和报告共用 `metrics.json` 中的稳定 `metric_id`。exporter 的物理指标名、查询 ID、曲线 ID 和展示名承担不同职责；只有指标定义决定取数口径，不能通过展示名反推查询。

## 名称与身份

| 名称 | 所属位置 | 用途 |
|---|---|---|
| 物理指标名 | Java exporter 与 PromQL | 描述服务实际上报的 counter、gauge 或 histogram；保留生产对齐的名称与单位 |
| `metric_id` | 指标集合、Python program、`metrics.json` | 门禁和报告共用的逻辑身份，格式为 `<namespace>/<name>` |
| 标签、来源实例、环境代次 | 指标定义和序列 | 区分角色、引擎、Master、采样窗口等具体序列 |
| 本地 `curve_id` | view YAML 的 `curves` 键 | 选择一个指标的标签投影和展示方式，由面板的 `curve_ids` 引用 |
| `name` | view YAML | 面向读者的图例，可使用中文或变量模板，不参与取数和门禁 |

Prometheus 查询的完整 ID 由 `sources` 的来源种类和查询键组成；例如 `sources.mock.running_avg` 对应 `mock/running_avg`。这里的 `mock` 是采集来源种类，物理名称仍可以是 `rtp_llm_running_stream_size`。Python producer 在 `produced` 中直接声明完整 ID。

命名遵循以下规则：

- 指标 namespace 和名称使用小写 `snake_case`，同一个 ID 在不同集合中保持相同的取数定义、单位、值类型与身份标签。`required` 是集合内的采集要求，可以按用途设置。
- 原始抓取可以使用 exporter 名，或明确的业务简称；映射只在指标 YAML 的 `promql` 中定义。Python 门禁和 view 均引用 `metric_id`，不再另建一套 exporter 到门禁名称的转换表。
- 聚合、分位数和统计口径体现在名称中，例如 `_sum`、`_mean`、`_max`、`_p99`、`_per_engine`；仅有相同来源或单位不能合并 ID。对 priority 样本直接取平均，与先逐引擎求和再取平均，结果可以不同。
- 同一指标的不同角色或实例通过标签、`source`、`epoch` 选择；已有的多角色查询不为每个角色再定义相同 PromQL。需要多个图例时定义本地曲线，均绑定同一个 `metric_id`。
- 改展示名、颜色或面板不改变指标 ID。改变业务含义、归约口径或单位时使用明确的新 ID；所有实际定义及 SHA 随运行冻结，报告读取冻结产物。

例如，一条 `running_avg` 查询可在 view 中分别绑定 `role: PREFILL` 和 `role: DECODE`，使用不同曲线 ID 和中文图例。两条曲线都来自 `mock/running_avg`；门禁也通过这个 ID 和明确的标签、时窗读取数据。

## 指标集合与查询

场景通过 `execution.monitoring.query_plan` 选择 `config/monitoring/` 中的集合。`default.yaml` 是公共默认集合，case 集合通过 `include` 引用；它与 `config/report_views/default.yaml` 全量诊断视图是独立配置。默认集合的文件名由 `monitoring.query_plan.DEFAULT_PLAN` 声明。

`config/monitoring` 定义指标身份、采集查询及单位、标签、测量口径元数据。`parameters.observation.inputs` 将这些指标按标签映射为 case 使用的本地字段；`observation.windows`、采样与 capture 设置限定取证范围和完整性要求。`parameters.analysis` 定义证据的解释规则，`parameters.checks` 绑定结果指标、窗口集合及门槛。指标采集、输入投影、测量计算与判定分别由其拥有者校验，输入绑定不创建新的采集后端。

```yaml
metric_plan_schema_version: 5
include: [default.yaml]
sources:
  mock:
    queue_depth:
      promql: sum(rtp_llm_wait_stream_size${selector})
      unit: requests
      value_kind: gauge
      labels: []
      exported_metrics: [rtp_llm_wait_stream_size]
```

上例的指标 ID 是 `mock/queue_depth`。`sources` 支持 `mock`、`client`、`master`；采集器仅注入 `${selector}` 和 `${window_ms}`，运算使用 PromQL 原文。默认 `mode: evaluated`；需要原始抓取时间的门禁声明 `mode: scrape`，此时 PromQL 只接受原始指标名加 `${selector}`。`labels` 声明每条结果必须携带的身份标签，`required: true` 表示缺失该查询会使监控归档失败。

`include` 合并集合，`exclude` 按完整指标 ID 显式删除。重复定义、循环引用和删除不存在的指标均失败；本契约不支持同 ID 覆盖，要调整口径时修改所属集合，或新增不同 ID 并排除旧指标。展开后的定义及 SHA 写入编译产物与运行归档。

难以用 PromQL 表达的请求归因、阶段统计由注册 Python producer 实现，在 `produced` 声明输出 ID、`producer`、`source_type`、`unit`、`value_kind` 和 `labels`。请求窗口计算额外声明 `calculation`；专属计算由 producer 的输出契约提供。`source_type` 记录实际来源：`prometheus`、`client_journal` 或 `debug_api`；计算后的数值仍保留原始来源，不用 `derived` 代替来源。是否执行生产器由 `producer` 决定，与来源标签无关。产物记录模块、源码 SHA、计算口径和证据引用；原始证据不能作为缺失 Prometheus 指标的自动替代。

`measurement` 将来源与统计语义分开：

| 字段 | 含义 |
|---|---|
| `method` | 计算方法名称；实现由 PromQL 或注册 Python 能力负责，不是表达式 DSL |
| `population` | 请求 cohort、完成窗口、角色/引擎群体等统计对象 |
| `accuracy` | `request_ledger`、`sampled`、`histogram_estimate` 或 `counter_delta` |
| `requires_request_identity` | 是否依赖请求身份进行归因、去重或终态核对 |

`produced` 的 `measurement` 由计算实现生成，YAML 不接受手写覆盖；加载时校验 producer 输出契约，发布时再次校验。公共查询集合显式记录 PromQL 口径。`request_ledger` 表示对已核实流水计算，不保证流水天然完整；完整性和时窗覆盖仍决定测量有效性。来源或精度标签都不能独自决定门禁是否 PASS。

直方图 p99 与逐请求 nearest-rank p99、PromQL 的 lookback 均值与按实际 scrape 时刻计算的整窗引擎等权均值、全 fleet hit ratio 与 survivor 过滤后的 counter 差分分别使用不同 ID。门禁声明自己使用的 ID、单位和时窗，view 选择需要展示的投影，不把相近曲线作为门禁替身。已由请求流水覆盖且没有展示消费者的查询在所属 case 集合中用 `exclude` 去掉；公共默认集合保留供其他用途选择。

## 可执行请求计算

`calculation.calculator` 选择 `analysis.request_metrics.CALCULATORS` 中的注册函数，不解析任意表达式。支持 `token_throughput`、`request_rate`、`mean`、`quantile`、`success_share` 和 `inflight`。`window` 引用 producer 支持的观测窗口，`selection.time_basis` 选择 `arrival`、`completion` 或 `lifetimes`，`selection.status` 选择 `all`、`ok` 或 `non_ok`，`bucket_s` 指定桶宽。字段、单位、选择组合、百分位和窗口引用在加载时校验；未知字段、缺字段、非法组合或不完整账本失败。

```yaml
calculation:
  calculator: token_throughput
  window: measurement
  selection:
    time_basis: completion
    status: ok
  token_field: input_len
  bucket_s: 1
```

计算器实际按窗口及选择条件读取请求，使用半开时间桶；最后不足一桶时按实际桶宽归一化。请求 ID 必须唯一，完成时间由发送时间与耗时计算，不回退到另一时钟。`inflight` 使用完整请求生命周期，在桶末边界计数；空均值或分位数为缺失，完整账本证明的零计数才是零。

`method` 自动记录注册 calculator，`population` 自动记录窗口、时间基准和状态选择，来源、单位与测量分类由计算器声明。修改选择或桶宽会改变实际计算与归档口径。复杂的路由、survivor 差分及冻结门禁仍由 case Python 实现，通过 `metric_contract` 绑定真实实现函数并生成口径；YAML 不能覆盖它们的方法或统计对象。归档保留展开的计算参数、实现模块及源码 SHA；报告读取冻结结果，不按当前配置重新计算。

## 选择统计口径

判断是否可以改用 Prometheus 时，先核对统计对象、时间边界和精度，而不是比较展示名：

| 需求 | 选择依据 |
|---|---|
| 全局发送/完成速率、服务状态趋势 | 优先使用 PromQL；`rate()` 是采样窗口内估计，不等于逐秒流水计数 |
| 延迟分位数 | exporter histogram 满足精度要求时用 `histogram_quantile`；需要指定发送 cohort 的精确分位数时读取已核实的请求流水 |
| 同一批请求的成功率、终态、重试、路由归属、TPOT | 保留请求身份及配对字段，由 producer 计算；现有聚合 exporter 不保留这些关联 |
| 整窗引擎 TPS | 区分 PromQL lookback 聚合与实际 scrape 样本的引擎等权均值；不同采样与归约口径不能互换 |
| counter 命中率 | 明确全 fleet 或 survivor 群体、差分边界及 counter reset 检查；来源均可为 Prometheus，但统计对象不同 |
| HTTP 可回读与 debug ledger | 可回读不是 scrape `up`；用 debug 证据必须声明具体字段和 `debug_api` 来源，不能把 engine 负载当请求分配归属 |

指标定义按实际消费准入：检查使用的测量、报告曲线，以及有明确排障用途的诊断指标。注册计算能力不要求把所有可计算值加入每个 case 的 plan；只在所属场景声明需要的输出。分子、分母等算法中间值保存在冻结分析证据中，除非需要独立消费，否则不另注册指标。请求账本可同时支持精确门禁和少量专属曲线，普通趋势优先读取 Prometheus；展示采样缺失不能以账本重算补图。

每条指标的权威口径在冻结的 `measurement` 中，包括 `method`、`population`、`accuracy` 和 `requires_request_identity`；门禁绑定的 ID 指定判定权威，view 绑定的 ID 指定展示口径。`exclude` 是集合选择，不是指标失效后的替代链。若明确需要两种口径作诊断，应分别保留 ID、声明用途，不计算“数值应相同”的漂移告警。

请求身份需求不适合通过给 Prometheus 增加逐请求标签解决。若 exporter 预先维护固定 cohort 或路由维度的统计，PromQL 可以查询这些结果，但终态配对与归因仍由 exporter 完成，且需要单独验证其 cohort 和完整性契约。

## 计算与落库

`analysis.statistics.percentile_nr` 使用 nearest-rank，不舍入；空样本返回 `None`，实测零返回 `0`。报告的空分位数不能补零，门禁仍由样本数、完整性与测量有效性决定是否可判定。

`analysis.time_buckets.TimeBuckets` 显式声明 `origin_epoch_s` 和 `width_s`，接收发送或完成的绝对毫秒时间戳，生成半开桶。测量起点分桶与自然秒分桶都合法，公共代码不替 case 选择；桶边界与报告的显示起点是两个概念。落库时间均为绝对秒，渲染时才减显示起点。发送 cohort 的终态计数与完成时窗的终态计数不能混为同一指标；HA 窗口只接受 `send_start_epoch_ms`，不回退到终态记录时间。

`series_row` 构造 producer 的序列信封，必须显式传入来源实例、标签和环境代次。`source` 表示该序列所属的数据流，`source_type` 表示原始数据的物理来源，`producer` 表示计算能力，三者独立。`epoch` 来自冻结的环境代次或资源句柄，不能写死为 `"1"`；Master 重启由 incarnation 表达，不自动产生新的环境代次。`publish` 统一校验定义、标签及有序有限样本；全空值序列标为 `ABSENT`。证据文件缺失直接报错，不能生成空曲线冒充已完成采集。

只有 `sources` 展开为真实查询，`produced` 即使标为 `source_type: prometheus` 也不会新增查询。`export_metrics` 只转换已归档的查询结果并保留 producer 输出，不发起抓取。查询展开可用 `queries_for_targets` 核对，实际采集清单保存在 `telemetry/<epoch>/queries.json`；报告分类审计区分已展示、门禁证据及显式诊断指标，未分类指标报错。

所有 `sources` 查询必须用 `exported_metrics` 列出依赖的物理指标名称；`mode: scrape` 的依赖必须与原始 selector 一致。`environment.metric_whitelist` 是 Java exporter 的暴露过滤器，query plan 是查询选择，两者不合并。编译期检查显式过滤器及其 profile 覆盖不会排除所选查询的依赖；未声明过滤覆盖时保留 Java 策略，运行时仍需按查询的 `required` 和覆盖契约核验实际数据。

视图 YAML 的 `curves` 用本地曲线 ID 声明 `metric_id` 和 `labels` 选择，并设置名称、颜色、轴和换算；面板用 `curve_ids` 选曲线。Python program 用 `case.metric(id)` 声明依赖，编译时拒绝未定义 ID。运行时 `MetricStore.select` 显式选择标签与时间窗、检查样本数及最大间隔，`reduce` 只对单条已选序列归约；缺失数据抛出 `MetricUnavailable`，定义冲突抛出 `MetricContractError`，均不能补零。

门禁的共享运行身份、制品与流量 SHA、拓扑和容量结构由 `workload.run_provenance.validate_gate_provenance` 校验，采集端和消费端使用同一合同。Fetch、请求 cohort 等测量前提仍由所属分析器校验。字段集合可从已有测量定义推导时不重复列举；消费单位和身份维度属于算法约束，不能从生产配置直接复制。诊断使用结构化结果及错误代码，不按错误文案决定是否忽略采集失败。

## 按需采集与数据源扩展

公共层不按 case 名分支。query plan 是能力目录，不意味着采集目录中所有指标。编译器按每个变体展开 `case.metric()` 的门禁与算法输入、view 曲线的 `metric_id`、view 的显式 `diagnostic_only`，冻结三者并集及各自的消费者。默认 view 只展示已选中的归档，不扩大采集集合。未知依赖在启动前拒绝；门禁 producer 的输入也必须通过 program 显式声明，不能等运行时再加载整份目录。

查询声明物理依赖，注册 producer 为必要的额外采集声明 `collection.source` 和 `collection.field`。这两个字段由 Python 输出契约生成，不能在 YAML 手写覆盖。编译产物的 `implementation.monitoring_query_plan.collection` 冻结 Prometheus 物理白名单、证据字段及适配器实现 SHA；运行期的 `session.json` 和 `queries.json` 保留实际 target 与计划。`implementation.monitoring_query_plan.definition.demand` 保存门禁、曲线和诊断需求；运行时校验计划摘要，使用冻结定义。producer 发布及离线重判沿用冻结清单，不按当前目录补回未选中的指标。报告对已采集指标保持严格分类；仅作算法输入的项归为 `GATE_INPUT`。

`monitoring.collection_plan` 从选中查询生成按来源的白名单。没有查询的来源不启动 scrape job；动态客户端同样遵守选择，未选中的客户端不开 exporter。每个 job 使用 Prometheus `metric_relabel_configs` 的 keep 规则，只入库声明的物理指标，保留原有身份标签和自动生成的 `up`。查询使用 histogram 时显式列出所需 bucket、sum、count，不能用同名 gauge 替换。过滤不解析或重写 PromQL，复杂表达式的依赖由定义维护者负责核对。没有原生查询也没有 probe 的纯事件证据计划不启动 TSDB。

入库过滤减少序列和存储量，不减少 exporter HTTP 响应生成或网络传输；只减少查询或图表也不会减少 scrape。Master 不增加生产指标、接口或日志。现有 Java 暴露白名单仍受编译校验；client、engine 的源端导出能力可按其自身合同扩展，不把服务端成本下降当成入库过滤的既有效果。

case 的 `CaseDefinition.sources` 声明额外数据源能力，由自动发现的公共注册快照收集，选中的指标决定启动哪些能力和字段。源协议可声明必需的完整性字段，例如 Master 状态需要同时保留 HTTP 可回读状态。普通时序复用 exporter；现有只读 HTTP 接口通过 `monitoring.probe.PrometheusEvidence` 接入官方 `prometheus_client` custom collector。SDK 只负责本地 exposition 服务，Prometheus 触发 scrape 时才调用有超时和响应字节上限的适配器，没有 Python 定时采集线程，也不新建第二套 TSDB。每个 probe 与原生 exporter 使用同一个 run 所有的 Prometheus；没有原生查询但选中 probe 时仍启动 Prometheus，只有请求流水等无需 scrape 的计划才不启动它。

适配器只执行一次严格字段投影，不创建线程、写文件或回退到日志。接口不可达本身是观测时，可以显式产出可达性 0，其余字段缺失；成功响应的坏 JSON、缺字段、非有限数报错。probe 启动必须完成一次成功 scrape 才开始发流。SDK 的 `up` 表示适配器 exporter 是否成功，Master 的 HTTP 可回读是另一条源端观测，不能混淆；坏响应和样本/字节预算超限会在启动或收尾显式失败。

收尾先结束适配器读取，再从 TSDB 按实际 scrape 时间导出选中字段。`*.prometheus.json` 保留原始查询、时间边界、源地址与错误，协议证据 JSONL 从这些样本生成；producer 再按冻结定义落入统一 `metrics.json`。不使用 PromQL lookback 补断档。物理来源仍是 `debug_api`，传输记录为 `prometheus`，不会因为用了 SDK 就改成生产 Master 原生指标。exporter、采集错误与收尾由共同资源所有者管理，结束 probe 后再关闭 TSDB；失败时仍尝试保存其他采集证据并回收所有进程。

接口协议与字段语义独立于测试目的时放公共源模块；专属协议解释放 `cases/<case>/`，通过相同注册入口接入公共生命周期。采样新增源遵循 [新增 case](../development/adding-cases.md) 的必要性与字段完整性规则。请求 journal 是事件证据，由注册 producer 按冻结窗口计算，不需要转成逐请求 Prometheus 标签。
