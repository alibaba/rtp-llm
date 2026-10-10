# 框架结构

```text
config/scenarios/*.yaml   身份、环境、输入、阈值与视图选择
        ↓
cases/<case>/program.py   步骤、分支、检查与能力声明
        ↓
scenario compiler        校验并生成有类型的执行计划
        ↓
parallel runner          分配 lane，执行并清理 Master / Mock
        ↓
frozen evidence/metrics  原始证据、指标定义与完整序列
        ↓
case analysis/report     明确判定后装配 HTML bundle
```

## 格式版本

版本字段采用 `<format>_schema_version`，字段名标明格式，整数标明该格式的版本。各格式独立演进，数字不表示项目版本或代码是否过时。

| 格式 | 版本字段 | 当前值 |
|---|---|---|
| case YAML | `case_schema_version` | 2 |
| 指标集合 | `metric_plan_schema_version` | 2 |
| 报告视图 YAML | `report_view_schema_version` | 1 |
| CI suite | `suite_schema_version` | 2 |
| mode profile | `mode_profiles_schema_version` | 1 |
| Python 内部 program document | `program_schema_version` | 1 |
| 编译 / 列举的实例清单 | `instance_catalog_schema_version` | 1 |
| 子执行器结果 / 父 runner 汇总 | `scenario_results_schema_version` / `run_summary_schema_version` | 1 |
| runner 计划 / 端口租约 / 耗时缓存 | `runner_plan_schema_version` / `lease_schema_version` / `timings_schema_version` | 1 |
| 冻结指标归档 | `metrics_schema_version` | 1 |
| 报告图表 / bundle manifest / 分析封装 | `report_spec_schema_version` / `report_manifest_schema_version` / `report_analysis_schema_version` | 1 |
| 报告运行信息 | `run_meta_schema_version` | 1 |
| workload 证据 / 分析 | `workload_evidence_schema_version` / `workload_analysis_schema_version` | 1 |
| 请求与引擎关联证据 / 客户端请求记录 | `request_engine_evidence_schema_version` / `client_record_schema_version` | 1 |
| 观测快照 / 冻结观测窗口 | `observation_snapshot_schema_version` / `observation_window_schema_version` | 1 |
| 性能证据 / 性能分析 / cache 分析 | `performance_evidence_schema_version` / `performance_analysis_schema_version` / `cache_scale_in_analysis_schema_version` | 1 |
| 实验归档 manifest / 请求计划 manifest | `archive_manifest_schema_version` / `request_plan_manifest_schema_version` | 1 |
| Master 模板 / 流量保真度分析 | `master_template_schema_version` / `traffic_fidelity_schema_version` | 1 |

case YAML 通过 `program: default` 生成内部 program document；后者包含 stages 和 typed reference，不是旧用户 YAML 的兼容入口。读取方校验本格式的版本字段及整数类型，拒绝其他格式或旧通用版本字段，不能根据文件名或版本数字猜测格式。报告生产器组装的内存图表可以暂不带版本字段，由 bundle 写入器明确标记；落盘文件和离线读取必须带有相应字段。

真实采集输入、固定合成流量 profile 及 Java 共享性能 / 流控协议保留它们原有的 `schema_version`，由相应协议读取器校验，不进入 case、执行计划、指标和报告的读取边界。它们的语义与准入见[数据目录](../../data/README.md)。调整这类字段需要同步固定输入及协议双方。

版本字段的名称属于格式身份；字段或语义发生不兼容变化时升级该格式，并同步其读写双方，不为数字一致整体替换。配置分层见[新增 case](../development/adding-cases.md)。

## 代码归属

| 目录 | 内容 |
|---|---|
| `cases/<case>/` | 该 case 的 program、action、输入合同、纯分析、指标生产、报告与离线重判 |
| `cases/config.py`、`cases/registry.py` | 公共 case 构建接口与显式注册 |
| `scenario/` | 配置编译、基础 action、资源句柄、阶段执行与清理 |
| `runtime/` | 端口/进程/Java/HTTP/gRPC 等协议与资源底座 |
| `workload/` | 连续负载的通用采集生命周期、证据核对、来源与全量视图 |
| `analysis/` | 可复用统计及输入保真度分析，不定义某个 case 的门槛 |
| `monitoring/`、`reporting/` | 指标查询/归档与图表/spec/bundle 的公共能力 |
| `traffic/`、`artifacts/` | 流量构造、播放与制品管理 |

归属由语义决定，不由文件名或当前调用数决定：包含特定阶段、门槛、指标输出或报告身份的代码属于 case；同一合同能被不同 case 使用的能力才放在公共组件。注册表可以指向专属实现，公共执行器不按 case 名分支。

集中归属不合并职责。`analysis.py` 根据显式输入计算，不访问网络、进程或报告；`actions.py` 操作现场并取证；`metrics.py` 发布声明的指标；`report.py` 展示既定结论，不能隐式重判。工作流最终归档后，program 可声明 `REPORT_FINALIZER(directory)` 刷新冻结报告的曲线；该 hook 不改变门禁结论或重新发布数值指标。

## 执行与资源契约

每个 case 有 `default` program，变体只追加测试点。YAML 保存数据，不能写 action、输出引用或任意表达式；配置层次和 action 边界见[新增 case](../development/adding-cases.md)。

资源句柄绑定环境代次。重建环境前清理旧消费者及进程，旧句柄只能显式作为历史证据读取。动态 worker 添加保留尝试预算；移除不返还容量预算。端口租约由父 runner 拥有，子执行器启动前核对范围与最大拓扑。

阶段使用剩余 deadline，清理使用独立预算并逆序执行。启动部分失败也要回收已登记资源；未退出线程、未回收进程或清理异常不能成为 PASS。普通失败阻断依赖步骤，独立观察应放在最终判定之前。finding 只能声明具体检查，不豁免证据、超时和清理。

请求发出、Schedule ACK、Fetch、业务 FINISHED、取消和资源释放分别取证，不互相推断。服务启动及诊断 API 成功应答不代替业务完成。具体协议见[请求生命周期](request-lifecycle.md)。

指标和报告使用冻结定义与序列，不通过 debug API、日志或文件自动兜底；缺失来源显式报错。采集、门禁与交付规则分别见[指标契约](metrics.md)、[结果与指标](../development/results.md)和[报告契约](reporting.md)。
