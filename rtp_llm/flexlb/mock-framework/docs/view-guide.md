# 实验结果与 View

[返回使用说明](../README.md)

下面的命令均在框架目录执行。View 读取已完成实验；默认大盘及指标覆盖另见 [大盘说明](../views/README.md)。

## 实验结束后查看基础 view

默认大盘的分区、聚合、地址筛选、标签来源和未实现观测见 [views/README.md](../views/README.md)。

Case 省略 `view` 时默认使用框架自带的 `views/default.json`，与 Case 所在目录无关；需要自定义大盘时，显式配置 `view: <路径>` 覆盖默认值。执行时把这份原生 Grafana JSON 保存为结果目录下的 view.json，避免以后修改源文件改变已完成实验的展示。View 用于实验结束后的复盘，不参与发压、调度、Prometheus 采集或门禁判定，也不会自动启动查看服务。运行准备阶段会读取并校验大盘文件，文件缺失或格式错误会使启动报错；未保存 view.json 的旧结果不会被自动补写。

```sh
uv run --locked mock-view /tmp/basic-result \
  --output /tmp/basic-view \
  --prometheus "$PROMETHEUS_BIN" \
  --grafana-home "$GRAFANA_HOME"
```

`GRAFANA_HOME` 是包含 bin/grafana、conf 和 public 的 Grafana 安装目录；输出目录必须是新的、位于实验目录之外的目录。命令先在回环地址提供回看页面并打印 URL，再启动独立的 Prometheus 与 Grafana。页面显示指标服务的启动状态，启动失败仍可查看配置和门禁。Ctrl+C 或 SIGTERM 统一回收页面服务和所拥有的子进程；页面关闭不等于停止服务。端口冲突直接报错，不连接其他查看会话或已有服务。

查看服务复制已完成实验的 TSDB，原始结果与数据不被重新打开写入；它不启动任何 Engine/Master/Client、不重新执行检查。多个查看会话使用不同输出目录和端口，可以独立查看同一份实验。

一个页面展示原始 Case、完整配置/启动配方、组件生效配置、代码版本/工作区差异、Phase/Step 时间线、保存的门禁，以及 Grafana 在线曲线。默认大盘所有 row 折叠，Prefill/Decode 分区并显式过滤各自 role；唯一线上参考是 RTP-LLM 大盘（包括其中的 FlexLB 分区）。当前已上报的 Mock 观测与 real 待埋点面板分开说明，ViT 和共享 JVM 的宿主资源缺口不伪造数据。Client 的 TTFT/请求耗时不能冒充 Engine 延迟。完整来源与覆盖见 [指标覆盖](../views/coverage.md)。每实例曲线保留 instance/role 等原有标签，支持 Grafana 自己的缩放与曲线选择。

点击门禁卡片定位其保存的指标窗口；执行侧 metric_window 事件记录真正使用的起止时间，展示层不根据当前实现重新计算。原始 PromQL 与非指标检查只标记查询点，图表显示附近上下文，不解析表达式推导窗口。每个 Step 只导入一个图表标注：普通操作标在开始时刻；有门禁时使用查询点或已记录的查询窗口，并在注释中展示接口、判定、预期条件、实际值（含单位）和错误。Phase 和 Step 的完整开始/结束记录仍保留在执行时间线。

事件线默认全部显示。在 Case 引用的 View JSON 顶层可加 `"annotation_steps": ["traffic", "kill_a", "kill_b"]` 指定默认选中的 Step，空数组表示默认全部隐藏；省略表示全选。它只属于展示配置，不扩展 Step，也不改变门禁卡片或执行记录。查看时校验重复和未知 ID；错误只影响指标展示，不改变实验结果。页面“事件线”可以逐项勾选、全部显示或全部隐藏，独立 Grafana 页面也有对应的 Step 多选变量。筛选只改变查询标签，所有标注仍然保留，不重启指标进程、不重跑实验。曲线在线查询保留的 TSDB，不固化进 HTML。

实验判定、产物完整性和查看服务状态分别展示：

- result.json 是核心记录，必须含有效的 status、带时区的 started_at/finished_at；缺失或损坏时命令报告文件和字段，拒绝猜测实验结果。只查看已完成实验。
- 配置、代码版本、接口元数据、采集记录和事件/门禁日志属于附加记录。缺失或损坏时显示具体文件；JSONL 中损坏的行标出行号，保留其他有效行。页面说明记录不完整，不把缺失门禁当作通过。没有配置也能按保存的 step 名展示门禁。
- TSDB、view.json 或依赖程序不可用，或指标进程启动失败/运行中退出时，指标区域说明原因，配置、门禁和时间线仍可读。已启动的指标进程统一回收，不自动重试。完整异常和进程 stdout/stderr 留在查看目录；修复后重新运行查看命令，使用新输出目录。
- ERROR/FAIL/CANCELLED、清理错误、采集不完整以及缺失指标均如实显示，缺失数据不补零。即使查看服务故障，也不改写实验的 PASS/FAIL/ERROR/CANCELLED。

实现分为 CLI（view/__main__.py）、服务生命周期与固定只读 HTTP 路由（view/service.py）、产物读取（view/records.py）、Grafana 配置与事件导入（view/grafana.py）、页面（view/page.html）。进程生命周期直接复用现有 ProcessSupervisor，不复制进程管理实现；Runner 只保存实验事实和 view 快照，不依赖展示层。

查看只读取指定的一次实验，不增加实验索引、实例历史台账或运行中控制。页面只轮询查看服务自身的指标服务状态，不轮询 Case 接口或重新判定门禁。自定义 view 继续使用原生 Grafana JSON；复用线上大盘时，需要把查询和标签改为本地实际存在的指标，不自动兼容线上 VIP 或 KMonitor。
