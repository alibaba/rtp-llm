# 报告管线

所有新报告通过 `online_eval.reporting` 发布。原始采集和领域分析器保留各自的数据格式，页面生成不再读取采集日志、计算门禁或者更新运行状态。

```text
原始证据 + 配置
  → 证据校验 / 领域分析
  → analysis.json
  → 布局函数 + 公共组件
  → report-spec.json
  → 公共 Chart.js 渲染器
  → report.html
```

## 产物约定

默认路径为 `<output>/reports/<kind>/<id>/`。kind 为 `run`、`comparison` 或 `sweep`。不安全或带分隔符的 ID 转为文件名并附内容哈希，原 ID 留在 manifest 中。

每个报告包包含四个文件：

- `manifest.json`：类型、ID、producer、默认入口，以及三个产物的相对路径和 SHA-256。
- `analysis.json`：版本化信封；`result` 是领域分析结果，`run_meta` 是公共元数据。读取使用 `load_analysis()`，同时支持旧归档的裸结果。
- `report-spec.json`：完整展示输入，包含图表、表格、证据折叠块和 RunMeta。
- `report.html`：单个离线阅读入口，Chart.js、图例和多曲线交互均内嵌。

manifest 最后写入。`read_bundle()` 检查版本和文件摘要。原始证据留在实验目录，不复制到每个图表包；归档应保留实验目录布局。仅搬动报告包也能独立重渲染，但包外原始证据链接需要一起归档才能访问。

`canvas_report_gen.py --out DIR` 直接输出标准包；显式传 `--out page.html` 时另导出独立页面，以保留用户指定导出路径。框架内部调用使用目录形式。比较工具显式要求的 JSON 摘要输出仍保留；默认 HTML 不再使用历史的多套名字。

| 入口 | 默认报告路径（相对于指定输出目录） |
|---|---|
| workload | `reports/run/<instance-id>/report.html` |
| cache gate | `reports/run/cache-scale-in/report.html` |
| stress A/B | `reports/comparison/stress-ab/report.html` |
| cache gate A/B | `reports/comparison/cache-scale-in-ab/report.html` |
| workload A/B | `reports/comparison/<instance-id>/report.html` |
| case A/B | `reports/comparison/case-ab/report.html` |
| remote comparison | `reports/comparison/remote/report.html` |
| twin | `reports/comparison/twin/report.html` |
| timeline / sweep | `reports/comparison/timeline/report.html` / `reports/sweep/sweep/report.html` |

## 分析与显示

`workload/evidence_analysis.py` 负责读采集证据、gap/journal 审计和运行状态修正；runtime 显式调用分析，再把结果交给 `workload/report.py`。`write_report()` 只消费分析结果，不重新判定有效性。

cache gate 的 `analyze()` 负责 baseline、门禁窗口和显示用滚动窗口，`build_spec()` 只组织分析结果和已有采样。判定阈值、引擎 incarnation 检查、零分母留空、缩容跨界窗口隐藏规则保持不变。

`reporting/statistics.py` 提供显式边界的窗口选择、counter delta/reset、nearest-rank percentile。请求窗口默认半开；计数器端点窗口明确声明包含右边界。领域 verdict 不合并成一个万能判定器。

公共组件包括图表、KPI、表格、details、链接。布局函数可各自组合，禁止入口拼 HTML 后再追加到页面。timeline/sweep 已使用共同 Chart.js 页面；同指标跨 run 的 Y 范围保持一致，缺口不补零，scatter 保留颜色、点大小及约束网格。

## RunMeta 与比较条件

RunMeta 字段固定为 `identity / implementation / workload / configuration / environment / clock / evidence`，缺失用 null 表示，不能当作已验证。旧入口通过已知元数据映射，不伪造原始归档没有的 commit 或证据。

公共 `compare_controls()` 递归比较全部控制字段，支持 required 与 allowed JSON Pointer。新增控制字段会参与比较；required 缺失或 null 返回 UNKNOWN。cache gate A/B 比较完整控制配置，明确排除文件位置和被比较的历史 master 身份，另核对 mock JAR 哈希与流量语义。workload / remote 保留各自既有的必需条件。

执行状态、证据有效性、可比性和领域判定分别保留，描述性比较不会升级为性能 PASS。旧 stress A/B 的警告/退出码政策没有在此次重构中改成另一种门禁。

## 离线重渲染

从工具目录执行：

```sh
python3 -m online_eval.reporting /path/to/report-spec.json --out /tmp/report.html
```

这个过程只读取 spec 和随代码安装的渲染资源，不接触原始遥测，也不改变 analysis 或实验结果。历史报告不原地重写；需要新页面时显式运行对应分析入口。

## 本次验证

2026-09-21：全量 Python 回归 985 项通过；随后新增的两个 CLI 产物检查随报告专项通过。图例、多曲线交互合约检查与生成页面内联 JavaScript 语法检查通过。四组代表性报告共 40 个面板与旧路径数据一致；五组 cache gate 样本的原有分析字段与判定逐项一致，覆盖 PASS、FAIL、INVALID。

本地 `file://` 页面预览被浏览器安全策略阻止，本次未作视觉验收。以上验证不代表重新执行了远端规模性能实验。

监控曲线的数据来源约束见 [monitoring-pipeline.md](monitoring-pipeline.md)。新运行的曲线由 Prometheus 查询生成；专用请求/接口/日志证据与曲线分开，报告层不能据此补算曲线。
