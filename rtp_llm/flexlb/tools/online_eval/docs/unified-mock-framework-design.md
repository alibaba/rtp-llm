# Mock 测试框架统一设计与落地状态

## 目标与边界

本地功能测试、场景测试、压测，以及 Whale 寄生/独立部署共享同一套 mock 引擎行为与 master 配置生成能力。运行模式决定网络地址、资源归属和默认观测方式；master 模式决定调度决策与派发方式。A/B、远端 real 对比、报告和归档是可选工具，不应成为启动引擎的前置条件。Whale 默认以 KMonitor 为主，不持续写大型 JSONL。

这次改动保留原有入口和参数。现有 `flexlb_cfg.py` 仍负责生成完整 master 配置，模式表只选择其已存在的四个 profile，不复制配置字段。后续若要求把全部参数迁到 YAML，应在该配置生成层做一次迁移，并对产出的 JSON 做等价检验。

## 两层模式表

[`mode_profiles.yaml`](../mode_profiles.yaml) 定义运行模式 `functional`、`scenario`、`stress`、`whale_embedded`、`whale_independent`，以及 master 模式 `sb`、`sn`、`wb`、`wn`。[`mode_profiles.py`](../mode_profiles.py) 校验表项和现有 profile 轴的一致性，给入口返回可执行计划。

| 运行模式 | 地址策略 | 默认观测 |
| --- | --- | --- |
| 功能、场景 | 本地引擎虚拟 loopback；macOS 退回同 IP 不同端口 | 有界 JSONL + 报告 |
| 压测 | 本地引擎虚拟 loopback | 运行目录 JSONL + 报告；归档时限制原始日志大小 |
| Whale 寄生 | 无外部 Fetch 时允许虚拟 loopback；有外部客户端时使用真实 Pod IP | KMonitor |
| Whale 独立 | 每个引擎一个 Pod，使用各自真实 Pod IP | KMonitor |

`sb/sn/wb/wn` 分别映射 SINGLE/BATCH、SINGLE/NON_BATCH、FIXED_WINDOW/BATCH、FIXED_WINDOW/NON_BATCH。地址和 master 模式分开选，避免把“能否从 frontend 访问引擎”误当成调度功能。

配置的 `observation` 是默认功能表，不是强制全开。JSONL、报告、KMonitor 必须由各入口按需启用。模式表已接入本地 case 启动、压测默认 profile 和 Whale 寄生 bundle；Whale 独立镜像携带从同一 YAML 生成的 `mode_defaults.sh`，不要求 Pod 安装 Python/YAML。两种 Whale 模式默认不写请求级 JSONL，诊断时可用 `MOCK_EVENT_LOG_ENABLED=1` 显式开启。

## 网络地址和监控身份

引擎有两个不同的标识：

- **可达地址**是 master/frontend 发请求的真实 `IP:port`。若 frontend 会 Fetch，寄生模式不能发布 `127.x.y.z`；独立 Pod 天然有各自 Pod IP。
- **逻辑监控身份**是 role、engine index、generation（现有上报以 `engine`、`engine_port` 等 tag 表示），用于区分同 Pod 中的多个引擎。`host_ip` 保持真实物理机地址，`container_ip` 保持真实 Pod 地址。

因此寄生模式下所有逻辑引擎同 Pod IP，现有只按 `host_ip` 分组的面板无法无损拆分它们。独立 Pod 虽有不同 `container_ip`，多个 CPU Pod 仍可能落在同一物理机，`host_ip` 也会合并；一 Pod 一引擎并不自动解决旧面板问题。若既要保留 `host_ip` 的真实物理机语义又不改面板，只能确保每引擎占独立物理机；通常应改用 `container_ip` 或 `engine`/`engine_port` 分组。不能把不可达的虚拟 IP 写进 `host_ip` 冒充地址。

## 扩展工具

- A/B 比较继续由 `stress/compare_ab.py` 负责门禁。`--html` 现在输出原指标表及 `ab_curves.html`；两次运行按相对秒对齐，缺失采样保留空值，稳态窗在每图说明中标明。曲线覆盖 QPS、TTFT、P/D TPS、在飞请求、KV 可用量/驱逐及局部差值，由现有通用 Chart.js 报告渲染器生成。
- 图表交互由 `stress/legend_interaction.js` 封装，只改变图例行为，不定主题：单击切换一条，双击隔离；只剩一条时再双击恢复全显，也提供“全选”按钮。
- `compare_case_runs.py` 比较功能/场景跑批的相同实例 ID、状态、失败断言与耗时，保留 `FINDING-CONFIRMED` 的独立语义；输出 JSON、离线表格、耗时曲线和可选档案，不擅自将发现探针判成普通失败。
- `experiment_archive.py` 可把 case、场景、压测、A/B 的运行目录和汇总打成一个 ZIP。`manifest.json` 记录执行状态、来源元信息、每个文件的大小、SHA-256 和完整度；结构化结果完整保存，超 2 MiB 的原始日志保留头尾并显式记录，疑似密钥文件名跳过。`parallel_runner.py --archive FILE`、压测 `EXPERIMENT_ARCHIVE_PATH`、`compare_ab.py --archive FILE` 均可生成档案。中断或聚合缺失的执行标为 `incomplete`。单档案是交接与阅读入口，不把原始日志的省略伪装成完整数据。

`remote_compare.py` 接受两个版本化 KMonitor 导出文件（real、mock），先核对流量指纹、master 模式、Fetch、P/D 规模、采样粒度及每条指标的名称、角色、单位、时空聚合口径；只比较共同采样时刻，缺口留空，产出离线 HTML、JSON 和可选单档案。它明确标为 `descriptive_only`，不把 KMonitor 聚合曲线当成请求级回归门禁。导出文件应包含 `schema_version=1`、`provenance` 和 `series`；每条 `series` 包含 `metric`、`role`、`unit`、`spatial_aggregation`、`temporal_aggregation`、`points[{t_ms,value}]`。平台查询及鉴权属于独立只读采集适配器，目前没有在本仓库内嵌凭据或特定部署地址。

## 下一阶段迁移

1. 将平台 KMonitor 查询结果转换为上述版本化导出格式；鉴权与查询留在平台适配层，不嵌入引擎骨架。
2. 如需让旧 `host_ip` 面板看到每个引擎，须有一引擎一物理机的放置保证；否则按 `container_ip` 或 `engine`/`engine_port` 分组，不能假造 `host_ip`。
3. 继续将 case/场景/压测的数据合成、任务编排、采集、断言与报告组件接到模式表；已有入口及执行结果格式保持兼容。

## 本地使用

```bash
python3 rtp_llm/flexlb/tools/online_eval/experiment_archive.py create \
  --kind scenario --source run=/path/to/run --out /path/to/run.experiment.zip
python3 rtp_llm/flexlb/tools/online_eval/experiment_archive.py inspect /path/to/run.experiment.zip
python3 rtp_llm/flexlb/tools/online_eval/stress/compare_ab.py \
  --run-a /path/to/a --run-b /path/to/b --html \
  --out /path/to/ab_summary.json --archive /path/to/ab.experiment.zip
python3 rtp_llm/flexlb/tools/online_eval/compare_case_runs.py \
  --run-a /path/to/case-run-a --run-b /path/to/case-run-b \
  --out-dir /path/to/case-ab --archive /path/to/case-ab.exp.zip
```
