# Mock 测试框架统一设计与落地状态

## 目标与边界

本地功能测试、场景测试、压测，以及 Whale 寄生/独立部署共享同一套 mock 引擎行为与 master 配置生成能力。运行模式决定网络地址、资源归属和默认观测方式；master 模式决定调度决策与派发方式。A/B、远端 real 对比、报告和归档是可选工具，不应成为启动引擎的前置条件。Whale 默认以 KMonitor 为主，不持续写大型 JSONL。

这次改动保留原有入口和参数。现有 `flexlb_cfg.py` 仍负责生成完整 master 配置，模式表只选择其已存在的四个 profile，不复制配置字段。后续若要求把全部参数迁到 YAML，应在该配置生成层做一次迁移，并对产出的 JSON 做等价检验。

## 两层模式表

[`mode_profiles.yaml`](../mode_profiles.yaml) 定义运行模式 `functional`、`scenario`、`stress`、`whale_embedded`、`whale_independent`，以及 master 模式 `sb`、`sn`、`wb`、`wn`。[`mode_profiles.py`](../mode_profiles.py) 校验表项和现有 profile 轴的一致性，给入口返回可执行计划。

| 运行模式 | 地址策略 | 默认观测 |
| --- | --- | --- |
| 功能、场景 | 本地引擎虚拟 loopback；macOS 退回同 IP 不同端口 | 有界 JSONL + 报告 |
| 压测 | 本地引擎虚拟 loopback | 采样 JSONL + 报告 |
| Whale 寄生 | 无外部 Fetch 时允许虚拟 loopback；有外部客户端时使用真实 Pod IP | KMonitor |
| Whale 独立 | 每个引擎一个 Pod，使用各自真实 Pod IP | KMonitor |

`sb/sn/wb/wn` 分别映射 SINGLE/BATCH、SINGLE/NON_BATCH、FIXED_WINDOW/BATCH、FIXED_WINDOW/NON_BATCH。地址和 master 模式分开选，避免把“能否从 frontend 访问引擎”误当成调度功能。

配置的 `observation` 是默认功能表，不是强制全开。JSONL、报告、KMonitor 必须由各入口按需启用；目前模式表已接入本地 case 启动和 Whale 寄生 bundle，压测及 Whale 独立入口的完整接线仍待迁移。

## 网络地址和监控身份

引擎有两个不同的标识：

- **可达地址**是 master/frontend 发请求的真实 `IP:port`。若 frontend 会 Fetch，寄生模式不能发布 `127.x.y.z`；独立 Pod 天然有各自 Pod IP。
- **逻辑监控身份**是 role、engine index、generation（现有上报以 `engine`、`engine_port` 等 tag 表示），用于区分同 Pod 中的多个引擎。`host_ip` 仍表示真实 Pod 地址。

因此寄生模式下所有逻辑引擎同 Pod IP，现有只按 `host_ip` 分组的面板无法无损拆分它们。仅靠修改 tag 不能使旧面板自动按引擎分组。无需改面板又要一 IP 一曲线，必须使用独立 Pod；或另行实现真实可路由的每引擎 IP，不能把不可达的虚拟 IP 写进 `host_ip` 冒充地址。

## 扩展工具

- A/B 比较继续由 `stress/compare_ab.py` 负责门禁。`--html` 现在输出原指标表及 `ab_curves.html`；两次运行按相对秒对齐，缺失采样保留空值，稳态窗在每图说明中标明。曲线由现有通用 Chart.js 报告渲染器生成。
- 图表交互由 `stress/legend_interaction.js` 封装，只改变图例行为，不定主题：单击切换一条，双击隔离；只剩一条时再双击恢复全显，也提供“全选”按钮。
- `experiment_archive.py` 可把 case、场景、压测、A/B 的运行目录和汇总打成一个 ZIP。`manifest.json` 记录每个源文件的大小、SHA-256 和完整度；结构化结果完整保存，超 2 MiB 的原始日志保留头尾并显式记录，疑似密钥文件名跳过。`compare_ab.py --archive FILE` 可把 A、B 两次运行和比较产物一起封装。单档案是交接与阅读入口，不把原始日志的省略伪装成完整数据。

本轮只接通本地 A/B 档案入口。远端 mock/real 对比应复用同一比较数据结构，但要先固定流量、版本、配置、时间窗和指标口径，再加远端采集适配器；不宜直接把不等价的 KMonitor 曲线送进回归门禁。

## 下一阶段迁移

1. 把压测和 Whale 独立部署入口接到运行模式表；保留原 CLI 默认值并做配置产物比对。
2. 将案例、场景和压测的 `result/aggregate/meta/evidence` 统一登记到档案清单；对大 JSONL 采用采样摘要而非在 Whale 上写文件。
3. 为远端 real/mock 对比增加只读采集适配器和来源校验；报告复用同一曲线组件。
4. 若寄生模式要求旧 `host_ip` 面板看到每个引擎，先验证平台是否支持真实路由别名；否则使用 `engine`/`engine_port` 分组或独立 Pod。

## 本地使用

```bash
python3 rtp_llm/flexlb/tools/online_eval/experiment_archive.py create \
  --kind scenario --source run=/path/to/run --out /path/to/run.experiment.zip
python3 rtp_llm/flexlb/tools/online_eval/experiment_archive.py inspect /path/to/run.experiment.zip
python3 rtp_llm/flexlb/tools/online_eval/stress/compare_ab.py \
  --run-a /path/to/a --run-b /path/to/b --html \
  --out /path/to/ab_summary.json --archive /path/to/ab.experiment.zip
```
