# 配置目录

配置保存手写测试输入，运行结果写入 `run/`。

| 位置 | 内容 |
|---|---|
| `scenarios/` | case 的身份、环境、参数、变体和视图选择，见[场景配置](scenarios/README.md) |
| `suites.yaml` | 默认 suite 与 CI 的 `case::variant` 清单 |
| `monitoring/` | 指标集合、PromQL 与 Python 输出声明，见[指标配置](monitoring/README.md) |
| `report_views/` | 指标绑定、面板和展示属性 |
| `performance_presets.json` | 引用 `data/performance/` 采集档案的登记表 |
| `mode_profiles.yaml` | 链接到根目录的运行模式表 |

字段顺序及覆盖见[新增 case](../docs/development/adding-cases.md)。采集档案的身份、完整性、块数口径与测试派生规则见[数据规则](../data/README.md)，不在配置登记表重复维护观测真值。
