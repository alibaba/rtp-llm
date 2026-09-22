# 配置目录

配置是测试输入，不是运行结果。根目录历史兼容文件 `mode_profiles.yaml` 与这里的同名软链接指向同一份模式表。

| 位置 | 内容与用途 |
|---|---|
| `scenarios/` | 当前 Python case 的 14 份 YAML 定义，按能力分组；由 `scripts/commands/list_cases.py` 和 `scripts/commands/run_cases.py` 加载。旧扩展功能矩阵已从当前源码移除。 |
| `suites.yaml` | core、functional、workload 等 suite 的选例、覆盖及采集档位。 |
| `scale_cases/` | 真实流量单 run 门禁配置与下游 A/B 分析策略；不进入日常回归。 |
| `report_views/` | 报告视图定义及参数扫描示例。 |
| `perf_presets/` | 规模实验的性能参数预设。 |
| `traffic_profiles/` | 流量重放时使用的前端 profile 配置。 |
| `load_client_env.txt` | Java load client 的环境变量清单。 |
| `mode_profiles.yaml` | 指向根目录同名表的链接，供模式配置统一管理。 |

用例结构及实例列表见 [scenarios/README.md](scenarios/README.md)。
