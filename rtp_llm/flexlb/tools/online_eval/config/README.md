# 配置目录

配置是测试输入，不是运行结果。根目录历史兼容文件 `mode_profiles.yaml` 与这里的同名软链接指向同一份模式表。

| 位置 | 内容与用途 |
|---|---|
| `scenarios/` | 当前 Python case 的 14 份 YAML 定义，按能力分组；由 `scripts/commands/list_cases.py` 和 `scripts/commands/run_cases.py` 加载。旧扩展功能矩阵已从当前源码移除。 |
| `suites.yaml` | core、functional、workload 等 suite 的选例、覆盖及采集档位。 |
| `scale_cases/` | 独立的规模实验配置（线上、历史 A/B、lineage、监控复核等），不进入日常回归。 |
| `report_views/` | 报告视图定义及参数扫描示例。 |
| `perf_presets/` | 手写的预设登记与基础性能声明；标定所得性能文件存于 `data/performance/`。 |
| `load_client_env.txt` | Java load client 的环境变量清单。 |
| `mode_profiles.yaml` | 指向根目录同名表的链接，供模式配置统一管理。 |

用例结构及实例列表见 [scenarios/README.md](scenarios/README.md)。
