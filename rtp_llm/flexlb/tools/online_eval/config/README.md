# 配置目录

配置是测试输入，不是运行结果。根目录历史兼容文件 `mode_profiles.yaml` 与这里的同名软链接指向同一份模式表。

| 位置 | 内容与用途 |
|---|---|
| `scenarios/` | 按 case 名扁平存放 YAML；实例性质、说明和采集档位由文件内的 `test` 声明，variant 可覆盖。 |
| `experiments/` | 依赖外部采集数据的显式实验，用 `--case-dir` 指定文件运行；不进入默认场景枚举，执行前必须提供配置中固定 SHA256 的输入。 |
| `suites.yaml` | CI 必跑 `case::variant` 清单及默认 suite；不定义实例性质或监控参数。 |
| `report_views/` | 报告视图定义及参数扫描示例。 |
| `performance_presets.json` | 性能预设登记表，指向 `data/performance/` 中的 JSON，附带必要的 mock 启动参数。 |
| `load_client_env.txt` | Java load client 的环境变量清单。 |
| `mode_profiles.yaml` | 指向根目录同名表的链接，供模式配置统一管理。 |

用例结构及实例列表见 [scenarios/README.md](scenarios/README.md)。
