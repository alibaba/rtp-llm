# 配置目录

配置是测试输入，不是运行结果。根目录历史兼容文件 `mode_profiles.yaml` 与这里的同名软链接指向同一份模式表。

| 位置 | 内容与用途 |
|---|---|
| `scenarios/` | 按 case 名扁平存放 YAML；实例性质、说明和采集档位由文件内的 `test` 声明，variant 可覆盖。 |
| `suites.yaml` | CI 必跑 `case::variant` 清单及默认 suite；不定义实例性质或监控参数。 |
| `report_views/` | 公共默认报告模板及 case 专属报告引用。 |
| `performance_presets.json` | 性能预设登记表，指向 `data/performance/` 的采集档案；只登记选择关系与不属于采集档案的运行参数。 |
| `load_client_env.txt` | Java load client 的环境变量清单。 |
| `mode_profiles.yaml` | 指向根目录同名表的链接，供模式配置统一管理。 |

用例结构及实例列表见 [scenarios/README.md](scenarios/README.md)。

模型/硬件档中照抄自部署的时延、容量、规模与块口径只在一份采集档案定义；preset 和 scenario 引用它。新档必须有可核验的部署身份、采集窗口、完整性信息；`legacy_unverified` 仅供兼容既有输入，不能作为新权威。测试专用偏离须在 scenario 的 `model_override` 声明基线与原因，不能用平行真值覆盖档案。各类块数的具体口径见 [数据规则](../data/README.md)。
