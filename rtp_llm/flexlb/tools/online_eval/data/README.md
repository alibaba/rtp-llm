# 版本化测试输入

这里仅放可复现测试所需的输入、其来源元数据和派生标定件；人工编写的用例、预设登记与视图声明放在 `config/`。压测和实验输出写入忽略的 `run/` 或指定归档目录。

`catalog.json` 是固定模型、manifest、Java 夹具与校准 profile 的路径清单。Java 与 Python 都从它定位伴生文件；`tests/test_data_catalog.py` 校验模型字节、SHA、夹具来源和 profile provenance。`config/perf_presets/index.json` 登记可选性能预设，标定所得参数包放在 `performance/`。

| 文件 | 用途 / 消费者 |
|---|---|
| `performance/dsv4_flash_performance.fast_ab.json` | 压测默认性能模型；Java 配置 schema 测试也覆盖此文件。 |
| `performance/dsv4_flash_performance.sm100_dev.json` | Java 配置 schema 回归输入。 |
| `performance/production_scale_20260920.json` | 规模实验标定性能包，入口由 `config/perf_presets/index.json` 登记。 |
| `calibration/prefix_lineage_v2_ac2f8aad.profile.json` | 从固定实测模型导出的统计生成画像；逻辑 profile 名保持 `frontend_20260921`。 |
| `traffic_models/prefix_lineage_v2_ac2f8aad.xz` | 匿名 prefix DAG 模型；规模实验和压测从它在运行目录生成请求计划。 |
| `traffic_models/prefix_lineage_v2_ac2f8aad.manifest.json` | 上述压缩模型的来源与校验信息，随模型一同保留。 |
| `traffic_models/prefix_lineage_v2_ac2f8aad.templates.json` | 从该模型前 128 个事件派生的 Java Master 回归样本，不含原始请求或 token；源 SHA 在测试时校验。运行 `python3 scripts/pipeline/derive_master_templates.py` 可重建。 |

后续移除数据前，先确认其引用方、默认参数和可复现性是否已迁移。

新采集快照命名为 `<codec>_v<codec版本>_<内容SHA前缀>.xz`；同前缀的 `.manifest.json` 保存清单格式版本、codec 身份、采集来源、时间窗和完整 SHA。日期、来源与时段不写入文件名，以 manifest 为准。由快照派生的 profile 与夹具也使用相同前缀并以不同后缀区分。`frontend_20260921` 是迁移前命名，逻辑 profile 符号与文件路径属于不同命名空间。

新 `.xz` 默认被 `traffic_models/.gitignore` 排除。只有同时登记于 `catalog.json`、具有真实 case 消费者并通过 `tests/test_data_catalog.py` 的字节及伴生文件检查后，才能加入例外白名单并提交。未入库的不可重采样本应保存于仓库外归档，不以删除代替筛选。删除已入库快照前，先清理消费者及登记；反向检查会拒绝仍被跟踪却未登记的文件。
