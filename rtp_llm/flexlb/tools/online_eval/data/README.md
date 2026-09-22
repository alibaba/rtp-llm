# 版本化测试输入

这里仅放可复现测试所需的输入；压测和实验输出写入忽略的 `run/` 或指定归档目录。

| 文件 | 用途 / 消费者 |
|---|---|
| `performance/dsv4_flash_performance.fast_ab.json` | 压测默认性能模型；Java 配置 schema 测试也覆盖此文件。 |
| `performance/dsv4_flash_performance.sm100_dev.json` | Java 配置 schema 回归输入。 |
| `traffic_models/frontend_20260921.xz` | 匿名 prefix DAG 模型；规模实验和压测从它在运行目录生成请求计划。 |
| `traffic_models/frontend_20260921.manifest.json` | 上述压缩模型的来源与校验信息，随模型一同保留。 |
| `traffic_models/master_batch_templates.json` | 从该模型前 128 个事件派生的 Java Master 回归样本，不含原始请求或 token；源 SHA 在测试时校验。运行 `python3 scripts/derive_master_templates.py` 可重建。 |

后续移除数据前，先确认其引用方、默认参数和可复现性是否已迁移。
