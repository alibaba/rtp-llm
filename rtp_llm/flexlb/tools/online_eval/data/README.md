# 版本化测试输入

这里仅放可复现测试所需的输入；压测和实验输出写入忽略的 `run/` 或指定归档目录。

| 文件 | 用途 / 消费者 |
|---|---|
| `online_logs/sample_access.json` | Java `MasterBatchEndToEndPerformanceTest` 的访问日志样本。 |
| `online_logs/trace_30min.jsonl` | 压测默认重放 trace；`scripts/stress/run_online_eval.sh` 与 Python harness 使用。 |
| `performance/dsv4_flash_performance.fast_ab.json` | 压测默认性能模型；Java 配置 schema 测试也覆盖此文件。 |
| `performance/dsv4_flash_performance.sm100_dev.json` | Java 配置 schema 回归输入。 |
| `traffic_models/frontend_20260921.xz` | 三份 `config/scale_cases/` 规模实验的前端流量模型。 |
| `traffic_models/frontend_20260921.manifest.json` | 上述压缩模型的来源与校验信息，随模型一同保留。 |

后续移除数据前，先确认其引用方、默认参数和可复现性是否已迁移。
