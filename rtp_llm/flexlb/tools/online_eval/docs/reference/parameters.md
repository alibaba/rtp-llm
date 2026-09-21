# 参数参考

本页集中解释三个开发机 runbook 共用的公开参数。更细的 case 数据以 `config/scenarios/*.yaml` 和命令 `--help` 为准。

## Master 形态

| 参数 | 含义 |
|---|---|
| `--profile` / `FLEXLB_PROFILE` | 完整 Master 配置名称 |
| `--master-mode` / `FLEXLB_MASTER_MODE` | `sb/sn/wb/wn` 简写；映射见 `config/mode_profiles.yaml` |
| `FLEXLB_CONFIG_OVERRIDE` | 对 profile 的受校验字段覆盖，格式为 `k=v,...` |
| `FLEXLB_CONFIG` | 完整配置逃生口；设置后必须保存并审查最终值 |

profile 与显式 master mode 不一致应直接失败，不能静默选一个。

## 拓扑与端口

| 参数 | 含义 |
|---|---|
| `N_PREFILL` / `N_DECODE` | 压测的 P/D 引擎数 |
| `MOCK_BASE_GRPC_PORT` | 压测 Mock gRPC 连续端口区间起点；控制口为它减一 |
| `FLEXLB_HTTP_ADDR` | 压测 Master HTTP 地址；management 另占一个端口 |
| `FLEXLB_FT_PARALLEL_MASTER_BASE` | 功能/场景 lane 0 的 Master 端口组起点 |
| `FLEXLB_FT_PARALLEL_MOCK_BASE` | 功能/场景 lane 0 的 Mock 端口窗口起点 |
| `--parallel` | 同时运行的 lane 数；功能测试可并行，性能和场景比较通常为 1 |
| `--mock-stride` | lane 间 Mock 端口跨度，必须覆盖每 lane 的最大窗口 |

## 负载

| 参数 | 含义 |
|---|---|
| `SEND_MODE=replay|uniform` | 按 trace 时间回放，或按目标速率合成 |
| `REPLAY_SPEED` | replay 时间缩放；不是直接 QPS |
| `SEND_MODE_QPS` | uniform 的目标发送速率 |
| `DURATION_S` | 发压时间 |
| `LOOP=1` | trace 结束后循环，直到时长结束 |
| `FLEXLB_WARMUP_SECONDS` | 从统计稳态窗口排除的预热时间 |
| `FETCH_OUTPUT_STREAM` | `1` 获取完整输出；`0` 为 schedule-only 形态 |
| `MAX_CONCURRENCY` | 客户端整请求并发上限，包括 Schedule 和结果消费 |

## 证据与输出

| 参数 | 含义 |
|---|---|
| `RUN_ROOT` / `RUN_ID` | 压测输出根和本次身份 |
| `--out-dir` / `--json` | 功能与场景的实例目录和汇总 JSON |
| `--archive` / `EXPERIMENT_ARCHIVE_PATH` | 可选的单文件实验包 |
| `COLLECTION_PROFILE` | `aggregate`、`request` 或 `diagnostic`；越靠后证据越多 |
| `PROMETHEUS_BIN` | Prometheus 可执行文件路径 |

## Mock 性能模型

性能 JSON 负责模拟 P/D 时间、KV 容量、并发和可选噪声。核心参数的实现说明见 [`flexlb-mock-engine/README.md`](../../../../flexlb-mock-engine/README.md)。修改性能模型后，报告必须记录文件内容或摘要；同一名称不保证内容相同。
