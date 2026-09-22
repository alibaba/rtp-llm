# 参数参考

这是三个开发机 runbook 的公开参数入口。命令均位于 `scripts/commands/`；以 `--help` 查看完整可执行词表。场景数据细节见 `config/scenarios/*.yaml`。

## 运行形态

| CLI 参数 | 含义 |
|---|---|
| `--master-mode sb\|sn\|wb\|wn` | 功能、场景、压测共用的 Master 形态缩写；映射在 `config/mode_profiles.yaml` |
| `--profile` | 完整配置名；和显式 mode 不一致时失败 |
| `--config-override k=v,...` | 压测专用的受校验配置覆盖；最终渲染写入 `master_config.json` |
| `--suite core\|functional\|workload\|all` | 功能/场景实例集合；`run_cases.py` 默认 core |
| `--parallel` | 功能/场景 lane 数；性能比较和故障场景通常为 1 |
| `--dry-run` | 不启动服务，显示资源或压测计划 |

## 压测拓扑与负载

| CLI 参数 | 含义和默认值 |
|---|---|
| `--n-prefill` / `--n-decode` | 逻辑引擎数，默认 12/40 |
| `--mock-base-grpc-port` | Mock 连续 gRPC 端口起点，默认 61000；控制口为起点减一；功能测试自动端口窗口不进入该 band |
| `--master-http-port` / `--master-management-port` | 默认 7001/7002；gRPC 另占 HTTP+2 |
| `--traffic-source-spec` | 注册流量源 JSON；省略时用 pinned 匿名 prefix DAG 模型；拒绝外部 `TRACE_FILE` |
| `--limit` | 物化计划的请求数上限，0 表示无上限；默认 1000 |
| `--send-mode replay\|uniform` | trace 默认 replay，synthetic 默认 uniform；合成源不得 replay |
| `--replay-speed` | replay 时间倍率，默认 10；不是直接 QPS |
| `--send-mode-qps` / `--ramp-up-s` | uniform 目标速率默认 650 QPS，以及从 0 上升的 30 秒 ramp |
| `--duration-s` / `--loop` | 客户端持续时间默认 120 秒；默认单次遍历，循环需显式开启 |
| `--warmup-s` / `--client-start-delay-s` | Master 启动后无流量预热默认 10 秒；所有 shard 共用客户端起始时间，默认再延迟 10 秒 |
| `--workers` / `--max-concurrency` | 客户端 shard 默认 8；整请求并发上限按 `ceil(total/workers)` 分给各 shard |
| `--fetch-output-stream 0\|1` | 默认 1，读完整输出；0 仅调度/引擎执行，不用于正式端到端 A/B |
| `--force-priority` | 默认 50；0 允许逐记录 priority |
| `--client-option KEY=VALUE` | 高级 JavaLoadClient 参数，可重复；键必须在 `config/load_client_env.txt`，且不能覆盖上述编排参数 |

## 证据与输出

| CLI 参数 | 含义 |
|---|---|
| `--run-root` / `--run-id` / `--run-dir` | 输出目录，默认 `run/<时间戳>`；`--run-dir` 直接指定完整路径 |
| `--collection-profile aggregate\|request\|diagnostic` | 默认 aggregate；diagnostic 才写事件与完整 pv.log |
| `--monitor-interval-s` | Prometheus 采样周期，默认 1 秒；需在 PATH 或 `PROMETHEUS_BIN` 提供可执行文件 |
| `--jfr-duration` | Master JFR 上限，默认 300s；文件为运行目录的 `flexlb_profile.jfr` |
| `--master-pv-log` | 显式保留逐请求 Master pv.log；默认关闭，即使选择 diagnostic 采集档 |
| `--archive` | 可选单文件实验包；失败运行保存为 incomplete |
| `--out-dir` / `--json` | 功能和场景实例目录及汇总 JSON |

## Mock 性能模型

`--performance` 读取 P/D 时间、KV 容量和可选噪声 JSON，默认 `data/performance/dsv4_flash_performance.fast_ab.json`。`--mock-heap`、`--master-heap`、`--client-heap` 设置 JVM heap；`--decode-max-concurrency`、`--prefill-cache-blocks`、`--decode-cache-blocks` 设置 mock 资源。性能模型说明见 [`flexlb-mock-engine/README.md`](../../../../flexlb-mock-engine/README.md)。修改模型后，报告必须记录文件内容或摘要；同名文件不保证内容相同。
