# 参数参考

功能、性能和故障场景通过 `scripts/commands/run_cases.py` 执行。命令参数用于实例选择、资源规划与产物位置；拓扑、流量、观察窗口和门槛以 `config/scenarios/*.yaml` 为准。

## 运行与选择

| CLI 参数 | 含义 |
|---|---|
| `--master-mode sb\|sn\|wb\|wn` | Master 形态缩写，映射在 `config/mode_profiles.yaml` |
| `--profile` | 完整 profile 名，和显式 mode 不一致时失败 |
| `--grade strict\|normal\|loose` | 配置门槛档位，传递给实际实例执行器 |
| `--suite core\|functional\|workload\|all` | CI suite 或实例性质筛选，默认值来自 `config/suites.yaml` |
| `--case-dir` | 场景 YAML 文件或目录 |
| `--instances` | 精确的编译实例 ID；可先用 `list_cases.py --list-json` 获取 |
| `--categories` | 按 category 选择实例 |
| `--parallel` | lane 数；性能比较与故障场景通常使用 1 |
| `--mock-stride` | 相邻 lane 的 Mock 端口间隔，须容纳声明的拓扑和预留端口 |
| `--dry-run` | 编译并展示实例、端口规划和报告视图，不启动服务 |

大型场景的 worker 端口容量通过 `FLEXLB_FT_WORKER_PORT_CAPACITY` 设置，编译与执行必须一致；见[编译与运行底座](build-and-runtime.md)。

## 证据与输出

| CLI 参数 | 含义 |
|---|---|
| `--out-dir` | 本次运行独占的实例与 lane 产物目录 |
| `--json` | 完整汇总 JSON 路径 |
| `--archive` | 可选的实验归档 ZIP |
| `--timing-json` | 按既有耗时计划 lane 分配，不改变 case 的预算或判定 |

采集档位、采样周期和缺采预算由 YAML 的 `test` 声明；指标定义由 `test.monitoring.query_plan` 选择。`reports` 选择报告视图，完整指标仍落在 `metrics.json`，见[指标配置](../../config/monitoring/README.md)和[运行产物](results.md#收取产物)。

## 性能模型与流量

`perf_preset` 选择 `config/performance_presets.json` 登记的引擎性能档案。偏离采集档案的测试参数通过 scenario 的 `model_override` 显式声明原因。流量源固定文件与 SHA，播放方式、速率和并发由 case 参数声明；调整拓扑或流量后不得直接沿用未重新校准的 band。

性能门禁先校验请求终态、采样覆盖和窗口有效性，再解释吞吐及延迟。模型说明见 [`flexlb-mock-engine/README.md`](../../../../flexlb-mock-engine/README.md)，播放规则见[播放与复现](playback-controls.md)。
