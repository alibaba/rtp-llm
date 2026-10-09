# 开发机编译与运行底座

功能、性能和故障场景共用 Java 21 的 FlexLB Master、Java Mock Engine 和同一套 Python 编排与报告代码。各类测试的运行前置以本页为准。

所有命令从 `rtp_llm/flexlb` 执行。`$OUT` 应指向本次测试独占的新目录。

## 前置条件

- JDK 21；`java -version` 与 Maven 实际使用的 JVM 都必须是 21。
- Python 3，且安装 `PyYAML`、`grpcio`、`grpcio-tools`、`protobuf`。
- Prometheus 可执行文件。workload 通过 `PROMETHEUS_BIN` 指向它；命令名为 `prometheus` 时可省略。
- 至少一个连续的空闲端口区间。runner 规划 Mock、Master HTTP、management 和 gRPC 端口。
- 按 YAML 的 P/D 拓扑和 JVM heap 预留内存；改变规模时需同步检查性能门槛。

## 编译

```bash
./mvnw -P'opensource,!internal' \
  -pl flexlb-api,flexlb-mock-engine -am package -DskipTests
```

必须同时存在：

```text
flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar
flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar
```

`-am` 不能省略，它会构建所需 reactor 模块。相邻目录存在内部源码时仍显式使用 `opensource,!internal`，防止 Maven 自动激活另一套依赖。

## 启动模型

runner 为每个实例启动并回收 Master、Mock Engine、load client 和所需采集器。不要在 runner 之前手工启动同一套进程。

启动参数分三层：

1. **运行形态**：`sb/sn/wb/wn`，定义 SINGLE/FIXED_WINDOW 与 BATCH/NON_BATCH 的组合。
2. **拓扑与资源**：P/D 数量、端口基址、JVM heap、KV block 容量。
3. **负载与证据**：请求数量或时长、replay/uniform、采集档位、输出目录。

四种形态由 `config/mode_profiles.yaml` 解释：

| 简写 | decision | dispatcher | profile |
|---|---|---|---|
| `sb` | SINGLE | BATCH | `single-batch` |
| `sn` | SINGLE | NON_BATCH | `single-nonbatch` |
| `wb` | FIXED_WINDOW | BATCH | `batch-window` |
| `wn` | FIXED_WINDOW | NON_BATCH | `window-nonbatch` |

测试需要覆盖多种形态时显式分别运行。不要把某一形态通过解释成全部形态通过。

大型场景的 worker 数超过默认端口容量时，编译清单与执行必须使用同一个 `FLEXLB_FT_WORKER_PORT_CAPACITY`。例如完整场景矩阵可使用 `2048`，执行时配合 `--mock-stride 2100 --parallel 1`，并预留相应端口范围；这只扩大端口规划，不改变 YAML 中的拓扑或门槛。

```bash
export FLEXLB_FT_WORKER_PORT_CAPACITY=2048
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios --suite all --list-json
```

`all` 包括功能与持续负载场景。依次选择 `single-nonbatch`、`single-batch`、`window-nonbatch`、`batch-window` 执行，每种形态使用独立输出目录。

## 运行前检查

```bash
test -f flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar
test -f flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar
test -x "${PROMETHEUS_BIN:-$(command -v prometheus)}"
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios --list-json >/tmp/flexlb-case-catalog.json
```

最后一条只编译 case 计划，不启动服务。若这里失败，应先修复配置或 Python 依赖，再运行测试。

## 常见错误

- `UnsupportedClassVersionError`：运行 JVM 不是 Java 21。
- Maven 找不到内部依赖：缺少 `-P'opensource,!internal'`，或只构建单模块而未带 `-am`。
- `Prometheus required`：设置 `PROMETHEUS_BIN` 为真实可执行文件。
- 端口占用：为本次运行换一组端口基址；不要杀死来源不明的进程。
- 只有 `dry-run` 或 `list-json` 产物：这只是计划验证，不代表服务启动或测试通过。

继续按[运行测试](running.md)选择和执行实例。

## 替换 Master 制品

版本对照使用独立进程和输出目录，固定相同的 Mock、负载及采集条件。以下环境变量在启动进程前设置，身份归档由 `runtime/master_artifact.py` 生成：

| 环境变量 | 契约 |
|---|---|
| `FLEXLB_FT_MASTER_JAR` | 目标 jar 的绝对路径，导入时读取 |
| `FLEXLB_FT_MASTER_CONFIG_FILE` | JSON 整体替换 Master 配置，不做合并；Mock 仍使用测试配置 |
| `FLEXLB_FT_MASTER_SOURCE_COMMIT` | 声明的完整源码 SHA；不代替 jar 的实际摘要 |

外部配置按目标版本的源码或部署原文编写，由目标 jar 的解析器验证，不只替换 `schemaVersion`。发现协议、字段消费、单位与补齐默认值都须核对；不支持的组合停止测试。schema 1 的动态发现需要显式 `discovery_file` 及编入目标 jar 的 [WhaleFileDiscovery](../../../whale_mock/discovery_adapter/WhaleFileDiscovery.java)，Python 的环境桥接不会自动改造 jar。

`actual-master-config.json` 是传入原文，不能当解析后回读。先验证目标制品能发现 worker 并完成完整请求，再开展版本对照。制品、适配代码和配置差异随运行归档；解释规则见[结果与指标](results.md)。
