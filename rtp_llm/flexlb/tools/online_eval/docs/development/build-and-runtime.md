# 开发机编译与运行底座

压测、功能测试和场景测试共用 Java 21 的 FlexLB Master、Java Mock Engine 和同一套 Python 编排与报告代码。三个 runbook 不再重复本页内容。

所有命令从 `rtp_llm/flexlb` 执行。`$OUT` 应指向本次测试独占的新目录。

## 前置条件

- JDK 21；`java -version` 与 Maven 实际使用的 JVM 都必须是 21。
- Python 3，且安装 `PyYAML`、`grpcio`、`grpcio-tools`、`protobuf`。
- Prometheus 可执行文件。场景和压测通过 `PROMETHEUS_BIN` 指向它；命令名为 `prometheus` 时可省略。
- 至少一个连续的空闲端口区间。功能/场景 runner 会规划端口；压测使用 `MOCK_BASE_GRPC_PORT`、Master HTTP 和 management 端口。
- 标准 12P/40D 压测建议预留约 60 GiB 内存；较小机器应降低 P/D 数量和 JVM heap。

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

功能与场景 runner 为每个实例启动并回收 Master 和 Mock Engine；压测入口启动 Mock cluster、Master、load client 和采集器。不要在它们之前手工启动同一套进程。

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

继续阅读对应 runbook：[压测](stress.md)、[功能测试](functional.md)、[场景测试](scenario.md)。
