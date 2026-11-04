# Mock 框架使用说明

用已有 Case 启动真实 FlexLB Master、Mock RTP Prefill/Decode、可选 ViT 和 Client，执行流量与故障步骤，保存门禁、请求证据和 Prometheus 数据，再用 View 查看结果。Master 使用生产调度和 ZK 选举；Mock Engine 使用生产协议，在 CPU 上模拟计算与缓存行为。

当前支持 RTP、Engine 扩缩容、Master 停启与恢复、多 Client 播放及主备发现。vLLM 尚未接通。小流量 Case 用于功能验证，不能作为真实 GPU 性能或规模容量结论。

本分支 `codex/mock-framework-dsv41` 基于公共 Mock 提交 `84b174c6d9`，真实 Java Master 和 Engine RPC 契约固定为 `feature/dsv41-dev-main-202161013@ba26ddf1dd86c02bcee3cc6eee966ca59e4ae7a6`。不随远端分支自动更新，也不包含真实 GPU 引擎实现。公共修改先进入 `codex/mock-framework-on-main`，再按独立提交同步；本分支的协议迁移不反向覆盖 main 的契约。

多模态回放使用原始 token 和图片占位位置 `Image.offset`，不再接受预展开的 `Image.length`。Client 使用显式配置的 image_token_id，将占位位置编码为该标记，V 给出长度/hash 元数据，P 准备计算输入并通过原生 PD 请求传递缓存身份。V4.1 几何、ViT 负载及 KVCM 接入范围见[组件参考](docs/component-reference.md)。

## 准备环境

以下命令均在本目录 `rtp_llm/flexlb/mock-framework/` 执行。若使用外层内部仓库，先进入 `github-opensource/` 中的对应目录。

| 依赖 | 要求 |
| --- | --- |
| Java | Java 21，设置 `JAVA_HOME`。构建使用仓库自带 Maven Wrapper。 |
| Python / uv | Python 3.11 或更高版本，安装 uv；Python 依赖由 `uv.lock` 锁定。 |
| Prometheus | 已安装的可执行文件，`PROMETHEUS_BIN` 指向程序本身，Runner 不负责下载。 |
| Grafana | 仅查看指标曲线时需要；`GRAFANA_HOME` 包含 `bin/grafana`、`conf` 和 `public`。 |
| 执行资源 | 端口、回环地址、磁盘、CPU 和内存满足所选 Case。双 Master 需要能绑定两个不同的回环 IP。 |

在实际执行机器上确认工具可运行，按锁定版本准备依赖，检查构建和测试所需配置、资源及 Case 输入完整可读。下载失败先解决依赖源或网络问题；不通过改版本或跳过检查绕过。共享机器上使用独立目录，不抢占其他服务。

```sh
export JAVA_HOME="<Java 21 安装目录>"
export PROMETHEUS_BIN="<Prometheus 可执行文件>"
export PATH="$JAVA_HOME/bin:$PATH"

uv sync --locked
./scripts/build-java.sh
```

`build-java.sh` 编译 Java 并运行 Java 测试，复用生产模块的公式、协议和 Master；构建 Mock 时跳过 Boot 重打包及 Whale 分发归档，不改变生产模块独立构建的默认行为。

## 跑第一个 Case

`cases/basic/case.yaml` 使用一个真实 Master、RTP P/D 和一个 Client。它以总 QPS=2、总并发=2 播放 20 条请求，检查唯一主节点、中段 QPS（1.5–2.5）及全部请求成功。

先编译 Case，再生成本机启动配方，最后执行同一份 Case：

```sh
CASE_FILE=cases/basic/case.yaml
RUN_ROOT="$(mktemp -d /tmp/mock-basic.XXXXXX)"

uv run --locked mock-case compile "$CASE_FILE" \
  --output "$RUN_ROOT/compiled.json"

uv run --locked mock-case launch "$CASE_FILE" \
  --java "$JAVA_HOME/bin/java" \
  --prometheus "$PROMETHEUS_BIN" \
  --output "$RUN_ROOT/launch.yaml"

uv run --locked mock-case run "$CASE_FILE" \
  --launch "$RUN_ROOT/launch.yaml" \
  --output "$RUN_ROOT/result"
```

`compile` 检查 Case 结构与引用，不启动服务，也不验证输入数据内容。`launch` 检查本地构建产物与配方，不启动组件或下载依赖。配方包含本机绝对路径，供本次运行使用，不提交到仓库。

Case 保存业务配置、流量和门禁；启动配方保存命令、程序路径与 JVM 参数。需要调整 JVM 参数时编辑生成的 `launch.yaml`。默认 `-XX:ActiveProcessorCount=2` 只限制 JVM 看到的处理器数，不保证 CPU 配额。

P/D/V、Master 和 Client 各实例直接启动独立 JVM。ZooKeeper 由单独的角色进程提供。

每次运行使用新的配方文件和结果目录。Runner 负责启动、执行和清理；异常或门禁失败会中止后续步骤并收尾。Ctrl+C 会取消执行并清理本次资源。环境问题可修复后重新准备；业务失败先保存证据，不为通过门禁改参数重跑。

## 检查结果与失败请求

| 产物 | 查看内容 |
| --- | --- |
| `result.json` | 最终状态、起止时间及清理错误。状态为 PASS / FAIL / ERROR / CANCELLED。 |
| `gates.jsonl` 和事件记录 | 实际执行的步骤、门禁预期与实际值、查询时刻和窗口；未执行的检查不算通过。 |
| 配置与源码记录 | 原始 Case、编译配置、启动配方、接口清单、源码版本和工作区差异。 |
| `components/` | 组件生效配置、端点、日志和 Client 播放结果/请求证据。 |
| `prometheus/` | 采集配置、日志、完整性记录及 TSDB/WAL；缺指标不能补零。 |
| `view.json` | 本次保存的大盘快照，供之后回看。 |

检查 Client 的 issued、finished、success/error/timeout/cancelled，以及 queued、inflight、unsent。Case 主动提前停止播放时，未发送的输入不算失败；已发送请求是否收齐终态、请求证据是否完整，要分别确认。清理错误和采集错误也应单独报告，不能解释成请求失败或忽略后宣称 PASS。

新运行的逐请求证据位于 `components/<client>/playback/<轮次>/requests.jsonl`，可直接查看。逐请求查询使用 Case 的 `client/requests` 接口，支持 Client、请求 ID、播放轮次、时间和结果筛选。显式查询会读取并校验 JSONL；`client/stop` 使用 Java 返回的最终汇总，不重复扫描文件。协议和示例见 [Client 逐请求记录](docs/architecture.md#client-逐请求记录与查询)。

## 查看报告

在展示机器上准备 Grafana 和 Prometheus，使用完整结果目录启动 View：

```sh
export GRAFANA_HOME="<Grafana 安装目录>"

uv run --locked mock-view "$RUN_ROOT/result" \
  --output "$RUN_ROOT/view-session" \
  --prometheus "$PROMETHEUS_BIN" \
  --grafana-home "$GRAFANA_HOME"
```

打开命令打印的 URL，检查实验时间、配置、时间线、门禁及指标服务状态。View 输出目录必须是新的，且位于实验结果目录之外；需要固定端口时可加 `--port <端口>`。

当前 Case 省略 `view` 时使用框架的 `views/default.json`，也可以显式指定原生 Grafana JSON。运行时保存 `view.json` 快照；旧结果缺少快照时不会自动补写。远端结果转到展示机器时需保留完整目录结构，尤其是 Prometheus 数据及请求证据。

View 只读已完成实验，复制 TSDB 后启动独立指标服务，不重跑流量或门禁。页面打开后仍需确认指标服务就绪；指标不可用时，配置、门禁和时间线仍可读。缺数据不改变实验状态。看完用 Ctrl+C 或 SIGTERM 停止 View；关闭浏览器不会停止服务。

事件线筛选、指标异常与进程清理见 [View 指南](docs/view-guide.md)，大盘分区和指标覆盖见 [大盘说明](views/README.md)。

## 运行其他 Case

将上述 `CASE_FILE` 换成目标 Case 路径，并为新实验使用新的 `RUN_ROOT`。例如 `cases/master_ha_failover/case.yaml` 持续播放输入，依次故障与恢复两个 Master，检查接管和近期请求成功率。先阅读 Case，确认规模、等待时间和资源需求；不要直接把 basic 的资源预期套到 HA。

多 Master 使用不同的回环 IP、相同服务端口。Linux 默认支持 127/8；macOS 未配置所需回环别名时无法运行双 Master，不会自动修改系统网络。每次实验独占 ZK 和选举路径。

Case 字段、相对路径、共享资源、引用、断言、扩缩容与 Master 停启用法见 [Case 指南](docs/case-guide.md)。

## 开发检查与详细文档

执行 Java 和 Python 检查：

```sh
PROMETHEUS_BIN="$PROMETHEUS_BIN" ./scripts/test-framework.sh
```

`test-framework.sh` 会先调用 `build-java.sh`，再运行 Python 测试。缺少 Prometheus 时会跳过真实采集集成测试；macOS 缺少回环别名时跳过双 Master 测试。报告检查结果时说明跳过项，测试通过不代替完整 Case 验收。

## CI Case

CI 构建一次 Java 运行产物，再由各 worker 解包并独立运行选中的 Case。构建和运行需要 Java 21，运行还需要预装的 Prometheus 和 uv 依赖缓存。

```sh
./scripts/build-ci-case.sh /absolute/path/to/mock-case-runtime.tar.gz
./scripts/run-ci-case.sh basic /absolute/path/to/mock-case-runtime.tar.gz /absolute/path/to/new-output
```

归档包含 Engine、Client、Master 和 ZooKeeper 运行产物。Case 脚本在目标 worker 重建 classpath，按锁文件离线安装 Python 环境，并保存编译结果、启动配方和完整的运行结果。收到 SIGTERM 时，Runner 会取消 Case 并回收子进程。

在源码仓库根目录构建预制环境：

```sh
docker build -f rtp_llm/flexlb/mock-framework/docker/ci-case-env.Dockerfile -t mock-case-ci:local .
```

镜像预热 Maven 和 uv 缓存。运行产物由当次 CI 构建生成。

- [Case 指南](docs/case-guide.md)：配置、路径、Step/ref、断言、启动配方与生命周期操作。
- [组件配置与指标参考](docs/component-reference.md)：Master、RTP、ViT、KV、Compute/EOS、Client 协议及指标口径。
- [架构与设计边界](docs/architecture.md)：组件职责、Scheduler/RtpEngineLoop、协议和资源生命周期、接口扩展。
- [View 指南](docs/view-guide.md)：报告内容、事件标注、异常展示与服务生命周期。
- [大盘说明](views/README.md)与[指标覆盖](views/coverage.md)：查询分区、标签与缺口。

本机启动配方临时为 Master 设置 `-Dflexlb.monitor.metric-whitelist=flexlb_`，导出全部 FlexLB 业务指标，生产默认值保持不变。全量采集在规模压测下的序列数、采样耗时和资源开销仍需验证。

Whale 单实例服务的镜像构建、ENV 启动和平台接线边界见 [Whale Engine 镜像与启动](docs/whale-engine.md)。
