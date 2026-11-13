# 启动实验 View

在 `rtp_llm/flexlb/mock-framework/` 目录执行。View 用于回看已结束的 Case。

## 1. 准备结果和依赖

- `RESULT_DIR` 指向完整的 Case 结果目录，即包含 `result.json` 的目录。要看指标曲线，保留其中的 `view.json` 和 `prometheus/`。
- `PROMETHEUS_BIN` 指向 Prometheus 可执行文件，`GRAFANA_HOME` 指向包含 `bin/grafana`、`conf/` 和 `public/` 的 Grafana 安装目录。
- 初次使用时按 [框架使用说明](../README.md#准备环境) 安装 Python 依赖。

## 2. 启动

```sh
RESULT_DIR=/path/to/result
VIEW_DIR="$(mktemp -d /tmp/mock-view.XXXXXX)/session"

uv run --locked mock-view "$RESULT_DIR" \
  --output "$VIEW_DIR" \
  --prometheus "$PROMETHEUS_BIN" \
  --grafana-home "$GRAFANA_HOME"
```

`--output` 必须是结果目录之外、尚未创建的路径。需要固定页面端口时加 `--port <端口>`。

## 3. 打开页面

打开终端打印的本机 URL。先看 Case 结果和门禁，再等指标服务就绪后查看曲线。点击门禁卡片可查看完整的实际值和预期值，Grafana 在当前时间区间内只显示该步骤的事件线。通过「事件线」可以重新多选或全部显示。切换地址、事件线和时间区间会保留大盘中已展开的行。

曲线加载失败时，先看页面给出的原因，再检查结果里的 `view.json`、`prometheus/` 和查看目录里的指标进程日志。修复后换一个新的 `--output` 重启。看完在启动终端按 Ctrl+C 停止。

大盘分区、指标口径和筛选方式见 [默认大盘说明](../views/README.md)。
