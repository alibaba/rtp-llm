# Master 组批实验归档

本目录保存 [A/B 合图](ab/report.html)、[A 单轮](a/report.html)、[B 单轮](b/report.html) 的离线 HTML 与对应 `analysis.json`，以及 [Whale zone 核对](zone-config-audit.md)。[manifest.json](manifest.json) 固定文件 SHA、源码与模型身份。`fit/` 中是捕获流量的拟合诊断，不是播放输入；播放输入是仓库 `data/traffic_trace/deepseek_v4_flash_20260923_0920_16m.xz` 与同名 manifest。

两轮来自同一源码提交 `280c22f9b8`、同一 Master/Mock JAR 和同一逐请求 trace。A 的组批决策是 `32/10/550`，B 是 `64/350/350`；比对器确认其他控制变量 `ALIGNED`。180 秒测量窗发送约 1536 QPS，两边请求成功率都是 100%。A 的 Prefill with-cache TPS/P 为 94,234，未达到绝对门槛 100,000；B 为 106,354，内层绝对检查 12/12 通过。

外层场景证据检查发现 Master 监控抓取空点：A 4 个，B 1 个，故两轮 `runtime_validity=INVALID`，B 的外层状态是 `ERROR`。这份归档可用于观察组批差异，不代表完整场景或生产门禁通过。报告中保留了运行时旧文件名 `local_flash_20260923_v3` 的原始路径；入库模型已按来源重命名，压缩内容 SHA 未变。请求级 journal 保留在远端 run `master-batch-ab-20260923`，不随本归档复制。
