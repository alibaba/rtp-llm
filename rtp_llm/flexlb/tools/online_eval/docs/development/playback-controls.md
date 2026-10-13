# 播放与离线复现

语义与边界以 [流量架构](../architecture/traffic.md#播放层) 为准。本页只说明配置与核对方式，命令从 `online_eval` 执行。

## 配置

```yaml
client:
  DURATION_S: '600'
  MAX_CONCURRENCY: '8192'
  FETCH_OUTPUT_STREAM: 'true'
  REPLAY_UNIQUE_PREFIX: 'false'
  playback:
    mode: uniform
    qps: 240
    ramp_up_seconds: 10
    max_laps: 3
    identity: structural-relabel
```

时戳播放改为 `mode: true-ts` 与 `speed`，不填写 QPS 或升压时间。`max_laps: 0` 必须同时有正的墙钟时长。要刻意保留跨圈缓存，填写 `identity: none`。

场景使用 playback 块，不同时填写裸 `SEND_MODE`、`LOOP` 或 `MAX_LAPS`。框架在启动前校验这些参数，并在 Java 启动后核对快照。配置包含不支持的控制字段时失败，不静默忽略。

## 核对与复现

物化请求的 manifest 记录输入 SHA、过滤/采样规则和数量；flow 目录中的 `playback.json` 记录 Java 实际使用的发送策略。两者随实验证据保存。

多圈计划可用 Mock jar 重建：

```bash
java -cp <mock-engine-all.jar> org.flexlb.mockengine.PlaybackPlan \
  <playback.json> <requests.jsonl> <reconstructed-plan.jsonl>
```

输入 SHA 不匹配时拒绝重建。输出包含请求 ID、圈号、理想发送时间、输入/输出长度与实际块 key；实际发送的延误、停止位置和结果仍以客户端事件为准。
