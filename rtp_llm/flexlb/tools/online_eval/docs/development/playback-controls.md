# 播放与离线复现

语义与边界以 [流量架构](../architecture/traffic.md#播放层) 为准。本页只说明配置与核对方式，命令从 `online_eval` 执行。

## 配置

```yaml
client:
  playback:
    mode: uniform
    qps: 240
    duration_s: 600
    ramp_up_seconds: 10
    max_laps: 3
    identity: structural-relabel
  limits:
    max_concurrency: 8192
    request_timeout_s: 120
    drain_timeout_s: 150
  transport:
    channels: 8
    event_loop_threads: 8
```

时戳播放改为 `mode: true-ts` 与 `speed`，不填写 QPS 或升压时间。`max_laps: 0` 必须同时有正的墙钟时长。要刻意保留跨圈缓存，填写 `identity: none`。

场景只接受 `playback`、`limits`、`transport` 三层配置。字段使用数字，不填写 Java 环境变量或字符串数字。框架在启动前校验这些参数，并在 Java 启动后核对快照。配置包含不支持的控制字段时失败，不静默忽略。

`playback` 声明实验节奏、时长与身份；`limits` 声明并发上限、单请求超时及停止发送后的收尾等待；`transport` 声明连接与事件线程资源。`duration_s` 是发压上限，输入播放完或轮数耗尽可以提前结束。场景必须显式声明正的时长，uniform 必须显式声明正的 QPS。

Python 的类型对象与 Java 环境解析共用 Mock resources 中的 `load-client-defaults.properties`。时间戳回放缺省速度为 1；缺省轮数为 1。Java 单独运行时 `DURATION_S=0` 表示单轮不设墙钟截止，场景入口拒绝这种无界运行配置。空环境变量按缺省值处理。

框架固定读取完整响应流，并禁用旧的首块加盐；跨轮身份由 `playback.identity` 唯一控制。端点、请求 priority、采集策略和 flow 标识由输入与运行时注入，不属于 case 的客户端配置。

`request_timeout_s` 是请求/RPC 超时；`drain_timeout_s` 是停止发压后所有未完成请求的共同等待窗口，必须覆盖请求超时。case 的 drain 阶段预算需额外留出退出和证据落盘时间。Java 环境变量仅作为启动协议保存在运行证据里，不是场景配置入口。

## 核对与复现

物化请求的 manifest 记录输入 SHA、过滤/采样规则和数量；flow 目录中的 `playback.json` 记录 Java 实际使用的发送策略。两者随实验证据保存。

多圈计划可用 Mock jar 重建：

```bash
java -cp <mock-engine-all.jar> org.flexlb.mockengine.PlaybackPlan \
  <playback.json> <requests.jsonl> <reconstructed-plan.jsonl>
```

输入 SHA 不匹配时拒绝重建。输出包含请求 ID、圈号、理想发送时间、输入/输出长度与实际块 key；实际发送的延误、停止位置和结果仍以客户端事件为准。
