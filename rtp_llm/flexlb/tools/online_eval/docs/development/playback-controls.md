# 播放层调节与复现

在已有匿名采集上研究到达节奏、输出长度或缓存冷暖变化时，配置转换参数与播放参数生成实验负载，无需重新采集。快照 `.xz`、SHA、manifest 保持原样；`path + sha256 + count` 仍然锁定输入。这里不规定门禁阈值，也不把新负载加入默认 CI。

这些旋钮用于构造受控实验，不意味着 mock 已与生产对齐。输出长度来自显式 trace `ol`，因此绕过 mock 引擎 EOS；不要把它描述为线上输出长度拟合。轮转保留率也不是缓存命中率，实际命中仍受前缀、容量、驱逐和路由影响。

## 到达曲线与随机到达

```yaml
client:
  DURATION_S: '60'
  REPLAY_UNIQUE_PREFIX: 'false'
  playback:
    mode: uniform
    qps: 240
    max_laps: 0
    seed: 42
    arrival: poisson
    rate_curve: [[0, 0.5], [10, 1], [30, 2], [50, 0.5]]
```

`rate_curve` 是 `[elapsed_seconds, multiplier]` 点表，在相邻点之间线性插值，末点后保持常数。首点必须为零秒，时间严格递增，范围 `[0, 1e9]`；倍率在 `[1e-6, 1e6]`，共 1–4096 个点。它支持上升、下降、三角和多段平台等形状，不接受可执行代码或回调。

uniform 的瞬时目标率为 `qps × multiplier(t)`。`qps` 是基准速率，整个窗口的均值为 `qps × integral(multiplier) / duration`，不是自动归一化后的均值。未指定 `arrival` 时沿用确定性到达；`arrival: poisson` 在累计强度轴上生成指数间隔，再反求发送时间。首请求有随机等待，有限窗口请求数自然波动；给定 seed 后计划固定。到达随机流使用独立 domain salt（`0xd1b54a32d192ed03`），避免与同 seed 的输出长度抽样形成确定的关联。

Poisson 必须用 uniform；`rate_curve` 或 Poisson 不能与burst、diurnal、ramp、GRADIENT 同用，组合会报错。这些模式使用解析式或数值积分。Poisson 到达必须显式提供 signed int64 `seed`，不能由运行名、墙钟或 shard ID 派生。

真实时间戳的时变变换：

```yaml
playback:
  mode: true-ts
  speed: 1
  max_laps: 1
  rate_curve: [[0, 0.5], [30, 2], [60, 1]]
```

设 `F(t)` 为倍率曲线积分，原时间偏移为 `s`，计划发送时间为 `F^-1(s / speed)`。这会压缩或拉伸局部时间轴，保留相同时间戳的重合关系和原始次序；它不保证 replay 的绝对 QPS 等于倍率值。replay 不支持 Poisson，因为那会替换原始到达过程。

调节的是计划时间，实际发送仍受 CPU、并发上限、背压和墙钟截止约束。观察 `send_due_epoch_ms`、`send_start_epoch_ms` 与 `pacing_lag_ms` 区分计划和执行；不能把生成器的积分精度当作线上实际发送保证。

## lineage 内容转换：输出分布与输入过滤

v2、v3 都支持以下 `source.parameters` 扩展，模型锁定字段不变：

```yaml
output_tokens: 8192
output_distribution:
  kind: discrete
  values: [64, 256, 1024, 4096]
  weights: [1, 3, 4, 2]
  seed: 42
```

无 `output_distribution` 时 `output_tokens` 仍为固定输出长度；有分布时它是 cap。discrete 的值必须为 `[1, cap]` 内整数，权重必须有限、非负，长度一致且总和有限并大于零；值超过 cap 会报错，不静默截断。零权重项不被采样。

原 geometric 形式不变：`{kind: geometric, mean_tokens: 400, seed: 42}`，均值参数至少为 1，样本被截到 cap，所以截断后的均值可能更小。geometric 与 discrete 的字段不能混用；二者都要求显式 signed int64 seed。

使用 SplitMix64 原始事件索引采样，过滤和 `max_requests` 不重新编号；改变 run namespace 不改变 `ol`。v2/v3 的 `max_input_tokens` 过滤仍展开被排除父请求，保留后代前缀身份。`realistic` 原有离散分布语法与随机流不变。

## 逐轮冷暖变化

```yaml
playback:
  mode: uniform
  qps: 240
  max_laps: 6
  identity: partial
  seed: 42
  retain_schedule: {kind: linear, start: 0, end: 1, laps: 5}
```

lap 0 始终播放原始身份。上例 lap 1–5 的 lap-0 保留概率依次为 `0, .25, .5, .75, 1`，之后保持 1。线性 `laps` 是达到终值的轮号，取值 `[2, 2147483647]`；`start/end` 均在 `[0,1]`。反向设置可逐轮降温。

也可使用 `retain_schedule: {kind: sequence, values: [1, 0.8, 0.3, 0]}`；值从 lap 1 开始，末值延续，序列必须单调（允许相等），长度 1–4096。不能与 `retain_probability` 同时声明，只能用于 `identity: partial`。

`retain_schedule` 以原始块 key 和 seed 的固定 hash 决定是否保留 lap 0；概率升高时保留集合嵌套扩大，降低时缩小。未保留块用当前 lap 的新身份。前缀前段改变仍会改变后续块 key，因此不是对命中率的承诺。常数 `retain_probability` 继续逐代回溯保留，算法和输出均不变；`none`、`structural-relabel` 不变。

## 原始环境变量与分片

| YAML 字段 | 原始 env | 格式 |
| --- | --- | --- |
| `rate_curve` | `RATE_CURVE` | JSON 点表 |
| `arrival` | `ARRIVAL_PROCESS` | `deterministic` / `poisson` |
| `retain_schedule` | `LAP_RETAIN_SCHEDULE` | JSON 对象 |
| `seed` | `PLAYBACK_SEED` | signed int64 |

`playback` 块不能混入原始 pacing/identity env。这些环境变量在 Python 启动层和 Java 均校验。高级播放要求 canonical token trace，关闭 `REPLAY_UNIQUE_PREFIX`，不接受运行时输入/输出截断、priority override 或 GRADIENT；在生成阶段完成所需投影。

配置 `rate_curve`、`arrival: poisson` 或 `retain_schedule` 后使用 `PLAYBACK_CONTROLS_V1`：在整个多轮序列上按 `global_index % num_shards` 分片，LIMIT 是全局上限。即使 trace 长度不能整除 shard 数，合并后的计划也不会重复或缺失到达位置。Poisson 的 seed 不含 shard ID，所有 shard 推进同一全局随机流。轮转 rid 仍保留 `_S{shard}_L{lap}` 防冲突，因此跨 shard 数比较应按源事件与 lap 判断语义相同，不比较带 shard 后缀的 rid 字符串。

未配置上述控制时使用 legacy 分片行为。若比较两种分片算法，使用 trace 条数可整除 shard 数的负载，或将差别列为实验变量。无限轮转的事件预算按曲线积分计算，Poisson 还按 seed 计算窗口内确切计划数；超出证据容量直接拒绝，不按均值猜一个可能丢数据的容量。

## 证据和离线复原

`flow-input.json` 的 trace manifest、门禁 `provenance.trace.playback` 与 Java `playback.json` 保存播放参数。Java 快照额外记录 planner 版本、实际 pacing 参数、trace SHA、token stride、分片和停止条件；只改输出分布也有完整的计划证据。分布策略及采样版本存于 trace manifest。A/B 差异检查把播放参数、seed、输出分布差异标为 `DIFFERENT / 不可直比`；绝对门禁独立判定。

使用**本次运行同一 jar**和快照，离线导出计划，不连接 master 或 engine：

```bash
java -cp flexlb-mock-engine.jar org.flexlb.mockengine.PlaybackPlan \
  evidence/playback.json pinned-trace.jsonl plan.jsonl
```

按实际归档路径替换以上参数。输入必须是生成后的 canonical JSONL，SHA 不符会拒绝。输出为每请求的全局索引、相对 `due_seconds`、rid、il、ol、iteration、block_keys；同快照同 trace 输出一致。支持 `PLAYBACK_CONTROLS_V1` 和带完整证据字段的 `PLAYBACK_LEGACY_V1`；后者按原有逐轮分片规则复原。历史快照缺少版本或完整字段时拒绝，运行时截断/GRADIENT 等覆盖也会拒绝。这里复原的是计划，不是 RPC 完成时刻、实际延迟或因提前停止未发送的尾部请求。
