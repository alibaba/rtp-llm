# 两源与播放层收敛

实现基线：`codex/ft-case-framework` 的 `4146012454`。前置源码经用户授权从
`rtp-llm-p-scale-cache-gate` 未提交工作区取必要依赖；没有覆盖原工作区。

## 源层只生成内容

注册表仅保留两个入口。功能场景内联 `requests:` 不变。

| 源 | 数据 | 时间字段 | 口径 |
|---|---|---|---|
| `trace/prefix_lineage/2` | SHA 固定的 XZ 列式模型 | 原始相对毫秒 | 实测前缀结构；token 长度块对齐有损 |
| `synthetic/realistic/1` | seed、长度分布、族/会话、精确块 | 请求序号 | 参数分布；客户端负责节奏 |

旧四项注册和 lineage v1 不再接受。压测入口现在也调用注册表，将匿名模型
或 `TRAFFIC_SOURCE_SPEC` 指定的参数化源物化到运行目录。`TRACE_FILE` 只作为
生成后的 Java 发送器输入，不能指定原始日志。旧场景必须迁移，不能通过别名
悄悄复用旧口径。

### Lineage v2

4 个 varint 列分别存储 Δts（zigzag）、完整块数、parent 回指距离、shared。
整个模型用 XZ 压缩。另有一位/事件的占号位图：旧实现为 partial block
分配过匿名标签，v2 继续保留该标签号但不输出尾块。这样后续 fresh 标签
不会整体错位，原有完整块的 token 标签与 Java hash 输入可以精确保持。

丢尾不能宣称“每请求误差 ≤0.5%”。例如 1023→512 会减少约 50%；小于
512 的输入补至一块可能膨胀很多。manifest 分别记录删去/补入 token、
短请求数和总体相对变化。补齐的短请求使用独立身份，不制造共享命中。

随库匿名模型含 141,113 事件：624,288 bytes。全量验证原有完整块标签、
相对时间戳均保持；实际总 token 变化 −0.2655%，其中 3,609 条短请求补齐。
这里只保留匿名结构，没有原始文本或原始 token 内容。

manifest 携带 realism、tail、arrival、采集窗、缺失 frontend 列表、模型
SHA、token 调整统计。输出长度从独立的 `output_tokens` 指定，不能由错误
截断为零的采集结果推断。不存在 source `qps` 参数。

```sh
python3 -m traffic.fit_frontend_prefix --source CAPTURE_DIR --out FIT_DIR \
  --expected-pods 20 --output-tokens 420
# 旧模型的一次性迁移入口；运行时注册表不支持 v1：
python3 -m traffic.prefix_lineage old-model.json.gz \
  --fit-report fit-report.json --out lineage-model.xz
```

### Realistic

`seed`、`count` 必须指定；block_size 默认 512，1024 为兼容值。输入/输出
分布通过 `{values: [...], weights: [...]}` 独立采样；可用离散混合分布或
经验分位点作为 values。输出必须显式给 `output_distribution` 或
`output_tokens`，没有从故障窗口捏造默认值。

保留 Zipf、cold_fraction、shared/prefix/suffix_blocks、session_requests、
session_growth_blocks。取消 128 块上限，所有生成输入都使用紧凑块编码；
整型项表示重复 token 块，数组项表示一块完整的精确 token 内容。
`pinned_blocks` 支持指定文档前缀，输入长度不得截断它。输出随机流独立，
改变输出分布不会改变输入共享关系。

默认 profile `glm-5.3_20260921_1400_15m` 由随库模型及其 fit-report 导出，包含稠密
经验长度分布、族集中度拟合、冷比例、共享深度。可通过如下命令重建参数：

```sh
python3 -m traffic.calibrate_traffic --model lineage-model.xz \
  --fit-report fit-report.json --out profile.json
```

只有分位点/Top5 占比无法识别 session 多轮语义；因此默认关闭 session
增长，支持实验者显式配置。重复间隔和分钟流量曲线留在校准 provenance，
不伪称静态 Zipf 已复现完整时间相关性。profile 标记
`CALIBRATED_STATISTICAL`、`held_out_validated=false`；显式实验形状标记
`NOT_VALIDATED`。任何参数覆盖都留在原始 source specification 中。

1 万请求固定种子检查：平均 90.6K（目标 90.5K），P99 514.6K（530.4K），
Top5 10.4%（10.85%），冷比例 10.02%（9.79%）。这些是样本内拟合结果，
不构成“典型线上分布”或留出集泛化结论。

## 播放层

新增可选曲线、随机到达、输出分布及逐轮保留计划，见 [播放调节与复现](../../development/playback-controls.md)。

场景 `client.playback` 显式声明节奏与循环。不能同时混入原始 pacing env。
原始 Java env 入口仍接受，供既有启动脚本迁移；同样不再隐式循环。

```yaml
client:
  DURATION_S: '600'
  MAX_CONCURRENCY: '8192'
  FETCH_OUTPUT_STREAM: 'true'
  REPLAY_UNIQUE_PREFIX: 'false'
  playback:
    mode: uniform
    qps: 240
    max_laps: 1
    identity: structural-relabel
```

- `uniform`：指定 qps，忽略源时间戳。
- `true-ts`：仅用于实测源，指定 speed（如 4），保留真实间隔并缩放。
- `gradient`：指定 qps 与 ramp_up_seconds，线性爬坡后恒速。
- `burst`：指定 qps、burst_factor、burst_period_seconds、burst_duty，
  平均周期强度归一化；可叠加 diurnal_amplitude / diurnal_period_seconds。
  这是确定性周期调制，不冒称已拟合随机 MMPP。

uniform/gradient 使用积分到达强度的逆函数；burst/节律使用 10ms 积分格，
在突发边界分段。发送晚到不会改变理想到达时间。多 shard 使用共同 trace
时间原点和全局序号，分片不能把每个 shard 的首条请求都提前到零时刻。
DURATION_S 是墙钟截止，先缩放时间再判断，不先截断源时间。

max_laps 默认 1；0 表示循环到明确的 DURATION_S。没有循环声明就结束，
uniform 不再自动 wrap。每条 issued/terminal/per_request 记录包含 iteration；
报告列出每轮实际发送窗口，请勿把多轮累计命中率当成一轮的自然演化。

身份策略：

- `none` 保持 token，用于有意的热缓存上限实验。
- `structural-relabel` 默认；每轮将 token 映射到不相交的新编号区间，
  对所有请求和 shard 使用同一映射，再按既有 Java 算法重算所有块 key。消费端的累计前缀/树关系保持，等价保留
  DAG 共享关系，避免仅改 key[0] 而 token/后续块身份不一致。
- `partial` 配 retain_probability；每个原始前缀块以可复现概率延续上轮
  身份。共享块采用相同决策。前面的块变了，后续保留块也未必形成命中，
  因此 p 是块身份保留率，绝不是跨轮命中率。

token 编号空间耗尽时报错，不能绕回或悄悄碰撞。历史
REPLAY_UNIQUE_PREFIX 不再用于修改首个 key，场景应使用 playback.identity。
播放参数写入 flow-input 的 trace manifest 与 Java playback.json。

## 迁移与验证边界

trace_scale_out 改为 realistic，原有执行节奏移入 playback。
cache_scale_in 使用 125P/536D 真实流量门禁；规模场景采用校准
profile。`config/scenarios/cache_scale_in.yaml` 为 v2 实测模型入口。
源码和模型 SHA 一并更新，历史实验归档保持不变。

规模场景更换长度/前缀分布后，不能要求门禁结论天然保持：旧 working_set
易缓存正是本次迁移的动机。判据没有调宽，必须重新运行固定源的旧/新
master A/B 才能产生新的性能结论。代码/发送器验证不能替代该实测。

A/B 报告检查 realism/tail/arrival 与跨轮身份策略，差异强制显示“不可直比
命中率绝对值”；缓存门禁 A/B 同时将这种差异计入控制条件未对齐。

2026-09-21 验证：Python 全量 976 项通过；最后的报告/运行时相关补充
53 项通过，multi-curve JS 合约检查通过。远端 111 的
`20260921_080828.` / `playback-final` 运行 Java 测试 42 项全通过，
包括实际 sender loop 的有限 uniform、4 倍真实时间回放、两轮 iteration
记录。sender 测试使用受控 Schedule 响应，只证明播放行为，不是规模性能验收。

联合结构采样、兼容性与独立观察工具见[合成保真度](synthetic-fidelity.md)。
