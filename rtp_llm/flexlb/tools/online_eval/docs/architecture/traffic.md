# 流量与播放模型

本页是解码、转换、播放三层职责的唯一架构契约。模块命令从 `online_eval` 目录执行，先设置 `PYTHONPATH=src:.`。

| 层 | 允许 | 禁止与校验 |
|---|---|---|
| 解码 | 按 manifest 版本校验 SHA、块大小和 DAG 引用，恢复规范事件及匿名标签 | 不按长度筛选、不重采样输出、不决定条数或节奏。版本/块大小由 `traffic.codecs` 与各 codec 拒绝不匹配输入。 |
| 转换 | 在完整展开后做输入长度**过滤**、输出分布采样、请求条数限制，或为派生匿名模板做输入长度**截断** | 不改写源 `.xz`；`lineage_transforms` 校验参数并保留被排除父请求的标签，`derive_master_templates` 对截断量计数。新增变换须记录源 SHA、参数、选中/排除或截断量。 |
| 播放 | 配置速率和循环；Java 客户端依据摊平 JSONL 控制发送 | 不读 `.xz`/DAG、不按 `il` 或标签改内容。`traffic_source.validate_plan` 校验摊平行，Java `filterAndShard` 只按时长、数量和 shard 控制；新增 Java 内容变换属评审约束，当前无独立静态检查。 |

源 `.xz` 与 sidecar SHA 不可变；场景使用 `path + count` 选择输入，不声明预期 SHA；生成时计算实际源文件摘要并写入 manifest。转换必须保持 `shared ≤ parent.blocks`，先展开所有父事件再过滤，不能把被排除父请求从标签空间删除。`input_length_filter` **丢弃整请求**，不缩短其输入；`fixture_shape_clip` **截断输入**并记录删去 token，二者不可同名混用。`request_limit` 在过滤之后计数，输出长度分布按原始事件索引采样。产物 manifest 的 `transformations` 保留源 SHA、应用项和影响计数；未转换时 `applied: []`。捕获行契约另见 [数据规则](../../data/README.md)。

## 源层只生成内容

场景通过统一 `trace/prefix_lineage` 入口加载真实流量，codec 由数据内容确定。合成生成器只用于离线工具，不注册为场景 source。功能场景内联请求不变。

| 源 | 数据 | 时间字段 | 口径 |
|---|---|---|---|
| `trace/prefix_lineage/3` | SHA 固定的 XZ 精确长度模型 | 原始相对毫秒 | 新采集默认；完整块共享，残缺尾块私有，总长精确 |
| `trace/prefix_lineage/2` | SHA 固定的 XZ 列式模型 | 原始相对毫秒 | 实测前缀结构；token 长度块对齐有损 |
| 离线 `synthetic/realistic/1` | seed、长度分布、族/会话、精确块 | 请求序号 | 参数分布；客户端负责节奏 |

场景通过 YAML 声明的 source specification 将匿名模型物化到运行目录。真实流量只需 `kind`、`model`、`parameters`，不必重复填写数据版本。可选 `version` 是一致性断言：与文件内容不一致即失败，不能强制换解码器。

`traffic.codecs.decode_model` 从 v2 二进制 magic 或 v3 envelope 的版本识别格式，并恢复统一 `LineageModel`。不根据文件名、长度是否恰好对齐或解码异常回退猜测。未知版本、损坏数据和 manifest 冲突均拒绝。物化 manifest 同时保存 `requested_source`、包含实际版本的 `source` 和 `source_codec`，自动加载仍保留可复现的格式证据。

增加采集格式时在 codec 层实现并注册解码器，转换和场景不维护版本名单。未来含输出长度的格式还需在统一事件模型中明确实测输出字段及缺失值语义，并定义实测长度与输出采样策略的选择；自动识别格式不会把当前采样长度解释成实测长度。
`TRACE_FILE` 只作为生成后的 Java 发送器输入，不能指定原始日志。未知版本直接拒绝。

### Lineage v3

v3 保存原始事件的精确输入长度、相对到达时间、parent 和共享完整块数。
残缺尾块分配私有标签，不参与匹配；其长度仍计入总输入 token。
采集契约及兼容、归档规则见 [数据规则](../../data/README.md)。

### Lineage v2

4 个 varint 列分别存储 Δts（zigzag）、完整块数、parent 回指距离、shared。
整个模型用 XZ 压缩。另有一位/事件的占号位图：partial block
占用匿名标签号，但不输出尾块。这样后续 fresh 标签
不会整体错位，原有完整块的 token 标签与 Java hash 输入可以精确保持。

丢尾不能宣称“每请求误差 ≤0.5%”。例如 1023→512 会减少约 50%；小于
512 的输入补至一块可能膨胀很多。manifest 分别记录删去/补入 token、
短请求数和总体相对变化。补齐的短请求使用独立身份，不制造共享命中。

manifest 携带 realism、tail、arrival、采集窗、分片覆盖信息、模型
SHA、token 调整统计。输出长度从独立的 `output_tokens` 指定，不能由错误
截断为零的采集结果推断。不存在 source `qps` 参数。

```sh
# 默认生成 v3；v2 仅用于带理由的历史复拟合。
python3 scripts/pipeline/fit_frontend_prefix.py --source CAPTURE_DIR --out FIT_DIR \
  --expected-shards SHARD_COUNT --output-tokens 420
```

### 采集与数据边界

采集和拟合入口位于 `scripts/pipeline/`。`capture_frontend_prefix.py` 逐行读取日志，
通过 `CaptureRecord` 校验后写入匿名记录；`fit_frontend_prefix.py` 根据累计前缀摘要
建立父引用。`CaptureMetadata.block_size` 由文件携带，默认 512，支持 64 至 4096 的
二次幂；同一次拟合的所有分片必须一致。日志 JSON 损坏时保留已提取记录，但 summary
标记为不完整，拟合拒绝消费。不完整记录不会占用去重身份。

`traffic.catalog` 负责发现输入文件，`traffic.datasets` 负责 manifest 生成与准入校验，
`traffic.statistics` 负责观测统计。codec 仅处理存储格式；所有版本解码成
`LineageModel`，由 `LineageExpander` 统一展开。保真度分析保留前 4K 标签，
完整标签分配和 v2 丢弃尾块的预留仍遵循同一算法。长度统计保留 v3 的精确 token 数。

### Realistic

`seed`、`count` 必须指定；block_size 默认 512，支持 64 至 4096 的二次幂。输入/输出
分布通过 `{values: [...], weights: [...]}` 独立采样；可用离散混合分布或
经验分位点作为 values。输出必须显式给 `output_distribution` 或
`output_tokens`，没有从故障窗口捏造默认值。

保留 Zipf、cold_fraction、shared/prefix/suffix_blocks、session_requests、
session_growth_blocks。生成输入使用紧凑块编码；
整型项表示重复 token 块，数组项表示一块完整的精确 token 内容。
`pinned_blocks` 支持指定文档前缀，输入长度不得截断它。输出随机流独立，
改变输出分布不会改变输入共享关系。

合成参数画像保存经验长度分布、族集中度、冷比例和共享深度。生成方式：

```sh
python3 scripts/pipeline/derive_synthetic_profile.py --model lineage-model.xz \
  --fit-report fit-report.json --out profile.json
```

只有分位点/Top5 占比无法识别 session 多轮语义；因此默认关闭 session
增长，支持实验者显式配置。重复间隔和分钟流量曲线留在参数反推 provenance，
不伪称静态 Zipf 已复现完整时间相关性。profile 标记
兼容字段 `CALIBRATED_STATISTICAL`、`held_out_validated=false`；显式实验形状标记
`NOT_VALIDATED`。任何参数覆盖都留在原始 source specification 中。

## 播放层

场景 `client.playback` 只声明发送节奏、圈数和跨轮身份。Java 不改变已物化请求的输入长度、输出长度或 priority。操作命令见 [播放与复现](../development/playback-controls.md)。

- `uniform` 按 QPS 发送，可选 `ramp_up_seconds` 线性升压后保持恒速。
- `true-ts` 保留真实时间间隔，并用 `speed` 缩放；Java env 对应 `SEND_MODE=replay`。
- `max_laps` 默认 1；正整数限定圈数，0 表示循环到明确的 `playback.duration_s`。uniform 不隐式循环。
- `identity: structural-relabel` 是默认值；lap 0 保持输入，后续圈对所有 token 加同一圈偏移，再重算块 key。圈内前缀共享不变，各圈 token 区间不相交。
- `identity: none` 显式保持各圈 token，用于有意测量热缓存。编号空间耗尽时失败，不回绕。

分片按源请求序号固定分配，每圈重复同一子集。uniform 的全局到达序号为 `sent_count × num_shards + shard_index`，各 shard 不把首条请求提前到同一零点。不能整除的分片在多轮下保持各自固定速率；合并后不声明等价于重新全局分片的单客户端序列。LIMIT 保持 Java 发送器的既有条数语义，框架 case 优先在物化时用 `max_requests` 选定输入，不重复设置 LIMIT。

`playback.duration_s` 是墙钟截止，先缩放真实时间再判断。stop/drain 负责停止发行并等待已经发出的请求完成。每条请求证据保留 iteration，不把多轮累计命中率解释为一轮的自然演化。

场景客户端按 playback、limits、transport 分层，禁止直接填写 Java 环境变量；不支持随机到达、任意速率曲线、burst/diurnal 和 partial 身份。无效字段在启动前失败。端点、flow 标识、制品目录及采集策略由框架注入，不能在 `traffic.client` 重复声明。

请求计划 manifest 记录内容变换与 SHA；Java `playback.json` 记录实际发送配置、`PLAYBACK_V1` 和 token stride。框架核对输入 SHA 和启动参数，并将快照嵌入 `flow-input.json` 的 `java_playback`。`PlaybackPlan` 用同一身份映射和发送公式离线重建各圈计划；Python 不实现调度算法。

Python 日志读取器按需保留事件，并以 `MAX_EVIDENCE_EVENTS` 限制证据容量；超出时明确失败。该上限不推演 QPS、lap 数或发送时刻。

## 比较边界

流量、长度过滤、前缀分布或播放策略改变后须重新建立对照，不能沿用原门禁结论。
A/B 检查 codec、realism、tail、arrival 与身份策略，不一致时提示命中率/TPS 绝对值不可直比。
发送器验证只证明播放行为，性能与有限缓存效应仍需独立验证。

联合结构采样和独立对比工具见 [合成保真度](../development/synthetic-fidelity.md)。

## 代码职责

`traffic/synthetic/` 负责离线统计合成：`configuration.py` 在离线 JSON 输入边界合并画像并解析为 `SyntheticWorkload`，`generator.py` 持有随机流、标签分配器和采样器，`derivation.py` 从固定捕获反推画像参数。独立采样与联合采样使用不同的形状对象；联合池控制长度与深度，独立模式控制固定前缀、会话增长和长度分布。输出采样使用独立随机流。

`traffic/playback/` 负责回放配置。`PlaybackConfigurationCodec` 将 YAML 或 Java 环境变量解析为同一个 `PlaybackConfiguration`，统一校验数值、枚举、圈数及跨轮身份；Java 环境与归档描述都是该对象的边界投影。Java 仍负责实际发送、节奏与循环。

捕获、谱系编码及投影模块保留在 `traffic/`，按输入契约、格式版本和转换职责划分；模块大小不决定是否单独拆文件。`traffic/token_plan.py` 流式读取 token plan，并统一显式 token 与紧凑块的完整块前缀身份；`analysis/workload_profile.py` 只对规范化请求迭代器计算前缀/LRU，`diagnostics/workload_profile.py` 连接文件读取与分析，命令入口为 `scripts/commands/profile_workload.py`；`analysis/traffic_fidelity.py` 提供纯保真度计算，`diagnostics/traffic_fidelity.py` 负责文件发现、来源审计与拟合生成编排，`reporting/fidelity.py` 负责保真度报告渲染。命令入口和库调用都是有效依赖，不能只根据运行时导入数判断文件是否无用。

## 类型化标定与准入复用

`request_rules` 只定义请求数值范围；`contracts` 只做驱动、priority 与门禁覆盖对账。
合成配置和真实投影不依赖播放配置。`randomness.event_uniform` 是稳定的事件索引随机数
原语；输出采样使用此口径，不改变已有种子序列。

`CapturedDataset.read` 校验 sidecar 与文件并返回已解码模型和输入 SHA。
模板生成、显式 lineage 物化及 profile 标定消费此快照，不再重复读取解压同一标本。
快照的路径、版本及请求数量必须与物化来源一致；报告记录实际消费快照的 SHA。
统计和标定接收 `LineageModel`，长度直接来自事件字段，块口径来自模型元信息。

采集、lineage 和合成生成器支持 64 至 4096 的二次幂块大小。
标定结果的 `lineage_tree_top5_share` 描述父引用树中的请求集中度；fit 报告的
`family_4k_top5_share` 描述相同前 4K 前缀的请求集中度，两者不得作为同一指标比较。
入库旧 profile 的 `family_top5_share` 按其原有 lineage 树语义解释，不修改固定输入。
