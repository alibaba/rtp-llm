# BTC prefill 离线命中率评估

从检出仓库根目录运行以下命令。需要 Python 3.8+；读取 .zst 需要 zstd 命令。Skill 随 BTC 模块维护在 `rtp_llm/cpp/cache/block_tree_cache/skills/btc-offline-replay/`。仓库级 `.agents/skills/btc-offline-replay` 是指向该目录的发现入口；也可把整个目录复制到自己的技能目录使用。

## 1. 将请求转换成有序的前缀 key

```sh
S=.agents/skills/btc-offline-replay/scripts
python3 "$S/prepare.py" part01.jsonl.zst part02.jsonl.zst \
  --format bailian --hash-mode raw --base-tokens 256 --domain shared --out trace
```

百炼输入每行含 request_id（字符串）、timestamp（Unix 秒）、input_length（token 数）、input_block_hash_ids（每项 16 位十六进制）。raw 使用已有回放工具的 64 位前缀组合算法；若日志已记录前缀 hash，选 prefix。数据源使用其他原始 hash 算法时，先按数据源规则转成前缀 key。每行只保留完整块，命中率分母保留整个输入长度。多个文件按参数顺序读取，同时间戳保留文件内外原顺序，遇到时间倒序会报出行号。具有不同实例或命名空间的流分别准备。

其他来源可输出 normalized JSONL。下面是单行示例。使用 --format normalized，省略 --hash-mode。

```json
{"request_id":"r1","timestamp_ns":1000000000,"input_len":513,"block_tokens":256,"key_form":"prefix","keys":[101,202],"instance_id":"shared"}
```

输出采用 BTCPFX01。默认每 1,000,000 条生成一个可直接回放的 .btcbin 文件，以 000000.btcbin 开始。--requests-per-file 可调节物理分片大小。新版 reader 按文件头中的粒度和记录数寻址；文件之间延续同一个缓存状态。旧格式的 tar 仍可读取。manifest.json 记录每片请求数、输入 token 总数和 SHA-256。块序列保持全局基础粒度。业务侧已做 CP 抽样的输入需要与后端另行对齐，重复抽样会改变结果。

## 2. 使用待测版本的原生后端

后端是调用者提供的 CPU 元数据回放程序。已有的 evaluation_replay 可直接通过下列接口接入；新源码版本可提供相同接口的适配程序。仅检出上游 main 或 DSv4.1 不会自动得到此可执行文件。准备二进制时随它分发所需动态库、构建说明及对应源码提交。确认回放保留 BTC 的真实缓存决策，在元数据层处理分配和迁移，避免分配 KV payload 或启动 GPU 计算。

在 tmux 会话中运行，预先提供最终区间的完整日志。
```sh
tmux new -s btc-replay
S=.agents/skills/btc-offline-replay/scripts
python3 "$S/replay.py" --backend /path/to/evaluation_replay --revision "$BTC_COMMIT" --config config.json --data trace --warmup 1000000 --measure 10000000 --pause-after 1000000 --out run-10m
```

按 Ctrl-B 再按 D 离开会话，进程保留。另一个终端查看阶段结果并继续。
```sh
test -f run-10m/native/PAUSED && cat run-10m/native/summary.json
# 确认 PAUSED 出现后才发送继续指令
test -f run-10m/native/PAUSED && touch run-10m/native/CONTINUE
tmux attach -t btc-replay
```

示例在 100 万预热加 100 万正式请求后暂停，继续到 1000 万正式请求结束。预热只执行一次，每次运行支持一个预先指定的暂停点。阶段 summary.json 可直接读取，最终 SUCCESS 出现后使用 analyze.py。进程保活仍占用元数据内存。

不支持原生检查点落盘和跨进程恢复。恢复树、资源引用与队列顺序需要依赖 BTC 私有实现，故删除检查点代码及专用 friend。进程退出或机器重启后须重新预热和回放。

配置 JSON 原样交给后端。容量单位、拓扑及默认值以该二进制为准。例如实验用 DSv4.1 后端读取 tiers.device、tiers.host、tiers.disk 的 full 和 swa 单 rank 逻辑块数。旧实验后端使用 CP=4 和固定 256 粒度；新版 reader 将日志粒度传给模型拓扑构造器，按模型实际支持范围运行。切换算法、粒度、配置、二进制或输入后应从预热重新开始。

可执行程序参数为 CONFIG CHUNK_DIR NEW_OUTPUT --requests 总终点 --warmup 预热数 --identity IDENTITY_JSON，可选 --pause-after 总请求暂停位置。Python 驱动器将正式请求暂停数加上预热数后传给后端。完成当前请求后写 PAUSED 并等待 CONTINUE，继续时删除两个标记。退出码 0 表示最终完成。输出 summary.json、match-lengths.u64 等分析数据。--resume 和 --stop-after 会被明确拒绝。

预热请求正常更新缓存。summary.json 的 total.requests、input_tokens、full_tokens.sum、joint_tokens.sum 仅累计正式请求，单位分别为请求、输入 token、FULL 命中 token、联合命中 token。match-lengths.u64 每个正式请求写两个小端 uint64（FULL、联合），继续时追加，summary 为累计值。后端可附带 timeline 和 evictions；各版本缺失的指标应标记为未采集。

## 3. 对比与复核

```sh
python3 "$S/analyze.py" baseline/run-10m candidate/run-10m
```

FULL 命中率 = FULL 命中 token 总数 / 全部正式输入 token 总数；联合命中率使用联合命中 token 总数作分子。达标率取本组联合命中率除以首组联合命中率，输出均为 0–1 比值，零基准输出 null。离线结果反映给定请求流和后端策略，线上并发、超时、传输延迟与 decode 另行评估。淘汰计数保留后端标签，按层和触发类型解读，区分调度与实际释放。

通过校验的阶段带有 SUCCESS 标记。保留数据清单、配置、二进制身份和完整输出目录，复现时使用相同输入顺序。准备工具发现输入损坏即退出；回放失败时保留诊断输出，重试使用新目录。

接入兼容 DSv4.1 测试配置的原生后端后，可运行 `python3 "$S/smoke.py" /path/to/evaluation_replay`。该用例使用合成请求，比较连续执行与进程内暂停继续的逐请求结果。

需要定位两种策略的分歧时，加上 `--cases 10`。工具对齐本阶段逐请求 FULL / 联合长度，给出变长、变短的请求数和损失最大的原始请求序号；逐 key 淘汰原因需要对应后端的事件追踪。

## 日志字段与自动判定

粒度来自日志中的 block_tokens。百炼记录省略该字段时，采样 input_length 和 hash 数，用完整块数与可选尾块的尺寸约束求交集，仅接受唯一正整数；不唯一或冲突则停止。--base-tokens 只作一致性断言。预处理开始时打印检查提示，判定后打印来源粒度、hash 类型和依据，写完后打印请求数、输出 hash 类型与清单位置。

百炼输入省略 --hash-mode 时先读取 key_form。没有声明时，检查首个文件最多一万条请求、保存最多 65,536 个不同块的前驱：若同一 key 出现在不同前驱之后（包括起点），在同一命名空间、相同算法且无碰撞的假设下可判定为 raw。无法区分时要求提供来源声明。手工指定与记录声明冲突会报错。混合来源应分开转换；这项抽样不能认证整个数据集的生产方式。

## 定位版本与构建

在完整的待测 Git 检出运行下列命令。locate 展示 HEAD 和 block_tree_cache 路径下 BUILD 中显式命名的规则。宏生成或改名、移动的包可以通过 Bazel query 定位，再显式传目标。编译命令接受一个目标，实际依赖闭包由 Bazel 按相同配置解析。示例标签仅适用于含该离线适配器的检出。

```sh
python3 "$S/backend.py" locate --repo "$RTP_REPO"
python3 "$S/backend.py" build --repo "$RTP_REPO" \
  --target //rtp_llm/cpp/cache/block_tree_cache/benchmark:evaluation_replay \
  --option=--config=cuda13 > build-receipt.json
```

输出包含提交、工作区状态、目标、构建选项、依赖和产物。缺少回放目标时需要接入该版本的适配器。元数据适配器应显式注入 transfer completion 和逻辑块池，保留引用、迁移、回滚及淘汰状态机。工具无法从任意源码自动推断哪些分配或 copy 可以删掉。每个新适配器须核验实际 payload 分配/搬移计数及失败路径。

## 网页报告

默认命令为 `python3 "$S/analyze.py" RUN --html report.html`。单版本模板仅依赖 Python 标准库，HTML 可独立打开。每段和累计值均先求 token 总和再相除，预热不进入分母。空段显示缺数，相同到达时间的请求归入同一段。旧版回放缺少逐请求到达和分层事实时保留图位，展示已知总量。

prepare.py 接受可选参数 --cluster、--prefill-ip、--capture-time，日志文件名自动记录。回放生成 RUN/report-environment.json。可按采集记录补充字符串字段 `source`（日志来源）、`cluster`（集群配置）、`prefill_ip`、`capture_time`（采集时间范围）。缓存实例标识与 prefill IP 分别展示。每段正式请求的 arrival-hits.u64 按序保存六个 uint64（纳秒到达时间、输入、FULL、联合、Host、Disk tokens）；Device 由联合减 Host 和 Disk 得到。report-stages 收齐各次续跑的正式请求记录。

显式跨版本比较示例：

```sh
python3 "$S/analyze.py" baseline/run-10m candidate/run-10m --cases 10 --html report.html
```

HTML 可离线打开，包含预热与正式区间、各组 FULL/联合命中率、达标率和回收计数器。--cases 会增加本阶段变长、变短和损失最大的请求。

## 自动构建与元数据验证

```sh
python3 "$S/backend.py" auto --repo "$RTP_REPO" --work /tmp/btc-build-new --option=--config=cuda13
```

默认只编译已提交的源码 HEAD。根标记 `btc-offline-backend.json` 指定固定目标、三项原生检查和 `dependency_contract`：`require` 列出必须参与的原生 BTC 与元数据目标，`forbid` 列出不应进入回放的计算目标。先对实际配置运行 cquery，再执行编译、自检与 smoke；依赖不符合约定时留下 dependencies.txt，不生成通过回执。无源码标记时停止。历史适配补丁已移除，不再通过打补丁把旧的 BTC 分析代码注入被测源码。

## 带图的报告

跨版本或显式指定理论基线时安装 Matplotlib。综合页面为每组生成单独的分析页，保留可折叠原始数据。原生 summary 的时间桶决定分段；累计命中率按 token 求和。指定 --theory-run 时使用同区间无限容量全保留参考组计算分段达标率。逐请求文件覆盖整个正式区间，进程内暂停继续后仍为完整累计数据。已有 btc-request-report-v1 格式报告数据也可直接复现按请求数分段的原版报告：

```sh
python3 "$S/charts.py" report-metrics.json report
```


### 新版本改变接口时的处理（维护本 skill 的代理执行）

在 auto 留下的独立 source 目录中，沿 BUILD 依赖和调用点定位该提交的 Coordinator 分配、match、insert、free，以及 Device/Host/Disk 池和 transfer 提交与完成回调。以当前原生 UT 作为接口契约样本。为该版本的池隔离 payload backing，为 transfer 注入仅更新元数据的完成回调，保留块 ID、引用、异步完成、失败回滚、水位、force_drop 和原有选块顺序。不得用历史完整实现覆盖当前版本的策略文件。把已验证的接口绑定与原生测试目标登记到根标记。构建并运行池的无物理分配测试、无实际 copy 检查、三层降层/终端淘汰与分配不足测试，再运行连续回放/进程内暂停继续逐请求一致性检查。无法验证时报告源码版本和具体接口，不能把其他提交的结果归入该版本。

元数据模式跳过模型 forward、decode、KV payload 分配与物理 copy；保留逻辑块分配、占用/引用、队列次序、联合匹配、迁移完成及失败回滚。新版 reader 可跳过未使用的 request_id 字符串和检查点之前的 key 反序列化；预热仍执行同样的缓存更新，不向正式阶段复制预热统计。

## BTC 重构后的回归验收

固定标签约定的是编译入口。源码中的 Pool、Coordinator、Transfer 等接口由 C++ 编译器检查；行为变化要由原生测试覆盖。使用干净源码检出和新的工作目录执行本节验证。

1. 运行 `scripts/backend.py auto`，用源码根目录的 `btc-offline-backend.json` 解析回放目标及其检查目标。核对回执中的提交、配置、二进制摘要和检查列表。目标移动时同步更新 BUILD 或标记。无标记版本应停止并报告缺少的入口。
2. 在无 GPU 设备的 CPU 回放环境运行。池测试覆盖逻辑块分配与引用、零物理 KV backing、禁止读取 payload 地址、无磁盘 KV 文件。迁移使用元数据完成回调，仍由被测 BTC 进行预留、提交、失败回滚和释放。若新代码新增 copy 或分配入口，应针对该入口增加禁止物理操作的检查。现有 copy 哨兵只覆盖已接入的路径。
3. 用小容量分别触发 Device → Host、Host → Disk、Disk 释放以及分配不足的 force_drop，核对源层释放、目标层占用、FULL 与 SWA 联合命中，并注入一次加载失败确认回滚和重试。改变策略时记录预期变化；仅重构时按相同输入、容量和请求顺序逐请求对照旧版 FULL / 联合命中长度及各层回收量。
4. 运行 smoke 检查预热、连续回放与进程内暂停继续；再用同一批真实日志比对逐请求结果。记录任何差异出现的首个请求。CI 可将上述命令接到 BTC 改动的验证任务，以退出码、构建回执和对照结果作为验收证据。
