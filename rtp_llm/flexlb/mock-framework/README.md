# Mock 框架

这是独立重写的实验框架。Python 编译 Case、编排本次运行的进程、执行步骤和门禁；Java 定义公共接口并实现组件。RTP Mock 使用生产协议，真实 Master 使用生产调度与 ZK 选举；共享组件不依赖旧 Mock 或 online_eval。

当前已接通本机真实 Master、新 RTP P/D、Client 原生收发核心和 Prometheus。每个 Engine 独立 JVM，实例间使用网络 gRPC。Client 播放器、多 JVM 发压协调、Master 停启和滚动重启、动态缩容、ViT/vLLM 及历史查看服务尚未接通；Java 接口已声明不等于运行能力已实现。已有验证是小流量功能接线，不能替代开发机/CI 的规模和压力验收。

## 代码职责

| 目录 | 负责内容 |
| --- | --- |
| `python/mock_framework/config/` | YAML 解析、检查、点分配置展开、引用和可执行步骤编译。 |
| `python/mock_framework/execution/` | 初始化、顺序执行、HTTP 传输、断言、记录和清理；`master.py` 负责 Master 配置接线及逐 Master 读取聚合，`processes.py` 只管理进程和端点。 |
| `java/interfaces/` | 公共接口、参数/返回类型及清单生成，是组件接口的唯一权威定义。 |
| `java/runtime/` | HTTP 接口绑定、参数和回执校验、调用上下文。 |
| `java/master/` | 真实 Master 的启动入口、生产状态到公共读取接口的适配、独立 ZK 启动入口。 |
| `java/client/` | 生产协议的调度、Fetch/Generate、真实客户端计时与取消。 |
| `java/engine/rtp/` | RTP 配置、原生 RPC、请求执行、P/D 交接与前缀树缓存。 |
| `java/shared/compute/` | Forward batch、公式、噪声、Decode 推进、停止模型和可取消计时。 |
| `java/shared/cache/` | 每个池独立的物理 block 容量、所有者引用与释放。 |
| `java/shared/metrics/` | Engine 指标、注册表、Prometheus 导出与查询。 |
| `tests/`、各模块 `src/test/` | Python/跨语言集成与 Java 测试；探针不作为正式业务入口。 |
| `scripts/` | 构建和检查入口。 |

共享指代码复用，不是不同实例共享资源。RTP 策略留在 RTP；后续 vLLM 可以复用 batch、计算、噪声、容量和指标组件，自行实现引擎协议与策略。

## 构建与检查

在本目录使用 Java 21、仓库 Maven Wrapper 和 uv：

```sh
export JAVA_HOME=<Java21目录>
./scripts/build.sh
PROMETHEUS_BIN=/path/to/prometheus ./scripts/check.sh
```

`build.sh` 运行 Java 测试并生成 `build/interfaces.json`；`check.sh` 继续运行 Python 测试。缺少 `PROMETHEUS_BIN` 时跳过真实采集集成测试。多 Master 测试需要操作系统支持对应的 127/8 地址；macOS 未配置回环别名时跳过双 Master 测试，不自动改系统网络。生成目录不入库，Python 依赖由 `uv.lock` 锁定。

构建复用生产 `flexlb-common`、`flexlb-grpc` 的公式与协议。Master 还使用生产 cache/sync/api，Spring 只进入 Master 运行时。框架构建跳过 Boot 重打包和 Whale 分发归档，不改变生产模块独立构建的默认行为。

## Case 配置和编译

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --catalog build/interfaces.json --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

每个 Case 独立目录组织自己的配置，数据集可共享。私有相对路径基于 Case 目录；显式提供共享根时，`data/`、`datasets/`、`views/` 基于共享根。编译不启动进程、不查询指标，也不检查数据内容。最新数据格式由 Java Client 读取；播放器和数据读取尚待实现，不提供旧格式兼容、转换器或生成器。

顶层为 `id`、可选 `name`、`describe`、`environment`、`phases` 和可选 `view`。phase 和 step 完全顺序执行；每个 step 是 `call` 或带 `expect` 的 `check`。phase 必须有正的 `timeout`，动作可设 `run` 和 `within`。ID 全局唯一，只能引用已定义的前置步骤返回字段。

组件配置分为 `deployment` 和 `config`。原生业务配置原样保留；Mock 点分键在本层展开成对象，不展开原生 JSON 文本。重复键、YAML merge、循环引用、点分路径冲突均报错。`FLEXLB_CONFIG` 保留 JSON 字符串，编译器只检查它是对象且无重复键，业务字段由 Master 加载器检查。

Client 的 `deployment.replicas` 表示 JVM 数量，不能把总 QPS 乘以副本数。当前启动器不落实 CPU、内存或 heap 限制，配置这些限制会拒绝运行。配置编译通过不代表播放器、资源限制或其他未接通能力已执行。

播放契约支持 trace 节奏或 uniform 总 QPS，二者专属字段不能混用。`input.prepare.output_tokens` 支持固定非负整数，以及以下两种分布；定义来自 Java `OutputTokens`，Python 不维护第二份参数表：

```yaml
output_tokens: 128
# 或整数均匀分布，含两个端点
output_tokens: {distribution: uniform, min: 64, max: 256, seed: 42}
# 或正态分布，裁剪后四舍五入，不做条件重采样
output_tokens: {distribution: normal, mean: 128, stddev: 32, min: 1, max: 512, seed: 42}
```

随机分布必须有 int64 seed，token 上下界为非负 int32；`min≤max`，数值有限，`stddev` 可为 0。这些目前只编译校验，实际采样待 Client 播放器实现。采样应按 seed 和稳定请求身份确定，不随分片/JVM/线程变化；重试保留同一目标。覆盖不修改数据集的 settings/observed 或输入拓扑，实际目标写入请求记录。

### 查询和断言

普通指标优先使用语义接口，YAML 不写 PromQL。例如 TTFT 契约为 `client/requests/ttft`，参数 `stat: avg/max/p99` 和 `range`；Client 指标实现尚未接入。查询范围为：

- `latest`：查询当前采集周期，具体统计方式由接口实现决定。
- `[-60s, 0s]`：相对于该检查固定时刻的明确区间，也可查询过去的区间。
- `all`：唯一一次 `client/traffic/run` 返回的 `started_at` 到本次检查时刻，不从 Prometheus 启动算起。使用前必须已经发流，排空后也不允许再次 run。

编译保留相对范围，执行器固定 `query_time` 并记录绝对范围；查询后端不能在收到请求时重新取时钟。`expect` 支持 `eq`、`count`、`gte/lte`；秒单位阈值可写 `2s`。集合 `eq` 忽略顺序、拒绝重复身份；Master 的 Engine 集合逐 Master 比较，不取并集掩盖差异，空 Master 集合不能通过。

特殊查询使用 `prometheus/query`，只接受 `promql`，没有外层 `range`。它在固定 `query_time` 执行一次 instant query；历史窗口写进表达式。报告记录查询点、表达式和样本，不推断表达式内部的门禁范围或自动生成曲线。普通语义接口不接受原始查询字段。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --catalog build/interfaces.json --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。每类组件按 replicas 启动独立进程，编号 `<组件>-<序号>`；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。`bindings` 按最长接口前缀定位入口，不承担指标求和或 Client 数据分片。

真实 Master 配方需要顶层 `zookeeper`；P/D 使用 `java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
zookeeper:
  command: [<Java21>, -cp, <Master的classpath>, org.flexlb.mock.master.ZookeeperNode]
components:
  master:
    command: [<Java21>, -cp, <Master的classpath>, org.flexlb.MasterNode]
    startup_timeout: 60s
  prefill:
    command: [<Java21>, -jar, <RTP的JAR绝对路径>]
  decode:
    command: [<Java21>, -jar, <RTP的JAR绝对路径>]
bindings:
  master/: master-0
  engine/: prefill-0
  prometheus/: prefill-0
```

配方必须覆盖 Case 实际声明的组件。组件通过 `MOCK_CONFIG_FILE` 读取本实例配置和完整 environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR` 提供身份和目录。初始化并绑定端口后原子发布 `MOCK_READY_FILE`，启动超时或提前退出都失败并回收已有进程。

Runner 是持续存活的进程所有者，负责启动、退出监视和最终清理。`Processes` 不识别 Master 业务接口，也不合并返回值。真实 Master 的状态查询由 `execution/master.py` 逐节点调用；其他接口使用正常 binding。聚合由是否使用原生 Master 配置决定，不由是否配置 ZK 来改变业务分发。

**Master 的 stop/start 尚未接通。** 后续必须由 Runner 侧调用适配器使用进程管理能力实现，不能依赖被停止 Master 自身的控制端口。届时要在发出停止前登记有意退出，启动时恢复该节点端点和监视；意外退出仍须失败。当前任何组件运行中退出仍是异常，不把 `/hook/pre_stop` 当成已经实现的进程停启或滚动重启。

所有步骤共享所在 phase 的单调时钟预算，call 的 within 只能缩短它。HTTP 调用按接口 schema 校验参数/回执，Java 通过 `InvocationContext` 取得 run 模式、固定查询时刻、观测范围和剩余预算。后台动作由组件真正启动活动并返回回执，HTTP 层不虚构后台执行。动作不自动重试，超时须保留结果可能未知这一事实。

进程退出、调用异常、空指标结果或门禁失败都会中止后续步骤并收尾。收尾对本次拥有的进程组先 TERM 后 KILL；主动取消也清理。结果为 PASS/FAIL/ERROR/CANCELLED，清理或采集异常不能得到 PASS。

运行目录不覆盖已有目录，保存原始 Case、编译配置、启动配方、接口清单、代码版本和工作区差异、组件配置和日志、phase/step 时间线、门禁范围/真实值/阈值/结果。逐请求持久化及正式历史查看服务尚未接入。测试 TransportProbe 只验证进程与传输，不替代业务实现。

## 真实 Master 适配

`org.flexlb.MasterNode` 与生产 Application、同步器和调度器在同一 JVM，不另起 Master 子进程，也不实现 Mock 调度器。Case 的 `environment.master.config` 使用原生环境变量字符串：`FLEXLB_CONFIG` 是业务 JSON，`MODEL_SERVICE_CONFIG` 是路由 JSON。

Runner 先启动独占 ZK，再启动 Engine、Master 和 Client，与 YAML 书写顺序无关。已启动 Engine 的 `endpoints.json` 形成 topology/discovery；Runner 只替换 `MODEL_SERVICE_CONFIG` 的 hosts/discovery_file，保留 service_id、组和角色域名，不读取线上 VIP，不改写 `FLEXLB_CONFIG`。当前只支持一个 endpoint 组、分离角色和 gRPC；ViT 发现映射可表达，但 ViT Engine 尚未实现。

发现使用实际 Engine gRPC 端口，生产 WorkerAddressService 再转换内部 HTTP 身份，不假定 Mock 控制端口相邻。原始组件配置、实际环境、effective-config 和 discovery 分开保存；Master 暴露生产 `/prometheus`。

只绑定 `MasterStateApi` 的两个读取接口：

| 接口 | 行为 |
| --- | --- |
| `master/leadership/get` | 读取各 Master 的生产 isMaster 声明，返回声明自己为 Leader 的 Master 句柄集合。数量由 expect 判断，不伪造唯一 Leader。 |
| `master/engine/get` | 读取各 Master 的 WorkerDirectory，按角色映射 Engine 句柄，逐 Master 返回。未知地址报错，不能按部署数量补齐或将各节点取并集。 |

单/多 Master 全部使用生产 ZK 选举，不提供 standalone 分支。独立 ZookeeperNode 使用已有依赖、独立 JVM/数据目录；Runner 分配地址和本次 HIPPO_ROLE，运行时同步配置为 needConsistency=true、会话超时 6s，并保存实际值。ready 必须等到真实选举产生 Leader。

Master 使用 127.0.0.1、127.0.0.2 等地址，相同 HTTP 端口，gRPC=HTTP+2，适配生产以 IP 作为身份和共用转发端口的约束。端口启动时被抢占则失败，不静默换协议。Linux 默认支持 127/8；macOS 缺少别名时明确报错。实验之间不共享 ZK 或选举路径。清理先停止组件，再停止 ZK，最后停止 Prometheus，保留日志和数据。

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

## RTP 节点与公共校验

节点仅监听本机回环地址，独立 JVM 提供原生 gRPC、控制 HTTP 和指标端点。`endpoints.json` 写实际地址、支持的 RPC 和阶段范围；`/health` 只列已绑定的公共接口，未启用查询时没有业务接口。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      SEQ_SIZE_PER_BLOCK: "64"
      MAX_SEQ_LEN: "32768"
      MAX_BATCH_TOKENS_SIZE: "8192"
      REUSE_CACHE: "true"
      TEST_BLOCK_NUM: "1025"
      RESERVE_BLOCK_RATIO: "5"
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持 server_args 小写或 Whale 大写 env 名，同一字段同时提供两种写法报错；整数/布尔接受对应原生文本，其余原生内容保留，但不能据此宣称功能已模拟。当前要求显式 block size、max sequence length 和 test_block_num，只支持 `dp_size=1`、`dp_rank=0`、`max_context_batch_size=1`。不加载模型层/head 或 GPU 布局，不从部署内存推导 KV 块数。

RTP 请求公共支持范围集中在 `RtpRequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：多模态、adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 管理资源和状态，只有 GENERATE 后进入计算队列；一轮一个执行槽、轮转请求，尚不合并动态 batch。每轮增长 KV、执行公式等待后才提交输出；临时容量不足等待 allocation generation 变化，永久不足返回 602。

P 的内部 `PrefillTransfer.start()` 注册活跃来源并复用 LocalPrefill 计算。Forward 完成后 KV 仍由请求持有；只有网络 `Read` 成功交付后才完成 P 并释放。正式服务**只保留 Read RPC，没有外部 Prepare RPC**；测试从包内 start 建立来源或调用正式生成链路，不为探针注册额外协议。

D 的 LOAD 从 peer_addrs 对应 P 发起真实网络 Read，校验 client/request 身份、完整 prompt、block size 和单 Reader 所有权；不能用缓存中存在的 key 冒充活跃来源。返回元数据为 block size、输入长度、完整块 cache keys 和 stop_after；不复制 KV 张量，不模拟 RDMA/TCP cache-store 或带宽。D 验证成功且预算有效后才能确认 LOAD，传入 KV 不算 D 历史命中，也不提前提交为缓存。

Read 等待计算就绪；D 取消/超时会取消 P 来源，P 取消/关闭也使 LOAD 失败。`load_cache_timeout_ms` 默认 5s，非正值按原生回退 5s，同时受总 deadline 限制。没有 Reader 或 Read 入口校验失败时，来源仍由正式 P 请求所有者管理，受请求取消、timeout/deadline 和节点关闭约束。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0、文本 P/D 和 max_new_tokens>1。入队 ACK 前必须取得 D 分配确认；P 可在 Fetch 前计算并持有 KV，但不能先发 LOAD 或执行 D。Fetch 单次接入后使用相同输出链路，不缓存整段结果。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。incremental 返回新 token，否则返回累计输出；aux.output_len 始终累计。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。Mock 保护上限是 1024 活动请求，TaskHistory 保留 1024 项，缺失旧版本返回 OUT_OF_RANGE，不冒充完整历史。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。D 当前要求 input+max_new_tokens<max_seq_len，未实现真实截断/reserveStep。成功且 reuse/device 均开启时写回完整 prompt 块；失败不提交，生成内容不额外建索引。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

共享 `BlockPool` 只管理物理容量和引用。每个实例/存储层各自建池，每个所有者持有独立 Reference，close 幂等，不能释放其他所有者份额或混用池。REQUEST/CACHE/LOAD/STORE/EVICTION 分别计数；只有 CACHE 引用的块算可回收容量，free 和 available 分开，分配失败不部分扣账，释放唤醒等待者。

RTP 的 `BlockTree` 管前缀与叶子候选，`DeviceCache` 管准入、复用、增长、完成和取消。按 2026-10-28 fetch 的 main `7cdc40370e5129c5f3d70a40d7cdc54633a7955`，real 统一使用 BlockTreeCache；`lru/lfu/fifo` 是树内候选排序，不是 tree 之外的模式。只驱逐未引用叶子，按需逐个回收，保留可匹配父节点；match 刷新热度，release/重叠 insert 不刷新。

容量使用原生 `test_block_num` 和 `reserve_block_ratio`，不支持旧 `capacity.device_blocks/reserve_blocks`。block 0 不可分配，可用块数=test_block_num−1；token 容量再乘 block size。reserve 是原生百分比向下取整，默认 5%，约束首次分配，允许运行中增长使用预留。先引用命中再校验净新增需求，失败回滚；增长失败保留已取得部分，由请求最终释放。

仅缓存完整块，命中排除 prompt 最后 token；遵守全局/请求 reuse 和 device 开关，D 还遵守 disable_decode_first_malloc_device_reuse。成功完成才提交新 key，取消不提交。当前只有 Full-attention device 池；host/disk/remote 尚未接通，显式开启对应 enable_*_cache 会拒绝启动。共享容量类可供各层使用，迁移与驱逐策略仍由 Engine 实现。

## Compute、Decode 和停止模型

每个组只在自己的 config 中放一份模型，没有 `environment.compute` 或覆盖优先级：

```yaml
environment:
  prefill:
    engine: rtp
    config:
      compute.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
      compute.noise.scale: 1
      compute.noise.stddev_ratio: 0.05
      compute.noise.max_deviation_ms: 20
      compute.noise.seed: 42
      generation.eos_probability: 0.01
      generation.seed: 42
  decode:
    engine: rtp
    config:
      compute.expression: "1 + 0.1*totalStepTokens + 0.001*totalContextTokens"
      decode.draft_tokens: 3
      decode.acceptance_rate: 0.8
      decode.seed: 42
```

组件加载器负责读取自己的文件外壳。RTP 用 `RtpConfig.load()` 提取本组 compute，再调用 `ComputeModel.fromJson()`；共享 Compute 不认识 Runner 的 component/environment 格式，不再提供 fromComponentFile。测试探针自行提取配置。effective-config 保存补齐默认值后的结果。schema 来自 Java ComputeSpec/DecodeSpec/StoppingSpec，Python 不复制字段表。

`ForwardBatch` 保留 DP、请求输入长度/cache 匹配/目标输出/已生成/stepTokens/优先级/图片原始 shape。Engine 构造 batch；ForwardModel 只返回 Duration。默认 FormulaForward 只接受一个非空 DP，不暗自合并多个 DP。

公式复用生产 ArithmeticFormula 的函数、`param(name, value)`、`sum(...)`。param 初始值按解析时求值，只用常量；不新增参数映射。结果单位 ms，可保留到 ns，不设隐式 1ms 下限；未知变量、负数、非有限值和溢出报错。逐请求变量只能在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。`totalComputeTokens^2` 和 `sum(computeTokens^2)` 不等价。computeTokens=input−hit，contextTokens=input+generated，stepTokens 由 Engine 提供，图片不计算特征 token。

噪声先乘 scale，再加高斯；默认 scale=1，stddev_ms/stddev_ratio=0，二者按平方和开根合并。启用随机噪声须有 seed 与正的 max_deviation_ms，裁剪扰动后耗时下限为 0。按 seed、Engine 身份、角色和 batch 身份确定一次 Sample，不随调用顺序改变；不同真实 batch 使用不同身份。

`plan()` 只算耗时，`SimulatedCompute.submit()` 才等待；Execution 可取消，唯一终态为 COMPLETED/CANCELLED。关闭取消未完成项并拒绝新任务，回调在独立虚拟线程执行。共享 Compute 不提交缓存、不释放 Engine 资源，也不把计时完成当请求成功。

DecodeStepModel 默认每轮 1 token。推测配置用 draft_tokens、acceptance_rate、seed：按首次拒绝得到接受前缀 A，再提交纠正/奖励 token，即 C=A+1；固定独立接受率 α、草稿长 k 时是封顶几何模型，E[C]=1+α+…+α^k，见[经典推测解码论文](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)。这不宣称复现所有 MTP/EAGLE。计算按验证数收费，指标按实际提交数累计，KV 覆盖验证上下文，接近输出上限缩短草稿预算。

StoppingModel 只配在 P 的 generation，默认 eos_probability=0，随机概率要求 seed。固定 EOS hazard 是 Mock 的封顶几何近似，不是线上长度规律；遵守 min/max_new_tokens，ignore_eos 强制上限。P 按稳定请求身份只抽样一次，经 LOAD 传 stop_after，D 不重复抽样；跨过停止位置只提交允许部分，仍计验证成本。首 token 即结束时使用 PREFILL_COMPLETE 释放 D，不再执行 Decode。真实 observed 长度或输出覆盖由 Client 指定目标并 ignore_eos。

## Client 原生收发核心

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。请求只调度一次；要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向原 Master 尽力发一次有界 Cancel，不重试结果未知的请求；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。播放器、数据集读取、多 JVM 分片/总 QPS 协调、Client 指标与请求记录、traffic run/stop/drain 的业务实现仍待接通。

## 指标、采集和查询

EngineMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。固定标签 engine/role/node，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| mock_engine_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| mock_engine_requests_finished_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| mock_engine_cache_lookup_tokens_total / mock_engine_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| mock_engine_forward_tokens_total | Counter，kind=input/compute/output | 成功完成 Forward 的 token；P 为输入、输入减命中、1；D 的 input/compute 为验证数，output 为实际提交数。取消的 Forward 不补成功计算。 |
| mock_engine_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| mock_engine_requests | Gauge，state=waiting/running | 当前等待/执行数。 |
| mock_engine_kv_tokens | Gauge，state=total/available/free/held | token 容量，available 含可驱逐缓存，不等于 free。 |

累计量预建 0，抓取不清零；分母为 0 不补比率。以下实现放共享 PrometheusQueries，通过节点控制端口绑定，不为只读节点宣称 scale_in 能力：

| 接口 | 口径 |
| --- | --- |
| engine/cache/hit_rate | P 历史命中 token 增量／输入 token 增量，token 加权；可选 engines。 |
| engine/prefill/tps | P 实际计算 token 增量／完成 Forward 耗时增量。 |
| engine/prefill/tps_with_cache | P 输入 token 增量／完成 Forward 耗时增量，输入含命中。 |
| engine/decode/tps | D 完成 Forward 输出 token 增量／墙钟时间。 |
| prometheus/query | 固定 query_time 的一次 scalar 或 float instant vector，保留标签、数值和 sample_time。 |

普通接口过滤本次 job、engine、role，先逐条 Counter 算增量再聚合以处理 reset；hit_rate 不能平均节点比率。空 engines 报错，选择器转义。range 在固定 end 执行单次 instant query，窗口=end−start，精度 ms，低于 1ms 拒绝；latest 用检查前 60s 内最近两次采样的 irate，不平均整分钟。缺少采样或没有可计算 Forward 时报错。

原始 PromQL 不自动加 job/engine 过滤。matrix/string/native histogram、非有限值、查询 warning 或后端失败都不作为有效门禁；普通指标恰好一个有限值。原始空 vector 可以传回，但门禁为 ERROR。全部样本须满足 expect。查询不重试，HTTP 和求值均受步骤剩余预算限制。

### Runner 托管 Prometheus

启动配方可加入：

```yaml
prometheus:
  binary: /path/to/prometheus
  scrape_interval: 1s
  startup_timeout: 30s
```

Runner 不下载 binary；默认 scrape 1s、startup 30s，目前验证 Prometheus 3.5.0。它在独立运行目录启动采集器，从结构化日志读取系统分配端口，写入组件实际 config 的 prometheus.url/job，不修改原始 Case。托管模式与手配这两个参数冲突时报错。

组件 ready 后必须提供合法 metrics URL。Runner 原子更新 file discovery，固定节点全部成功抓取两次才执行 phase；动态目标增删尚未接通。每次独立 TSDB、job=mock，不要求用户定义实验 ID。若自用外部 Prometheus，组件可配置 `prometheus.url` 和 `prometheus.job`；调用方负责采集及实验隔离。

收尾等待每个目标在收尾开始后成功采样，再停止组件、ZK，最后 TERM Prometheus，超时才 KILL。最后采样失败、采集器异常退出会标记数据不完整并令实验 ERROR。启动失败/取消仍清理并保留现场。

prometheus/ 保存配置、targets、版本、日志、data/TSDB/WAL 和 collection.json；complete 仅说明采集边界及关闭成功，不代替业务指标或中途 up 判断。门禁 FAIL/主动取消也可以有完整采集。曲线不导出 JSON，数据不随收尾删除，后续回放服务在线查询；目前仍用默认保留策略，没有长期归档或正式报告服务。Grafana 大盘映射和 Client 指标尚未实现。

## 如何扩展公共接口

在所属 Java API 加 `@Endpoint` 方法，Master 状态放 MasterStateApi，动作放 MasterApi，Engine/Client 同理。一个方法接收一个 record 参数，无参数用 Types.None；参数注解描述范围/可选性，类型及返回引用直接从 record 生成。重新构建即可更新清单，Python 不新增枚举或改 compiler。

实现放组件自己的适配器，使用 ApiServer 绑定；进程生命周期动作由 Runner 适配器执行，不能把管理自身进程的能力放进被管理 JVM。新增整组 API 只在 InterfaceRegistry.standard 登记一次。同名接口或不支持类型生成时失败；跨语言测试验证新增定义可直接编译和拒绝错误参数，业务实现另测。
