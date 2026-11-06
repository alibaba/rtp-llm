# Case 配置与生命周期

[返回使用说明](../README.md)

本文说明 Case 字段、引用、断言、启动配方、扩缩容和 Master 停启。下面的命令均在框架目录执行；首次运行先看 [README](../README.md)。

## Case 配置和编译

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

每个 Case 保存自己的声明和角色配置。配置与数据必须按以下要求组织：

- `config` 使用 Case 自己目录内的配置，文件统一放在 `configs/`，通过 `config: configs/master.yaml` 等相对路径引用；空配置可以直接写 `config: {}`。禁止跨 Case 引用配置，也不把角色配置放进共享 data；需要借用其他 Case 的配置时先复制到本 Case，再独立维护。
- 回放 `data` 使用共享的 `data/traffic/`，不在各 Case 内复制流量文件。例如 `requests_file: data/traffic/deepseek.binpb.zst` 默认从框架目录解析，也可以用 `--shared-root` 指定共享根。输入支持 JSONL 和 Protobuf + Zstd，编译阶段检查 Case 声明，播放时检查文件内容。
- 使用默认大盘时省略 `view`，框架自动使用 `views/default.json`；只有使用自定义大盘时才显式声明 `view`。

顶层为 `id`、可选 `name`、`describe`、`environment`、`phases` 和可选 `view`。Phase 顺序执行。每个 Step 直接写 `interface` 与 `args`，可选 `expect`、`within`、`wait_after` 和 `background`。phase 必须有正的 `timeout`。ID 全局唯一，引用必须来自前置步骤或允许的全局发现字段。

### ID 命名要求

新增或修改 Case 时，ID 统一使用小写 `snake_case`，并在同一个 Case 中全局唯一。Case ID 描述场景并与目录名一致，例如 `version_regression`；Phase ID 描述阶段，例如 `startup`、`performance`。

Step ID 必须采用“动作 + 对象”，必要时追加统计量或时序条件。启动和停止使用 `start_client`、`stop_client`；读取并保存结果使用 `get_master_a`；带断言的检查使用 `check_master_leader`、`check_prefill_tps`、`check_client_ttft_p95`、`check_client_success_rate_after_drain`。不要只写 `traffic`、`leader`、`drain` 或指标名。ID 用于引用和事件定位，具体参数、阈值与说明放在 args、expect 和 describe 中。重命名时同步更新 Step 引用及 View 的 annotation_steps。

这些是 Case 编写和评审要求；编译器目前只校验标识符格式、唯一性和引用，不强制动作前缀。

### Step 执行与等待

Step 默认在前台运行。`background: true` 让 Runner 在当前 Phase 并行执行该 Step。Phase 结束前等待未完成的后台 Step，时间计入 Phase 预算。一个后台 Step 失败时，Runner 取消同 Phase 的其他在途 Step 并回收。后续 Phase 可以引用已成功完成的后台结果。

固定等待写成 step 的 `wait_after: 5s`，默认 `0s`，必须是非负的带单位时长。执行顺序为“调用接口 → 检查 expect → wait_after → 下一步”；最后一个 step 也执行等待。等待计入该 step 的 within 和 phase 的 timeout，期间继续监视组件退出并响应取消；调用或断言失败不执行等待。返回值和 ref 仍是原调用结果，等待不会刷新结果或查询时间；等待后的检查需要下一次接口调用。

不再登记 `runtime/wait`，旧 Case 的等待步骤应迁到前一个业务 step 的 `wait_after`，不保留兼容入口或只等待的空 step。接口成功后等待超时/中断，记录保留其 result 和 `error_stage: wait_after`，不能把已知成功的写操作标成 outcome_unknown，也不会重试动作。

组件配置分为 `deployment` 和 `config`。原生业务配置原样保留；Mock 点分键在本层展开成对象，不展开原生 JSON 文本。重复键、YAML merge、循环引用、点分路径冲突均报错。`FLEXLB_CONFIG` 保留 JSON 字符串，编译器只检查它是对象且无重复键，业务字段由 Master 加载器检查。

Client 的 `deployment.replicas` 表示 JVM 数量，不能把总 QPS 乘以副本数。当前启动器不落实 CPU、内存或 heap 限制，配置这些限制会拒绝运行。配置编译通过不代表资源限制或其他未接通能力已执行。

下面的读取保存 Leader 地址列表，后续步骤直接引用整个返回值，不另造句柄或 selector：

```yaml
phases:
  - id: observe
    describe: 检查发现信息
    timeout: 10s
    steps:
      - id: check_master_leader
        describe: 检查唯一主 Master
        interface: master/get
        args: {is_leader: true}
        expect: {count: 1}
        wait_after: 1s
      - id: get_leader_prefill_engines
        describe: 记录主 Master 的 Prefill 发现结果
        interface: master/engine/get
        args:
          targets: {ref: steps.check_master_leader}
          role: prefill
      - id: check_prefill_tps
        describe: 检查所有 Prefill 的计算吞吐
        interface: engine/prefill/tps
        args:
          targets: {ref: globals.prefill}
          range: [-60s, 0s]
        expect: {gte: 1}
```

省略 targets 等价于该操作域全部实例；显式空列表、重复地址或未知地址报错。全局发现信息在 step 开始时取值；已保存的 step 返回值保持原样，不跟着发现变化自动改写。Client 控制接口单独约定：start 作用于全部已配置 Client，stop 无参数；指标接口仍可指定 targets。

### 查询和断言

普通指标优先使用语义接口，YAML 不写 PromQL。range 使用相对该 Step 固定查询时刻的区间，例如 `[-60s, 0s]`，不支持 `latest` 或 `all`。省略 range 时默认最近 60 秒；`client/success_rate` 必须显式提供 range，不隐式切换为整轮累计成功率。

数值指标的 `mode` 默认为 `interval`，Case 中省略 `mode: interval`，只有逐点检查时才写 `mode: series`：

- `interval` 返回区间汇总数值。Gauge 每个采样点先合并同实例 priority，再按实例取区间采样均值并求和；吞吐 counter 用区间增量除以时长；命中率、成功率分别汇总分子与分母的增量后相除，不平均比率采样点；延迟均值使用 sum/count 的增量比值，P95/P99 合并 histogram bucket 增量后计算。
- `series` 返回按采集周期排列的数值列表，gte/lte 要求每一个点都满足。Counter 派生指标每点使用四个采集周期的滚动统计长度，由实现确定，不另配置 lookback 或 step。

执行器固定每个 Step 的 `query_time`，指标实现记录实际查询点、表达式和时间范围，不重新取墙钟。缺点、缺实例、非有限数值或零分母均报错。`expect` 支持 eq、列表 count 或数值 gte/lte。eq 是结构相等，列表顺序参与比较；实例查询返回排序后的地址列表。普通数值返回 float，series 返回数值列表，不包装为 `{value, unit}`；单位在接口定义里。逐 Master 状态返回 `{master服务地址: [engine服务地址, ...]}`。

原始 `prometheus/query` 只接受 promql，在固定 query_time 执行一次 instant query；历史窗口写进表达式。返回 `{samples: [{labels, value, sample_time}], query_time}`，不自动把它压成数值或对全部序列应用同一阈值。空 vector 保留为空；业务数值门禁使用语义接口，缺数据报错而不是补零。

### 性能门禁接口

`client/start` 的 `reuse_cache` 是请求级缓存复用开关，默认 false，转发到原生 GenerateConfig；Engine 的缓存策略配置同时需要允许复用。仍使用原来的 start 接口，不增加独立复用动作或 Engine RPC。

以下查询均从 Runner 管理的 Prometheus 取数，支持上文的 range、mode 和实例 targets：

| 接口 | 指标口径 |
| --- | --- |
| `client/qps` | Client 已发请求 counter 的增量 / 时长，不是成功完成速率。 |
| `engine/prefill/tps` | Prefill 的 Real 同名 gauge `rtp_llm_context_tps`。 |
| `engine/prefill/tps_with_cache` | Prefill 的 `rtp_llm_context_tps_with_cache`，吞吐包含复用 token。 |
| `engine/decode/tps` | Decode 已提交输出 token counter 的增量 / 墙钟时长。 |
| `engine/cache/hit_rate` | Prefill 实际复用 token / 查询输入 token 的增量比值。 |
| `client/success_rate` | 成功终态 / 全部终态的增量比值，包括 error、timeout、cancelled。必须显式指定 range。 |
| `client/ttft` | 调用开始到收到首 token 的延迟，包含所有收到首 token 的请求。 |
| `client/tpot` | 首末 token 间隔 / 首 token 之后的 token 数，只统计成功且输出多于一个 token 的请求。 |

TTFT、TPOT 查询支持 `statistic: mean`（默认）、`p95`、`p99`，返回单位为秒，断言可以写 `lte: 200ms`。Prefill TPS 接口读取有效执行速率，Decode TPS 接口读取已提交输出的墙钟速率，两者分母不同；均不读取原生 generate/total 窗口计数。吞吐单位为 tokens/s，QPS 为 requests/s，比率为 0–1。`client/stop` 收集全部已发请求的终态；区间成功率不能替代这项完整性检查。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。Master/Client 按 replicas 启动独立进程；Engine 按 replicas 启动独立 JVM，编号 `<组件>-<序号>`；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。配方不再包含 bindings；目标选择和查询聚合由 Python 接口函数完成。

真实 Master 配方需要顶层 `zookeeper`；P/D/V 使用 `java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
zookeeper:
  command: [<Java21>, -cp, <ZooKeeper 的 classpath>, org.flexlb.mock.zookeeper.ZookeeperNode]
components:
  master:
    command: [<Java21>, -cp, <Master的classpath>, org.flexlb.MasterNode]
    startup_timeout: 60s
  prefill:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.engine.EngineMain]
  decode:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.engine.EngineMain]
  vit:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.engine.EngineMain]
```

配方必须覆盖 Case 实际声明的组件。组件通过 `MOCK_CONFIG_FILE` 读取本实例配置和完整 environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR` 提供身份和目录。初始化并绑定端口后原子发布 `MOCK_READY_FILE`，启动超时或提前退出都失败并回收已有进程。

统一 Engine JAR 的默认入口为 `EngineMain`。本地 Runner 为每个实例分配监听端口、配置和产物目录，启动 JVM 后等待 ready 和端点回读。Whale 每个 Pod 使用同一入口，由平台管理发现与进程。

宿主的 `host/create`、`host/stop`、`host/status` 是内部原子接口，不接收集群副本数。`create` 接收 Python 分配的实例 id、rpcPort 和 controlPort；metrics 与该实例的 control 共用 HTTP 监听端口。宿主控制端口通过 `MOCK_HOST_PORT` 指定。Python 在启动监听前选择本次实验未使用的端口，Java 绑定指定端口，Runner 再核对返回地址与分配一致。端口探测不构成预留，若被其他进程抢占，启动失败并结束实验，不偷偷换端口；不能先启动服务再检查是否复用了旧地址。

Case 使用 `engine/scale` 调整 Engine 实例数。必填参数 `direction` 为 `out`（扩容）或 `in`（缩容）。`role` 默认为 `prefill`，`count` 指定时取值为 1–1024。返回实际新增或关闭的服务地址列表。

| direction | 参数 |
| --- | --- |
| `out` | `count` 默认 1。复用该角色的 Case config，角色必须已配置。`targets` 仅适用于缩容。 |
| `in` | `targets` 省略时选择该角色全部活动实例，`count` 省略时关闭整个候选集。指定 `count` 时按候选顺序只选一次。显式 `targets` 保持输入顺序，否则按创建顺序。 |

空集合、重复/失效地址和超过候选数量的 count 在任何停止前报错。操作成功表示实例与发现文件更新完成，不表示 Master 已传播更新；在后续 step 读取各 Master 的实际观察。部分执行后失败会保留已知事件并结束实验，不重试或伪造整体回滚。

```yaml
- id: added
  describe: 增加一个Decode
  interface: engine/scale
  args: {direction: out, role: decode, count: 1}
- id: removed
  describe: 关闭刚增加的实例
  interface: engine/scale
  args:
    direction: in
    role: decode
    targets: {ref: steps.added}
```

主动下线会终止该实例的在途请求；生产 Master 淘汰旧地址前仍可能有路由失败，不能将当前操作称为无损缩容。Case 应按场景判断过渡期错误和收敛后的成功率。生命周期验收持续记录请求结果，验证两种 JVM 放置下剩余实例继续服务、真实 Master 最终移除旧地址、新实例能接请求，以及历史指标仍可查询。另用故意保留旧 Prometheus target 的场景检查采集不会串到新实例，并注入第二个创建/停止失败，检查已完成操作不重放、整次实验报错并清理。

每次新增使用新编号、目录和未用过的监听地址。编号仅用于进程管理与产物路径。Case 仍使用服务地址。每个实例在 `components/<实例>/` 保存配置、日志和端点，进程停止后保留证据。

`topology.json` 只保存当前活动实例，不生成 topology-history.json，不保存实例起止时间或退役台账。活动 Master 的 discovery.json 随当前 Engine 更新，角色缩到零时写空地址列表。Master 查询直接返回生产 WorkerDirectory 当前观察到的地址，不经实例 ID 或历史地址表转换；因此生产发现尚未淘汰的旧地址仍会如实返回。消费者的刷新和淘汰由自身实现负责，Case 检查其收敛结果。最终清理清空当前拓扑。

Prometheus 对每个实例独立抓取，以 RPC 地址作为 instance；宿主不重复输出业务指标。新增实例先完成采集基线，下线前完成末次抓取再移出目标；不保留退役目标、实例生命期或退役实例的末次采样台账。语义查询按当前活动实例选择范围并检查采样覆盖；时间窗只限定指标数据，不改变成员选择。原始 PromQL 直接查询 Prometheus 已采集的数据。

Runner 监视每个 Engine JVM 的退出状态。意外退出使实验失败。主动缩容停止指定进程，确认退出后撤销端点。

Python 运行环境持续存活；`ProcessSupervisor` 持有进程并监视退出，各角色服务负责自己的生命周期，`RunEnvironment` 统一安排最终清理。进程监督器不识别 Master 业务接口，也不合并返回值。真实 Master 的状态查询由 `interfaces/master.py` 按 targets 逐实例调用，并保留每个 Master 的独立结果。

`master/stop(targets)` 由 Python 停止选定的独立 Master JVM，省略 targets 时停止全部活动 Master；空集合、重复或失效地址在任何停止前报错。停止前完成末次采样、登记预期退出，确认退出后移出活动发现与 Prometheus 目标，删除 ready，保留原 endpoints、日志和配置。默认 `mode: terminate` 使用正常停止；若需要 SIGKILL 兜底仍报错。显式 `mode: kill` 直接杀掉选定进程组并等待回收，只有这次退出视为预期故障注入；其他实例退出、操作前已退出及部分停止失败仍报错，不重放已完成动作。

`master/start(count=1)` 的 count 范围为 1–1024，从 environment.master 的配置创建新 JVM，复用本实验的 ZK、HIPPO_ROLE 和公共服务端口，使用递增编号、新回环 IP、新产物目录。控制 HTTP 也绑定新 IP，防止旧控制引用命中新进程。初始启动和动态启动共用同一路径；补回时允许某个 Engine 角色已缩到零，按当前空发现启动。返回新 RPC 地址列表，选举和采集基线完成才返回。这里的 start 是新增/补回，不恢复旧身份。旧 step 引用始终指向旧地址，不能自动替换成新地址。Java 不提供 Master 集群生命周期服务。

`master/restart(targets)` 恢复本次实验中已停止的 Master，省略 targets 时恢复全部已停止 Master；活动、未知、重复或显式空目标拒绝执行。恢复保留原服务地址、逻辑身份和选举组，启动新进程，当前拓扑与采集重新注册。每次启动使用独立进程监督身份和 `components/<实例>/restart-<序号>/` 目录；原配置、endpoints 和日志不覆盖。管理器只在内存保留可恢复实例的配置与当前启动信息，不增加实例历史文件。`master/start` 仍表示新建实例，旧地址引用不会自动指向它。

滚动替换直接组合查询、stop、start。下面使用固定观察等待，实际 Case 应按需要配置等待时间及门禁；新主产生不保证在途请求无损：

```yaml
- id: old_leader
  describe: 查询当前主节点
  interface: master/get
  args: {is_leader: true}
  expect: {count: 1}
- id: stopped
  describe: 停止原主 JVM
  interface: master/stop
  args: {targets: {ref: steps.old_leader}}
  wait_after: 10s
- id: elected
  describe: 检查剩余实例产生唯一新主
  interface: master/get
  args: {is_leader: true}
  expect: {count: 1}
- id: added
  describe: 补回一个独立 Master JVM
  interface: master/start
  args: {count: 1}
  within: 60s
```

Engine 独立宿主在对象关闭后登记预期退出并回收；其他运行中退出仍是异常。Master 的 `/hook/pre_stop` 仅是生产生命周期钩子，不等于 JVM 已停止。

所有步骤共享所在 phase 的单调时钟预算，step 的 within 只能缩短它。Python 在引用解析后校验参数，返回后校验结果；Java 内部 HTTP 协议另有独立校验与剩余预算。写操作不自动重试，调用开始后的异常或超时记录 outcome_unknown；回执已返回而 expect 不满足记 FAIL，不记结果未知。

进程退出、调用异常、空指标结果或门禁失败都会中止后续步骤并收尾。收尾对本次拥有的进程组先 TERM 后 KILL；主动取消也清理。结果为 PASS/FAIL/ERROR/CANCELLED，清理或采集异常不能得到 PASS。

运行目录不覆盖已有目录，保存原始 Case、编译配置、启动配方、接口清单、代码版本和工作区差异、组件配置和日志、phase/step 时间线、门禁查询时刻/真实值/阈值/结果及实际 PromQL。Client 逐请求记录和离线读取已接入，Engine 全链路记录尚未实现。测试 TransportProbe 只验证进程与传输，不替代业务实现。

## 组件配置文件

随仓 Case 将 Master 和 Engine 自身的配置放到 Case 目录下的 `configs/`，在 `environment.<role>.config` 填写文件路径：

```yaml
environment:
  master:
    config: configs/master.yaml
  prefill:
    engine: rtp
    config: configs/prefill.yaml
    deployment:
      replicas: 48
  decode:
    engine: rtp
    config: configs/decode.yaml
    deployment:
      replicas: 192
```

路径相对 `case.yaml` 解析，不依赖启动时的工作目录；只读取本地 YAML/JSON mapping，不加载远程配置。文件内容就是原 `config` mapping：Master 文件保存原生环境变量字符串（`FLEXLB_CONFIG`、`MODEL_SERVICE_CONFIG`），Engine 文件保存容量、调度和 Compute 参数。`engine` 后端选择、部署数量、资源/堆设置，以及 phase、step、流量与判定参数仍放在 Case YAML。Client 的发压参数也留在 Case。

编译时读取文件并沿用内联配置的重复键、点路径冲突和 Master JSON 校验，结果展开到原有组件配置和运行产物，Runner/Java 不需要重新读取源文件。不增加文件继承或覆盖合并；小配置与测试夹具仍可直接使用内联 mapping。

Engine 配置名、Mock 参数命名、ENV 映射和忽略规则见[组件配置参考](component-reference.md#engine-配置规则)。Case 使用每实例文件作为唯一配置源；本地 Engine 不从进程 ENV 覆盖业务配置。

### 可读回放数据

Case 的 `client/start.args.requests_file` 可以引用独立数据集。示例见 `data/traffic/basic.jsonl`。共享路径以 `data/` 开头，其他相对路径按 Case 所在目录解析，也接受本机绝对路径。JSONL 每行一条请求，字段为 `token_ids`、`max_new_tokens`，以及可选的 `priority`、`arrival_offset_ms`、`images`。Client 完整校验输入后才发布分片。现有 `.binpb.zst` 大流量数据可继续回放。
