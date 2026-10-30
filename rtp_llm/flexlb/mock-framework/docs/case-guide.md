# Case 配置与生命周期

[返回使用说明](../README.md)

本文说明 Case 字段、引用、断言、启动配方、扩缩容和 Master 停启。下面的命令均在框架目录执行；首次运行先看 [README](../README.md)。

## Case 配置和编译

```sh
uv run --locked mock-case compile <case目录>/case.yaml \
  --shared-root <共享资源根目录> \
  --output <输出目录>/compiled.json
```

每个 Case 独立目录组织自己的配置，数据集可共享。私有相对路径基于 Case 目录；显式提供共享根时，`data/`、`datasets/`、`views/` 基于共享根。编译不启动进程、不查询指标，也不检查数据内容。Client 只接受 start 参数 requests_file 指向的标准 Protobuf＋Zstd 文件，不提供运行时旧格式兼容。

顶层为 `id`、可选 `name`、`describe`、`environment`、`phases` 和可选 `view`。phase 和 step 完全顺序执行；每个 step 直接写 `interface` 与 `args`，可选 `expect`、`within` 和 `wait_after`；不再使用 call/check 外壳或 run 字段。phase 必须有正的 `timeout`。ID 全局唯一，引用必须来自前置步骤或允许的全局发现字段。

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
      - id: leader
        describe: 检查唯一主 Master
        interface: master/get
        args: {is_leader: true}
        expect: {count: 1}
        wait_after: 1s
      - id: leader_engines
        describe: 记录主 Master 的 Prefill 发现结果
        interface: master/engine/get
        args:
          targets: {ref: steps.leader}
          role: prefill
      - id: tps
        describe: 检查所有 Prefill 的计算吞吐
        interface: engine/prefill/tps
        args:
          targets: {ref: globals.prefill}
          range: [-60s, 0s]
        expect: {gte: 1}
```

省略 targets 等价于该操作域全部实例；显式空列表、重复地址或未知地址报错。全局发现信息在 step 开始时取值；已保存的 step 返回值保持原样，不跟着发现变化自动改写。Client 控制接口单独约定：start 作用于全部已配置 Client，stop 无参数；指标接口仍可指定 targets。

### 查询和断言

普通指标优先使用语义接口，YAML 不写 PromQL。当前支持 `latest` 和相对该 step 固定查询时刻的区间 `[-60s, 0s]`；`client/success_rate` 必须显式提供区间，不接受 latest 或 all，也不隐式切换为整轮累计成功率。其他指标的时间语义保持各自定义；`all` 的跨轮次查询窗口尚未实现，目前拒绝。

`latest` 使用前 60 秒内最近两次样本的 irate；区间使用 increase。执行器固定 `query_time`，指标实现记录实际查询点、表达式和窗口，不重新取墙钟。`expect` 支持 eq、列表 count 或数值 gte/lte。eq 是结构相等，列表顺序参与比较；实例查询返回排序后的地址列表。普通数值直接返回 float，不包装为 `{value, unit}`；单位在接口定义里。逐 Master 状态返回 `{master服务地址: [engine服务地址, ...]}`。

原始 `prometheus/query` 只接受 promql，在固定 query_time 执行一次 instant query；历史窗口写进表达式。返回 `{samples: [{labels, value, sample_time}], query_time}`，不自动把它压成数值或对全部序列应用同一阈值。空 vector 保留为空；业务数值门禁使用语义接口，缺数据报错而不是补零。

## 本地运行与生命周期

```sh
uv run --locked mock-case run <case目录>/case.yaml \
  --launch <启动配方.yaml> \
  --output <新的运行目录>
```

启动配方与 Case 业务配置分开。Master/Client 按 replicas 启动独立进程；Engine 的 replicas 表示逻辑实例数，编号 `<组件>-<序号>`，由 engine_mode 决定放置；Compute 是组内配置，不单独启动。`command` 是 argv，不经过 shell；cwd 相对配方解析。配方不再包含 bindings；目标选择和查询聚合由 Python 接口函数完成。

真实 Master 配方需要顶层 `zookeeper`；P/D/V 使用 `java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。Master classpath 是 `java/master/target/classes` 加 `target/dependencies.txt` 的内容。示意：

```yaml
engine_mode: shared  # shared（默认）：一个 Engine JVM；isolated：每实例一个 JVM
zookeeper:
  command: [<Java21>, -cp, <Master的classpath>, org.flexlb.mock.master.ZookeeperNode]
components:
  master:
    command: [<Java21>, -cp, <Master的classpath>, org.flexlb.MasterNode]
    startup_timeout: 60s
  prefill:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
  decode:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
  vit:
    command: [<Java21>, -cp, <Engine JAR绝对路径>, org.flexlb.mock.host.EngineHost]
```

配方必须覆盖 Case 实际声明的组件。组件通过 `MOCK_CONFIG_FILE` 读取本实例配置和完整 environment；`MOCK_NODE_ID`、`MOCK_COMPONENT`、`MOCK_OUTPUT_DIR` 提供身份和目录。初始化并绑定端口后原子发布 `MOCK_READY_FILE`，启动超时或提前退出都失败并回收已有进程。

统一 Engine JAR 的默认入口为 `EngineMain`，只启动配置指定的一个 P/D/V Engine，不启动管理接口。Whale 每个 Pod 用 `java -jar <Engine JAR>` 运行一个实例，生命周期由 Whale 管理。

本地 Runner 显式用 `java -cp <Engine JAR> org.flexlb.mock.host.EngineHost` 启动宿主；同一个宿主管理 P/D/V，不为 V 单设管理者。共享模式下 P/D/V 的 command、cwd、env 和 startup_timeout 必须一致；角色差异写入各自 config，不能用进程 env 区分同 JVM 的实例。宿主读取 `MOCK_COMPONENTS_DIR`，按实例目录读取配置并创建对象。Runner 的 isolated 模式仍使用每 JVM 一个对象的本地宿主，便于验证统一扩缩容协议；这不是 Whale 的启动方式。两种入口共用实例装配，RtpNode/VitNode 不再各自提供进程入口或可执行 JAR。

宿主的 `host/create`、`host/stop`、`host/status` 是内部原子接口，不接收集群副本数。`create` 接收 Python 分配的实例 id、rpcPort 和 controlPort；metrics 与该实例的 control 共用 HTTP 监听端口。宿主控制端口通过 `MOCK_HOST_PORT` 指定。Python 在启动监听前选择本次实验未使用的端口，Java 绑定指定端口，Runner 再核对返回地址与分配一致。端口探测不构成预留，若被其他进程抢占，启动失败并结束实验，不偷偷换端口；不能先启动服务再检查是否复用了旧地址。Python `execution/engines.py` 管理放置和实例归属，`interfaces/engines.py` 提供两项 Case 操作：

| 接口 | 参数和回执 |
| --- | --- |
| `engine/scale_out` | role 默认 prefill；count 默认 1、范围 1–1024。复用该角色的 Case config，返回新服务地址列表。角色必须已配置。 |
| `engine/scale_in` | role 默认 prefill；targets 省略时选择该角色全部活动实例，count 省略时关闭整个候选集。count 提供时按候选顺序只选一次；显式 targets 保持输入顺序，否则按创建顺序。返回实际关闭的服务地址列表。 |

空集合、重复/失效地址和超过候选数量的 count 在任何停止前报错。操作成功表示实例与发现文件更新完成，不表示 Master 已传播更新；在后续 step 读取各 Master 的实际观察。部分执行后失败会保留已知事件并结束实验，不重试或伪造整体回滚。

```yaml
- id: added
  describe: 增加一个Decode
  interface: engine/scale_out
  args: {role: decode, count: 1}
- id: removed
  describe: 关闭刚增加的实例
  interface: engine/scale_in
  args:
    role: decode
    targets: {ref: steps.added}
```

主动下线会终止该实例的在途请求；生产 Master 淘汰旧地址前仍可能有路由失败，不能将当前操作称为无损缩容。Case 应按场景判断过渡期错误和收敛后的成功率。生命周期验收持续记录请求结果，验证两种 JVM 放置下剩余实例继续服务、真实 Master 最终移除旧地址、新实例能接请求，以及历史指标仍可查询。另用故意保留旧 Prometheus target 的场景检查采集不会串到新实例，并注入第二个创建/停止失败，检查已完成操作不重放、整次实验报错并清理。

共享宿主关闭最后一个 Engine 后仍保留，允许再次扩容；独立宿主在关闭实例后回收进程。每次新增使用新编号、目录和未用过的监听地址。编号表示一次创建，例如删除 prefill-0 后再次扩容会创建 prefill-2，不填补旧编号；创建失败也不回收编号。它只用于内部对象映射和产物路径，Case 仍使用服务地址，不增加 node/generation 标签或选择器。实验目录和实例目录拒绝覆盖；Java 以 CREATE_NEW 写入 effective-config.json，另一宿主也不能重新使用旧实例目录。重新创建必须使用新编号与新目录，Runner 断点恢复、Engine 同身份原地重启不在当前能力范围。`hosts/engine-shared/` 或 `hosts/engine-<实例>/` 保存 JVM 日志，`components/<实例>/` 保存 config、effective-config、endpoints 和 ready，不复制宿主日志到各实例。下线删除 ready；endpoints、日志和配置作为该组件的原始产物保留，不另行维护实例历史。

`topology.json` 只保存当前活动实例，不生成 topology-history.json，不保存实例起止时间或退役台账。活动 Master 的 discovery.json 随当前 Engine 更新，角色缩到零时写空地址列表。Master 查询直接返回生产 WorkerDirectory 当前观察到的地址，不经实例 ID 或历史地址表转换；因此生产发现尚未淘汰的旧地址仍会如实返回。消费者的刷新和淘汰由自身实现负责，Case 检查其收敛结果。最终清理清空当前拓扑。

Prometheus 对每个实例独立抓取，以 RPC 地址作为 instance；宿主不重复输出业务指标。新增实例先完成采集基线，下线前完成末次抓取再移出目标；不保留退役目标、实例生命期或退役实例的末次采样台账。语义查询按当前活动实例选择范围并检查采样覆盖；时间窗只限定指标数据，不改变成员选择。原始 PromQL 直接查询 Prometheus 已采集的数据。

Runner 每秒开始一轮宿主健康检查，单次 HTTP 调用限时 10 秒，检查实际成员与 Engine 状态；进程退出另行实时监视。操作期间用同一管理锁隔离成员变更与健康快照。共享宿主意外退出记录受影响实例并使整个实验 ERROR；预期停止仅登记将回收的独立宿主，不屏蔽共享 JVM 意外退出。Master 始终单独 spawn，不进入 EngineHost。

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
