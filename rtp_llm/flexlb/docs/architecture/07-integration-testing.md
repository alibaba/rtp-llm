# Integration Testing Architecture

`flexlb-integration-test` 是 FlexLB 的单引擎、transport-real 集成测试模块。它验证真实
Spring Boot/WebFlux 应用和生产路由链路在引擎状态、缓存匹配、队列与外部 RPC 不稳定时的
行为；它不是生产运行时的一部分，也不替代各模块的单元测试。

本文描述当前测试架构与回归边界。运行方式和新增用例约定见
[模块 README](../../flexlb-integration-test/README.md)。生产请求链路、排队和缓存实现分别见
[01-routing-and-balancing](01-routing-and-balancing.md)、
[02-queue-scheduling](02-queue-scheduling.md) 和
[04-worker-sync-and-cache](04-worker-sync-and-cache.md)。

## 1. 目标与范围

测试模块的目标是让以下路径同时处于真实实现中：HTTP 接入、请求 block-key 准备、
`RouteService`、队列调度、worker 状态同步、cache provider、策略选择、回滚和 fallback。
只有跨进程的引擎与 KVCM 边界可被脚本化，从而既能覆盖实际协议，也能稳定地制造故障。

当前范围是：

- 单个物理引擎实例内的角色路由；不引入 multi-engine logical identity，也不模拟逻辑
  engine index。
- `SHORTEST_TTFT` 与 `CACHE_AFFINITY_FIRST`；`RANDOM` 与 `WEIGHTED_CACHE` 不纳入当前
  回归门禁。
- vLLM、SGLang、RTP-LLM 三种 cache-key 合约，以及 KVCM、`LOCAL_SYNC`、Local Standby
  三条缓存匹配路径。
- 正常、降级、错误和有限规模并发稳定性。高压测试验证正确性与活性，不提供机器无关的
  QPS 性能承诺。

队列满、请求取消、master 转发、worker 重启版本抖动和跨引擎逻辑身份属于后续独立增量，
不能借由当前单引擎 fake 隐式宣称已经覆盖。

## 2. 测试拓扑与边界

每个 Failsafe fork 启动一个随机 HTTP 端口的完整 Spring 上下文，并拥有一组仅供该 fork
使用的 loopback gRPC fake。应用内部 bean 不被 Mockito 替换。

```text
HTTP client
    |
    v
POST /rtp_llm/schedule
    |
    v
real Spring Boot / WebFlux application
    |
    +--> real RequestBlockHashService / RouteService / QueueManager
    |        / DefaultRouter / strategy / cache orchestration
    |
    +--> real gRPC clients -----------------------------------+
                                                              |
                         loopback fake boundary               v
                    +----------------------------------+  GetWorkerStatus
                    | Scripted worker-status service   |  GetCacheStatus
                    | KVCM metadata RPC adapter        |  KVCM lookup/meta RPC
                    +----------------------------------+
```

fake 只负责返回可配置的协议响应、注入延迟或 gRPC error，并记录线上客户端实际发出的请求。
断言既检查最终 HTTP 调度结果，也检查该场景的可观测副作用，例如被选 endpoint、KVCM wire
keys、worker status 快照、队列排空或 RPC 调用次数。这样不会因只 mock 一个内部接口而跳过
同步、hash 或 fallback 编排。

## 3. 模块与测试支撑层

测试代码按业务能力隔离，而不是全部放在一个 `.it` 包中：

```text
org.flexlb.it
├── scenario
│   ├── strategy.shortestttft
│   ├── strategy.cacheaffinity
│   ├── queue
│   ├── cache.kvcm
│   ├── cache.rtpllm
│   ├── cache.standby
│   ├── fallback
│   ├── worker
│   └── stability
├── configuration
└── fixture
    ├── engine
    ├── kvcm
    └── spring
```

`IntegrationTestFixtures` 仅维护跨场景共用的拓扑、worker 脚本与可观测状态；
`KvcmIntegrationTestFixtures` 仅维护 KVCM discovery、查询脚本和 wire 观测。各场景的
strategy、队列或缓存配置放在本场景的 context initializer 中，避免公共 fixture 演变成
全部测试配置的汇集点。

`WorkerTopology` 以 `RoleType -> count` 声明 worker 数量；worker 的稳定标识为
`(role, index)`。这允许同一场景声明 `PREFILL=2`、`DECODE=3`、`PDFUSION=1` 等任意角色
组合，而不是通过 `WORKER` / `SECOND_WORKER` 这类固定槽位表达拓扑。

每个测试上下文在启动前注册其 fake 地址和静态发现配置，启动后等待生产同步器写入
`WorkerStatus`。fork 级隔离和显式 reset 防止静态 worker map、fake 调用记录或本地任务跨
用例污染。

## 4. 引擎与 cache-key 合约

请求是否自带 `block_cache_keys` 是引擎协议的差异，不能以统一 hash mock 掩盖：

| 引擎模式 | 输入与 key 来源 | 匹配路径 | 集成测试证明的事实 |
| --- | --- | --- | --- |
| vLLM | 从请求 `input_ids` 用生产 vLLM 策略计算 | KVCM | 真实 hash 结果被发送到 KVCM wire。 |
| SGLang | 从请求 `input_ids` 用生产 SGLang 策略计算 | KVCM | SGLang 的 block hash 和实际 KVCM 查询 key 一致。 |
| RTP-LLM | 调用方在请求中提供 `block_cache_keys` | `LOCAL_SYNC` | FlexLB 保留调用方 key；真实 `GetCacheStatus` 回填本地索引后据此路由。 |

因此，RTP-LLM 场景不验证不存在的 hash strategy；vLLM/SGLang 场景也不通过手工塞入
request keys 伪造 KVCM 命中。三类场景都通过生产 cache orchestration 进入候选决策。

## 5. 策略、队列与 fallback 回归矩阵

| 能力 | 代表场景 | 关键断言 |
| --- | --- | --- |
| 基础调度 | `SingleEngineSchedulingIT` | 只选健康 worker；全 down 返回受控错误；短请求避开 1M 上下文、16K/32K chunk 的长 prefill。 |
| 相似度阈值 | `ShortestTtftSimilarityRatioIT` | 以 `0.2`、`0.5`、`0.8` 验证 cache 候选是否可进入相近 TTFT 集合及最终选择。 |
| 短桶排队 | `ShortBucketQueueIT` | 多个短请求在真实 `QueueManager`/`RequestScheduler` 中形成确定的 waiting 队列；恢复后全部完成并排空。 |
| Cache affinity guard | `CacheAffinityFirstIT` | 验证 max extra work、`outstandingUncachedTokensThreshold`、`cacheAffinityFirstOutstandingUncachedTokensThreshold` 和 `cacheAffinityFirstMinHitRate` 的拒绝/保留边界。 |
| KVCM failover | `CacheAffinityFirstIT`、`LocalStandbyCapacityIT` | 有效空结果仍走 KVCM；连续查询失败达到阈值才进入 Local Standby。 |
| Local Standby 容量 | `LocalStandbyCapacityIT` | 达到容量后不驱逐未过期 mapping；高水位清理只删除 TTL 已过期项，释放容量后才接纳新 mapping。 |
| 全局 fallback | `FallbackGateIT` | `enableFallback` 直接返回 `FALLBACK`，不开始调度也不请求 worker。 |

`SHORTEST_TTFT` 的长桶用例强调单个请求跨多个 step 的状态演进；短桶用例强调多个请求在
很短 waiting 队列内聚合并被 scheduler 推进。两类负载共同避免只用单请求 happy path
推断生产决策正确性。

## 6. worker status 韧性与规模回归

worker fake 实现真实 `GetWorkerStatus` 和 `GetCacheStatus` RPC，可按 worker、调用次数和
角色脚本化响应。当前稳定性覆盖：

- 一个 worker 持续返回 gRPC `UNAVAILABLE` 时，请求仍可选择健康 worker；预期的 fake
  RPC error 不被误判为测试失败。
- 同步周期为 50ms 时，某 worker 每第三次响应延迟 200ms；并发调度持续完成，且多轮延迟后
  状态仍保持健康。
- 动态角色拓扑被同步到对应的 role map，而非只验证单一默认角色。
- `stress-it` profile 启动 200 个 PDFUSION worker-status endpoint，等待生产状态快照全部
  就绪后，以并发度 32 完成 400 个 HTTP schedule。

规模用例为可选 Maven profile，避免把开发机或 CI 的瞬时资源差异纳入普通 PR 的时延阈值；
它的失败语义是请求无法完成、worker 状态无法收敛或调度出现错误，而不是低于某个固定 QPS。

## 7. 异步断言与时间约束

所有异步状态转换直接使用 test-scope 的 Awaitility 4.2.2，不保留自定义 `Eventually`
封装。每个等待点都必须显式声明：

- `alias`，使超时信息对应具体业务状态；
- 零 poll delay；
- 10ms poll interval；
- 与场景风险匹配的 `atMost` 超时。

禁止用固定 `sleep` 作为同步正确性的依据，也不设置 Awaitility 全局默认值。测试应等待
HTTP 完成、fake RPC 调用、队列大小或同步后的 `WorkerStatus` 等业务可观测条件，而不是
猜测后台线程何时执行。这使偶发 200ms worker-status 延迟能够被稳定复现和诊断，同时保持
普通场景的快速反馈。

## 8. 执行模型与质量门禁

`*IT` 由 Maven Failsafe 在 `integration-test` / `verify` phase 执行，普通单测的
`test` phase 只负责编译该模块。常用命令：

```bash
./mvnw -pl flexlb-integration-test -am verify
./mvnw -pl flexlb-integration-test -am verify -Dit.test=CacheAffinityFirstIT
./mvnw -pl flexlb-integration-test -am verify -Pstress-it -Dit.test=WorkerScaleLoadIT
```

运行第一条命令或完整的 `./mvnw verify` 后，`flexlb-integration-test` 在其 `verify` phase
把跨模块 JaCoCo HTML、XML 和 CSV 写入 root 的 `target/site/jacoco-aggregate/`。root POM 统一
定义输出路径；集成测试模块位于其生产依赖之后，因而此时可以同时收集生产模块 class/source
与 unit/IT execution data，而不会把测试类计入被覆盖的生产代码。该模块为聚合目的显式声明
生产模块的 optional compile 依赖，避免这些依赖向其空测试产物的消费者传播。

新增场景必须使用 `should_xxx_when_xxx` 方法命名，在相关 feature package 下放置最小化的
initializer，并同时覆盖成功决策和至少一个失败、降级或边界分支。若新增生产配置字段，
需要在 initializer 中显式设定，不能依赖环境默认值；若新增外部 RPC，则应该先扩展 fake
协议边界，再保留内部生产 bean 的真实调用路径。

完整回归前还应执行：

```bash
./mvnw spotless:check -Pspotless-check
git diff --check
```

## 9. 保障边界

该模块保障的是 FlexLB 在受控、真实 transport 条件下的功能和活性回归：输入如何变成 cache
key、状态如何进入候选集、策略 guard 如何拒绝不合格 cache leader、故障如何退化，以及队列
和同步如何最终收敛。它不证明真实集群网络、KVCM 服务本身、模型引擎的算法实现或跨逻辑
engine 路由的可用性；这些需要部署级观测、端到端流量或各自组件的测试补充。
