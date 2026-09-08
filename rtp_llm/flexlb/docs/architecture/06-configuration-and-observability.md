# Configuration and Observability

## 配置边界

FlexLB 使用两份职责独立的配置：

- FLEXLB_CONFIG：负载均衡行为，包括 scheduler、dispatcher、路由、worker registry、cache
  matching、observability、服务发现运行参数、optimizer 和 consistency。
- MODEL_SERVICE_CONFIG：模型服务拓扑，包括 service_id、按 group 的 role_endpoints，以及 KVCM /
  Optimizer 的地址和 discovery 定位。

MODEL_SERVICE_CONFIG 在进程启动时解析为 ServiceRoute；动态 FLEXLB_CONFIG 更新不改变模型拓扑。
相反，MODEL_SERVICE_CONFIG 不接受调度、cache matching、服务发现 timeout/poll 或主选举等行为字段。

当前公共行为配置的 schemaVersion 是 2。ConfigService 严格拒绝未知字段、null、重复 key、类型
coercion 和不符合 active tagged-union 分支的字段。版本 0 的历史合并配置仍可由 V0 parser
归一化；版本 1 到版本 2 的迁移由 FlexlbConfigMigration 显式完成，不应把 v1 作为运行时标准
文档发布。

## 配置来源与更新

进程启动时按以下优先级选择一个行为配置来源：

| 条件 | 来源 | 更新方式 |
|---|---|---|
| FLEXLB_UNICONF_ENABLE=true | UniConfigConfigSource | 本机 Turbo UniConfig 轮询 |
| 否则 FLEXLB_NACOS_SERVER_ADDR 非空 | NacosConfigSource | Nacos listener |
| 否则 | EnvironmentConfigSource | 启动时读取 FLEXLB_CONFIG |

不论行为来源为何，EnvironmentConfigSource 都读取 MODEL_SERVICE_CONFIG。来源内容经相应 parser
归一化后由 FlexlbConfigMerger 与默认配置深度合并：对象字段递归覆盖，数组/标量替换；带 type
的对象只有 type 相同才递归合并，切换 type 则整体替换。结果必须通过 schema 与跨字段校验，才会
原子替换当前 FlexlbConfig；非法运行时更新保留 last-known-good 快照。

组件是否热生效取决于其读取方式。RouteService 和选择器从请求绑定的快照读取行为；在 Bean
构造时缓存的线程池规模、同步间隔、delivery strategy、ZooKeeper 客户端和模型拓扑需要重启
才能改变。

### UniConfig 与 Nacos

UniConfig 要求 Spectrum deployment identity（SPECTRUM_WORKSPACE_ID、
SPECTRUM_APPLICATION_NAME、SPECTRUM_DEPLOYMENT_NAME）和
FLEXLB_UNICONF_ENABLE=true。它请求：

    GET http://127.0.0.1:18080/v2/configs/modelstudio.spectrum.deployment.<workspace>.<deployment>.runtime.meta

连接和读取 timeout 都是 3 秒。启动阶段如果本机 agent 尚未就绪，会以 1 秒间隔持续重试；
成功后每 30 秒 fixed-delay 轮询，内容变化才发布。运行时读取失败保留当前快照。

Nacos 的可选环境变量为 FLEXLB_NACOS_SERVER_ADDR、FLEXLB_NACOS_DATA_ID、
FLEXLB_NACOS_GROUP 和 FLEXLB_NACOS_NAMESPACE。未显式设置 data id 时，使用
DeploymentIdentity 的 deployment id。

## FLEXLB_CONFIG 结构

顶层字段及其所有权如下：

| 字段 | 责任 |
|---|---|
| scheduler | DIRECT/QUEUE、QUEUE timeout、排序、decision、全局容量和 lifecycle |
| dispatcher | BATCH/NON_BATCH 的交付类型、per-Prefill in-flight 上限和 enqueue timeout |
| router | Prefill estimator/candidate/cache affinity、Decode KV/load 策略和 group selector |
| workerRegistry | worker status/cache status 的轮询、timeout、陈旧与 cache interval 参数 |
| cacheMatching | LOCAL_SYNC 或 KVCM；KVCM 分支包含 health、retry、P2P 与 Local Standby 参数 |
| observability | cache-hit recent key window 与 FlexLB logger 配置 |
| serviceDiscovery | discovery client 的连接、读取、轮询和 keep-alive 运行参数 |
| optimizer | Optimizer 开关与 discovery poll interval |
| consistency | NONE 或 ZOOKEEPER |
| blockHashStrategy | VLLM 或 SGLANG |
| enableFallback | 当前由 RouteService.isFallbackEnabled() 暴露给调用方；本检出中的 Schedule 路径不据此短路请求 |
| fallbackBatchTokenCapacity | 引擎没有声明 batch/sequence 容量时的 reservation 兜底 |

DIRECT 要求 dispatcher.type=NON_BATCH。QUEUE 的 ordering.type=FIFO 不允许 preemption；
PRIORITY 的 engine-owned Decode preemption 必须同时配置 engineCancellation。decision.type=SINGLE
不允许 fixed-window 字段；FIXED_WINDOW 才使用 maxRequests、maxCollectionWaitMs 和
maxPredictedExecutionMs。

内部线程池参数位于 InternalRuntimeSettings，不能通过公共 JSON 覆盖。它包括全局规划线程数、
batch dispatch completion 线程数与 worker/status 同步线程数等实现参数。

## 端口与控制面

application.yml 的默认端口如下：

| 服务 | 默认端口 |
|---|---|
| WebFlux HTTP | 7001 |
| Spring management / actuator | 7002 |
| FlexLB gRPC | HTTP 端口 + 2，即默认 7003 |

后端 engine 的 HTTP/gRPC 转换规则与 FlexLB 自身 gRPC 端口无关。多 engine endpoint 的
worker_status_port 是 worker-control gRPC 基址；N>1 时 index i 使用 base+i，且配置加载时检查
端口范围。frontend HTTP/gRPC 地址仍是共享物理地址。

主要 HTTP 端点：

- /health；/hook/process_ok、/hook/after_start、/hook/pre_stop；
- /flexlb/update_log_level、/flexlb/cache_match/status、/flexlb/cache_match/failover；
- /rtp_llm/master/info、/rtp_llm/schedule_snapshot、/rtp_llm/notify_master；
- /rtp_llm/queue_snapshot、/rtp_llm/inflight_status；
- /rtp_llm/server_latency、/rtp_llm/server_latency/reset。

Schedule、GetRequestState 和 Cancel 是 FlexLB gRPC 服务的方法，不是 HTTP schedule 路由。

## 日志与指标

FLEXLB_LOG_PATH、FLEXLB_APP_LOG_PATH 和 FLEXLB_LOG_ASYNC_QUEUE_SIZE 控制 logback 文件输出。
observability.logging.level 更新 flexlb logger group，stdoutEnabled 动态控制 root/PV console
appender；/flexlb/update_log_level 提供显式运行时调整。

FlexMonitor 是统一的指标抽象。默认 provider 是 noop；启用的 provider 负责 engine health、worker
poll、routing、queue、delivery、preemption、forward-to-master、cache/KVCM/standby、block hash、
优雅生命周期和请求延迟等指标。FlexlbGrpcServer 还直接向可用的 MeterRegistry 注册其 executor
active threads、队列长度、pool size、completed tasks 和 rejection counter。
