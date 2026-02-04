package org.flexlb.constant;

/**
 * Metric constants for monitoring and observability
 * Use with standard monitoring libraries like Micrometer/Prometheus
 */
public class MetricConstant {

    /* ------------------------ 引擎状态统计 -------------------------- */

    /**
     * 引擎状态检查更新周期
     */
    public static final String ENGINE_STATUS_CHECK_SUCCESS_PERIOD = "app.engine.health.check.success.period";

    /**
     * 引擎worker数量
     */
    public static final String ENGINE_WORKER_NUMBER = "app.engine.health.check.engine.worker.number";

    public static final String ENGINE_PREFILL_WORKER_NUMBER = "app.engine.health.check.engine.prefill.worker.number";

    public static final String ENGINE_DECODE_WORKER_NUMBER = "app.engine.health.check.engine.decode.worker.number";

    /**
     * 服务发现 Client 请求结果
     */
    public static final String ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT = "app.engine.health.check.engine.worker.number.service.discovery.result";

    /**
     * 引擎worker剩余可用并发数
     */
    public static final String ENGINE_STATUS_AVAILABLE_CONCURRENCY = "app.engine.health.check.available.concurrency";

    public static final String ENGINE_STATUS_VISITOR_RT = "app.engine.health.check.visitor.rt";

    public static final String ENGINE_STATUS_VISITOR_SUCCESS_QPS = "app.engine.health.check.visitor.qps";

    /**
     * 引擎状态检查失败信息
     */
    public static final String ENGINE_STATUS_CHECK_FAIL = "app.engine.health.check.fail";

    /**
     * Master负载均衡服务的总qps
     */
    public static final String ENGINE_BALANCING_MASTER_ALL_QPS = "app.engine.balancing.master.all.qps";

    public static final String ENGINE_BALANCING_MASTER_SCHEDULE_RT = "app.engine.balancing.master.all.rt";

    public static final String ENGINE_BALANCING_MASTER_SELECT_DETAIL = "app.engine.balancing.master.select.detail";

    /**
     * 引擎队列等待时间
     */
    public static final String ENGINE_RUNNING_QUEUE_TIME = "app.engine.health.check.running.queue.time";

    /**
     * 引擎本地任务缓存大小
     */
    public static final String ENGINE_LOCAL_TASK_MAP_SIZE = "app.engine.health.check.local.task.map.size";

    /**
     * 引擎已完成任务列表大小
     */
    public static final String ENGINE_FINISHED_TASK_LIST_SIZE = "app.engine.health.check.finished.task.list.size";

    /**
     * 引擎正在运行任务信息大小
     */
    public static final String ENGINE_RUNNING_TASK_INFO_SIZE = "app.engine.health.check.running.task.info.size";

    /**
     * prefill master节点监控
     */
    public static final String ZK_MASTER_NODE = "app.engine.zk.master.node";

    /**
     * prefill master节点事件监控
     */
    public static final String ZK_MASTER_EVENT = "app.engine.zk.master.event";

    /**
     * 负载均衡服务的线程池状态
     */
    public static final String ENGINE_BALANCING_THREAD_POOL_INFO = "app.engine.balancing.thread.pool.info";

    /**
     * 负载均衡服务的 NioEvenLoopGroup 状态
     */
    public static final String ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO = "app.engine.balancing.event.loop.group.info";

    /**
     * 获取引擎worker信息服务的step latency 方差
     */
    public static final String ENGINE_WORKER_INFO_STEP_LATENCY_VAR = "app.engine.worker.info.step.latency.var";

    public static final String ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR = "app.engine.worker.info.running.query.len.var";

    /* ------------------------ 缓存健康监控 -------------------------- */

    /**
     * 每个引擎局部缓存的数量
     */
    public static final String CACHE_ENGINE_LOCAL_COUNT = "app.cache.engine.local.count";

    /**
     * 全局缓存的总数量
     */
    public static final String CACHE_GLOBAL_TOTAL_COUNT = "app.cache.global.total.count";

    /**
     * 局部缓存占用的字节数
     */
    public static final String CACHE_ENGINE_LOCAL_BYTES = "app.cache.engine.local.bytes";

    /**
     * 全局缓存占用的字节数
     */
    public static final String CACHE_GLOBAL_BYTES = "app.cache.global.bytes";

    /**
     * 缓存命中数量
     */
    public static final String CACHE_HIT_COUNT = "app.cache.hit.count";

    /**
     * 缓存命中百分比
     */
    public static final String CACHE_HIT_RATIO = "app.cache.hit.ratio";

    /**
     * 缓存请求总数
     */
    public static final String CACHE_REQUEST_TOTAL = "app.cache.request.total";
    
    /**
     * 查找匹配引擎的响应时间
     */
    public static final String CACHE_FIND_MATCHING_ENGINES_RT = "app.cache.find.matching.engines.rt";
    
    /**
     * 更新缓存的响应时间
     */
    public static final String CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT = "app.cache.update.engine.block.cache.rt";
    
    /**
     * 缓存状态检查响应时间
     */
    public static final String CACHE_STATUS_CHECK_VISITOR_RT = "app.cache.status.check.visitor.rt";

    public static final String CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS = "app.cache.status.check.visitor.success.qps";
    
    /**
     * 缓存状态检查成功周期
     */
    public static final String CACHE_STATUS_CHECK_SUCCESS_PERIOD = "app.cache.status.check.success.period";
    
    /**
     * 缓存状态检查失败信息
     */
    public static final String CACHE_STATUS_CHECK_FAIL = "app.cache.status.check.fail";
    
    /**
     * 缓存块大小
     */
    public static final String CACHE_BLOCK_SIZE = "app.cache.block.size";
    
    /**
     * 缓存键大小
     */
    public static final String CACHE_KEY_SIZE = "app.cache.key.size";
    
    /**
     * 已使用的KV缓存Token数量
     */
    public static final String CACHE_USED_KV_CACHE_TOKENS = "app.cache.used.kv.cache.tokens";
    
    /**
     * 剩余可用的KV缓存Token数量
     */
    public static final String CACHE_AVAILABLE_KV_CACHE_TOKENS = "app.cache.available.kv.cache.tokens";
    
    /**
     * KV缓存Token总量
     */
    public static final String CACHE_TOTAL_KV_CACHE_TOKENS = "app.cache.total.kv.cache.tokens";

    /**
     * KV缓存已使用比例（已使用Token / 总Token）
     */
    public static final String CACHE_USED_KV_CACHE_RATIO = "app.cache.used.kv.cache.ratio";
    
    /**
     * 缓存diff计算中新增块的数量
     */
    public static final String CACHE_DIFF_ADDED_BLOCKS_SIZE = "app.cache.diff.added.blocks.size";
    
    /**
     * 缓存diff计算中移除块的数量
     */
    public static final String CACHE_DIFF_REMOVED_BLOCKS_SIZE = "app.cache.diff.removed.blocks.size";
    
    /**
     * 引擎视图Map的大小（即当前有多少个引擎）
     */
    public static final String CACHE_ENGINE_VIEWS_MAP_SIZE = "app.cache.engine.views.map.size";
    
    /* ------------------------ gRPC连接池监控 -------------------------- */
    
    /**
     * gRPC连接池中的连接数量
     */
    public static final String GRPC_CHANNEL_POOL_SIZE = "app.grpc.channel.pool.size";

    /* ------------------------ 请求排队监控 -------------------------- */

    /**
     * 当前队列长度
     */
    public static final String ROUTING_QUEUE_LENGTH = "app.routing.queue.length";

    /**
     * 入队列 QPS
     */
    public static final String ROUTING_QUEUE_ENTRY_QPS = "app.routing.queue.entry.qps";

    /**
     * 超时 QPS
     */
    public static final String ROUTING_QUEUE_TIMEOUT_QPS = "app.routing.queue.timeout.qps";

    /**
     * 队列满拒绝 QPS
     */
    public static final String ROUTING_QUEUE_REJECTED_QPS = "app.routing.queue.rejected.qps";

    /**
     * 取消 QPS
     */
    public static final String ROUTING_QUEUE_CANCELLED_QPS = "app.routing.queue.cancelled.qps";

    /**
     * 等待时间（毫秒）
     */
    public static final String ROUTING_QUEUE_WAIT_TIME_MS = "app.routing.queue.wait.time.ms";

    /**
     * 路由执行时间（毫秒）
     */
    public static final String ROUTING_ROUTE_EXECUTION_TIME_MS = "app.routing.route.execution.time.ms";

    /**
     * 路由成功 QPS
     */
    public static final String ROUTING_SUCCESS_QPS = "app.routing.success.qps";

    /**
     * 路由失败 QPS
     */
    public static final String ROUTING_FAILURE_QPS = "app.routing.failure.qps";

    /**
     * 路由重试 QPS
     */
    public static final String ROUTING_RETRY_QPS = "app.routing.retry.qps";

    /* ------------------------ 资源监控 -------------------------- */

    /**
     * Worker 许可容量
     */
    public static final String WORKER_PERMIT_CAPACITY = "app.worker.permit.capacity";

    /**
     * 请求到达 Netty 的延迟 (客户端 requestTimeSeconds 到服务端 startTime 的差值，毫秒)
     */
    public static final String REQUEST_ARRIVAL_DELAY_MS = "app.request.arrival.delay.ms";

    /**
     * 优雅上下线生命周期事件
     */
    public static final String LIFECYCLE_EVENT_METRIC = "graceful.lifecycle.event";

    /* ------------------------ 请求转发监控 -------------------------- */

    /**
     * 转发到Master的结果QPS (status: success/failure)
     */
    public static final String FORWARD_TO_MASTER_RESULT = "app.forward.to.master.result";
}
