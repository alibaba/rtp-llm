package org.flexlb.constant;

/**
 * Metric constants for monitoring and observability
 * Use with standard monitoring libraries like Micrometer/Prometheus
 */
public class MetricConstant {

    /* ------------------------ Engine Status Metrics -------------------------- */

    /**
     * Engine status check success period
     */
    public static final String ENGINE_STATUS_CHECK_SUCCESS_PERIOD = "app.engine.health.check.success.period";

    /**
     * Engine worker count
     */
    public static final String ENGINE_WORKER_NUMBER = "app.engine.health.check.engine.worker.number";

    public static final String ENGINE_PREFILL_WORKER_NUMBER = "app.engine.health.check.engine.prefill.worker.number";

    public static final String ENGINE_DECODE_WORKER_NUMBER = "app.engine.health.check.engine.decode.worker.number";

    /**
     * Service discovery client request result
     */
    public static final String ENGINE_NUMBER_SERVICE_DISCOVERY_RESULT = "app.engine.health.check.engine.worker.number.service.discovery.result";

    /**
     * Engine worker remaining available concurrency
     */
    public static final String ENGINE_STATUS_AVAILABLE_CONCURRENCY = "app.engine.health.check.available.concurrency";

    public static final String ENGINE_STATUS_VISITOR_RT = "app.engine.health.check.visitor.rt";

    public static final String ENGINE_STATUS_VISITOR_SUCCESS_QPS = "app.engine.health.check.visitor.qps";

    /**
     * Engine status check failure information
     */
    public static final String ENGINE_STATUS_CHECK_FAIL = "app.engine.health.check.fail";

    /**
     * Master load balancing service total QPS
     */
    public static final String ENGINE_BALANCING_MASTER_ALL_QPS = "app.engine.balancing.master.all.qps";

    public static final String ENGINE_BALANCING_MASTER_SCHEDULE_RT = "app.engine.balancing.master.all.rt";

    public static final String ENGINE_BALANCING_MASTER_SELECT_DETAIL = "app.engine.balancing.master.select.detail";

    /**
     * Engine queue wait time
     */
    public static final String ENGINE_RUNNING_QUEUE_TIME = "app.engine.health.check.running.queue.time";

    /**
     * Engine local task map size
     */
    public static final String ENGINE_LOCAL_TASK_MAP_SIZE = "app.engine.health.check.local.task.map.size";

    /**
     * Engine finished task list size
     */
    public static final String ENGINE_FINISHED_TASK_LIST_SIZE = "app.engine.health.check.finished.task.list.size";

    /**
     * Engine running task info size
     */
    public static final String ENGINE_RUNNING_TASK_INFO_SIZE = "app.engine.health.check.running.task.info.size";

    /**
     * Prefill master node monitoring
     */
    public static final String ZK_MASTER_NODE = "app.engine.zk.master.node";

    /**
     * Prefill master node event monitoring
     */
    public static final String ZK_MASTER_EVENT = "app.engine.zk.master.event";

    /**
     * Load balancing service thread pool status
     */
    public static final String ENGINE_BALANCING_THREAD_POOL_INFO = "app.engine.balancing.thread.pool.info";

    /**
     * Load balancing service NioEventLoopGroup status
     */
    public static final String ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO = "app.engine.balancing.event.loop.group.info";

    /**
     * Engine worker info service step latency variance
     */
    public static final String ENGINE_WORKER_INFO_STEP_LATENCY_VAR = "app.engine.worker.info.step.latency.var";

    public static final String ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR = "app.engine.worker.info.running.query.len.var";

    /* ------------------------ Cache Health Monitoring -------------------------- */

    /**
     * Per-engine local cache count
     */
    public static final String CACHE_ENGINE_LOCAL_COUNT = "app.cache.engine.local.count";

    /**
     * Global cache total count
     */
    public static final String CACHE_GLOBAL_TOTAL_COUNT = "app.cache.global.total.count";

    /**
     * Local cache bytes occupied
     */
    public static final String CACHE_ENGINE_LOCAL_BYTES = "app.cache.engine.local.bytes";

    /**
     * Global cache bytes occupied
     */
    public static final String CACHE_GLOBAL_BYTES = "app.cache.global.bytes";

    /**
     * Cache hit count
     */
    public static final String CACHE_HIT_COUNT = "app.cache.hit.count";

    /**
     * Cache hit percentage
     */
    public static final String CACHE_HIT_RATIO = "app.cache.hit.ratio";

    /**
     * Recent cache-key hit count for requests in the current metric bucket.
     */
    public static final String CACHE_RECENT_KEY_HIT_COUNT = "app.cache.recent.key.hit.count";

    /**
     * Recent cache-key total count for requests in the current metric bucket.
     */
    public static final String CACHE_RECENT_KEY_TOTAL_COUNT = "app.cache.recent.key.total.count";

    /**
     * Recent cache-key hit ratio for the current request against the sliding time window.
     */
    public static final String CACHE_RECENT_KEY_HIT_RATIO = "app.cache.recent.key.hit.ratio";

    /**
     * Requests observed by recent cache-key hit metrics.
     */
    public static final String CACHE_RECENT_KEY_REQUEST_COUNT = "app.cache.recent.key.request.count";

    /**
     * Requests observed by recent cache-key hit metrics without cache keys.
     */
    public static final String CACHE_RECENT_KEY_EMPTY_REQUEST_COUNT = "app.cache.recent.key.empty_request.count";

    /**
     * Cache request total count
     */
    public static final String CACHE_REQUEST_TOTAL = "app.cache.request.total";

    /**
     * Find matching engines response time
     */
    public static final String CACHE_FIND_MATCHING_ENGINES_RT = "app.cache.find.matching.engines.rt";

    /**
     * Update cache response time
     */
    public static final String CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT = "app.cache.update.engine.block.cache.rt";

    /**
     * Cache status check response time
     */
    public static final String CACHE_STATUS_CHECK_VISITOR_RT = "app.cache.status.check.visitor.rt";

    public static final String CACHE_STATUS_CHECK_VISITOR_SUCCESS_QPS = "app.cache.status.check.visitor.success.qps";

    /**
     * Cache status check success period
     */
    public static final String CACHE_STATUS_CHECK_SUCCESS_PERIOD = "app.cache.status.check.success.period";

    /**
     * Cache status check failure information
     */
    public static final String CACHE_STATUS_CHECK_FAIL = "app.cache.status.check.fail";

    /**
     * Cache block size
     */
    public static final String CACHE_BLOCK_SIZE = "app.cache.block.size";

    /**
     * Cache key size
     */
    public static final String CACHE_KEY_SIZE = "app.cache.key.size";

    /**
     * Used KV cache token count
     */
    public static final String CACHE_USED_KV_CACHE_TOKENS = "app.cache.used.kv.cache.tokens";

    /**
     * Remaining available KV cache token count
     */
    public static final String CACHE_AVAILABLE_KV_CACHE_TOKENS = "app.cache.available.kv.cache.tokens";

    /**
     * Total KV cache token count
     */
    public static final String CACHE_TOTAL_KV_CACHE_TOKENS = "app.cache.total.kv.cache.tokens";

    /**
     * KV cache usage percentage (used tokens / total tokens)
     */
    public static final String CACHE_USED_KV_CACHE_RATIO = "app.cache.used.kv.cache.ratio";

    /**
     * Added blocks count in cache diff calculation
     */
    public static final String CACHE_DIFF_ADDED_BLOCKS_SIZE = "app.cache.diff.added.blocks.size";

    /**
     * Removed blocks count in cache diff calculation
     */
    public static final String CACHE_DIFF_REMOVED_BLOCKS_SIZE = "app.cache.diff.removed.blocks.size";

    /**
     * Engine view map size (current number of engines)
     */
    public static final String CACHE_ENGINE_VIEWS_MAP_SIZE = "app.cache.engine.views.map.size";

    /* ------------------------ gRPC Connection Pool Monitoring -------------------------- */

    /**
     * Connection count in gRPC connection pool
     */
    public static final String GRPC_CHANNEL_POOL_SIZE = "app.grpc.channel.pool.size";

    /* ------------------------ Request Queue Monitoring -------------------------- */

    /**
     * Current queue length
     */
    public static final String ROUTING_QUEUE_LENGTH = "app.routing.queue.length";

    /**
     * Queue entry QPS
     */
    public static final String ROUTING_QUEUE_ENTRY_QPS = "app.routing.queue.entry.qps";

    /**
     * Timeout QPS
     */
    public static final String ROUTING_QUEUE_TIMEOUT_QPS = "app.routing.queue.timeout.qps";

    /**
     * Queue full rejection QPS
     */
    public static final String ROUTING_QUEUE_REJECTED_QPS = "app.routing.queue.rejected.qps";

    /**
     * Cancellation QPS
     */
    public static final String ROUTING_QUEUE_CANCELLED_QPS = "app.routing.queue.cancelled.qps";

    /**
     * Wait time in milliseconds
     */
    public static final String ROUTING_QUEUE_WAIT_TIME_MS = "app.routing.queue.wait.time.ms";

    /**
     * Routing execution time in milliseconds
     */
    public static final String ROUTING_ROUTE_EXECUTION_TIME_MS = "app.routing.route.execution.time.ms";

    /**
     * Routing success QPS
     */
    public static final String ROUTING_SUCCESS_QPS = "app.routing.success.qps";

    /**
     * Routing failure QPS
     */
    public static final String ROUTING_FAILURE_QPS = "app.routing.failure.qps";

    /**
     * Routing retry QPS
     */
    public static final String ROUTING_RETRY_QPS = "app.routing.retry.qps";

    /* ------------------------ Resource Monitoring -------------------------- */

    /**
     * Worker permit capacity
     */
    public static final String WORKER_PERMIT_CAPACITY = "app.worker.permit.capacity";

    /**
     * Request arrival delay at Netty (difference between client requestTimeSeconds and server startTime, in milliseconds)
     */
    public static final String REQUEST_ARRIVAL_DELAY_MS = "app.request.arrival.delay.ms";

    /**
     * Graceful online/offline lifecycle events
     */
    public static final String LIFECYCLE_EVENT_METRIC = "graceful.lifecycle.event";

    /* ------------------------ Request Forwarding Monitoring -------------------------- */

    /**
     * Forward to master result QPS (status: success/failure)
     */
    public static final String FORWARD_TO_MASTER_RESULT = "app.forward.to.master.result";

    /* ------------------------ V1 DP Batch Monitoring -------------------------- */

    /**
     * V1 DP batch flush QPS, tagged by trigger reason
     * (BUCKET_FULL / PER_REQUEST_TIMEOUT / DEADLINE / WINDOW_TIMER)
     */
    public static final String V1_DP_BATCH_FLUSH_QPS = "app.v1.dp.batch.flush.qps";

    /**
     * V1 DP batch real-request size at flush (excludes fake-pad slots), tagged by flush reason
     */
    public static final String V1_DP_BATCH_SIZE = "app.v1.dp.batch.size";

    /**
     * V1 DP rank hit QPS, tagged by rank — measures load distribution across dp_ranks
     */
    public static final String V1_DP_RANK_HIT_QPS = "app.v1.dp.rank.hit.qps";

    /**
     * V1 fake-pad slot emission QPS (sum of fake slots injected to fill a DP barrier)
     */
    public static final String V1_DP_FAKE_PAD_SLOT_QPS = "app.v1.dp.fake.pad.slot.qps";

    /**
     * V1 fake-pad slot count per batch (gauge), tagged by dpSize
     */
    public static final String V1_DP_FAKE_PAD_COUNT_PER_BATCH = "app.v1.dp.fake.pad.count.per.batch";

    /**
     * V1 InflightBatchRegistry batch count gauge
     */
    public static final String V1_DP_INFLIGHT_BATCH_COUNT = "app.v1.dp.inflight.batch.count";

    /**
     * V1 InflightBatchRegistry request count gauge
     */
    public static final String V1_DP_INFLIGHT_REQUEST_COUNT = "app.v1.dp.inflight.request.count";

    /**
     * V1 InflightBatchRegistry safety-net eviction count (cumulative — should be 0 in steady state)
     */
    public static final String V1_DP_INFLIGHT_EVICTED_COUNT = "app.v1.dp.inflight.evicted.count";

    /**
     * V1 active per-model batcher count
     */
    public static final String V1_DP_BATCHER_COUNT = "app.v1.dp.batcher.count";

    /**
     * V1 total queued request depth across all per-model batchers
     */
    public static final String V1_DP_QUEUE_DEPTH = "app.v1.dp.queue.depth";

    /**
     * V1 per-request wait time (ms) spent in the per-model batcher queue
     * before being drained into a batch, tagged by flush reason. Reported
     * once per request at dispatch time so KMonitor can derive p50/p99.
     */
    public static final String V1_DP_BATCH_WAIT_TIME_MS = "app.v1.dp.batch.wait.time.ms";

    /* ------------------------ SLO Violation Monitoring -------------------------- */

    /**
     * Routing SLO violation QPS — incremented when a batch is dispatched past
     * its SLO deadline (head request's deadline already expired). Distinct from
     * {@link #ROUTING_FAILURE_QPS} so SLO pressure can be alerted on without
     * being drowned by worker-reject failures.
     */
    public static final String ROUTING_SLO_VIOLATION_QPS = "app.routing.slo.violation.qps";

    /* ------------------------ V1 DP SloBudgetBatcher Monitoring -------------------------- */
    /*
     * 这些指标专属于 SloBudgetBatcher（dpSize=1 单 DP 单 FIFO + SLO 预算驱动凑批）。
     * 所有标签均使用 model + role + group + endpoint 组合，不直接用 raw IP。
     */

    /**
     * 算法结果 - SloBudgetBatcher 一次 dispatch 中分配给某个 DP rank 的请求数量。
     * 标签: model + role + group + endpoint + dp_rank。dpSize=1 时 dp_rank 恒为 0。
     */
    public static final String V1_DP_SLO_BATCH_DP_REQ_COUNT = "app.v1.dp.slo.batch.dp.req.count";

    /**
     * 算法结果 - SloBudgetBatcher 一次 dispatch 中分配给某个 DP rank 的 token 总数。
     * 标签: model + role + group + endpoint + dp_rank。
     */
    public static final String V1_DP_SLO_BATCH_DP_TOKENS = "app.v1.dp.slo.batch.dp.tokens";

    /**
     * 算法结果 - SloBudgetBatcher 目标 batch token 上限 (batchMaxTokens 配置值)。
     * 标签: model。用来对比实际 actual tokens 看打满率。
     */
    public static final String V1_DP_SLO_BATCH_TARGET_TOKENS = "app.v1.dp.slo.batch.target.tokens";

    /**
     * 算法结果 - SloBudgetBatcher 实际打包的 batch token 总数。
     * 标签: model + reason。和 TARGET_TOKENS 比可知 batch 利用率。
     */
    public static final String V1_DP_SLO_BATCH_ACTUAL_TOKENS = "app.v1.dp.slo.batch.actual.tokens";

    /**
     * 算法状态 - SloBudgetBatcher 队列中等待中的请求数量 (含 head)。
     * 标签: model。每次 loop 迭代上报。
     */
    public static final String V1_DP_SLO_QUEUE_REQUESTS = "app.v1.dp.slo.queue.requests";

    /**
     * 算法状态 - SloBudgetBatcher 队列中等待中的请求 token 总数。
     * 标签: model。配合 QUEUE_REQUESTS 可知队列平均请求大小。
     */
    public static final String V1_DP_SLO_QUEUE_TOKENS = "app.v1.dp.slo.queue.tokens";

    /**
     * 算法状态 - SloBudgetBatcher 单条请求实际排队时间 (ms)。
     * 标签: model + reason。PRECISE 类型用于 p50/p95/p99 分位数。
     */
    public static final String V1_DP_SLO_QUEUE_WAIT_MS = "app.v1.dp.slo.queue.wait.ms";

    /**
     * 算法状态 - SloBudgetBatcher 请求失败 QPS。
     * 标签: model + reason (SLO_EXCEEDED / PLANNER_ERROR / DISPATCH_ERROR)。
     */
    public static final String V1_DP_SLO_FAILURE_QPS = "app.v1.dp.slo.failure.qps";

    /**
     * 算法性能 - SloBudgetBatcher 主循环单次 stepOnce 耗时 (微秒)。
     * 标签: model + outcome (dispatch / fail / park)。PRECISE 类型。
     */
    public static final String V1_DP_SLO_LOOP_DURATION_US = "app.v1.dp.slo.loop.duration.us";

    /**
     * 算法性能 - SloBudgetBatcher 每次成功 dispatch 之前经历的 loop 次数 (含本次)。
     * 标签: model + reason。值大表示 SLO 宽松 / park 频繁。
     */
    public static final String V1_DP_SLO_LOOPS_PER_DISPATCH = "app.v1.dp.slo.loops.per.dispatch";

    public static final String V1_DP_PREFILL_ACTUAL_TIME_US = "app.v1.dp.prefill.actual.time.us";

    public static final String V1_DP_PREFILL_PREDICTION_ERROR_MS = "app.v1.dp.prefill.prediction.error.ms";

    public static final String V1_DP_PREFILL_BATCH_INPUT_TOKENS = "app.v1.dp.prefill.batch.input.tokens";

    public static final String V1_DP_PREFILL_BATCH_SIZE = "app.v1.dp.prefill.batch.size";
}
