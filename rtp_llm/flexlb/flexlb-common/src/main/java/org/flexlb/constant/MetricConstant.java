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
     * Engine tasks sent by FlexLB but not yet confirmed by the worker.
     */
    public static final String ENGINE_IN_TRANSIT_TASK_SIZE = "app.engine.health.check.in.transit.task.size";

    /**
     * Engine finished task list size
     */
    public static final String ENGINE_FINISHED_TASK_LIST_SIZE = "app.engine.health.check.finished.task.list.size";

    /**
     * Engine running task info size
     */
    public static final String ENGINE_RUNNING_TASK_INFO_SIZE = "app.engine.health.check.running.task.info.size";

    /**
     * Engine waiting task info size
     */
    public static final String ENGINE_WAITING_TASK_INFO_SIZE = "app.engine.health.check.waiting.task.info.size";

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
     * Load balancing service EventLoopGroup status
     */
    public static final String ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO = "app.engine.balancing.event.loop.group.info";

    /**
     * Engine worker info service step latency variance
     */
    public static final String ENGINE_WORKER_INFO_STEP_LATENCY_VAR = "app.engine.worker.info.step.latency.var";

    public static final String ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR = "app.engine.worker.info.running.query.len.var";

    /**
     * Elapsed time from the master recording a local task to the first worker waiting confirmation.
     */
    public static final String ENGINE_WORKER_STATUS_MASTER_DECISION_TO_WAITING_CONFIRM_MS =
            "app.engine.worker.status.master.decision.to.waiting.confirm.ms";

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
     * Predicted cache-hit tokens from scheduling, compared with actual engine cache hits.
     */
    public static final String CACHE_HIT_COMPARISON_PREDICTED_TOKENS = "app.cache.hit.comparison.predicted.tokens";

    /**
     * Actual cache-hit tokens reported by the engine, compared with scheduling prediction.
     */
    public static final String CACHE_HIT_COMPARISON_ACTUAL_TOKENS = "app.cache.hit.comparison.actual.tokens";

    /**
     * Difference between actual and predicted cache-hit tokens.
     */
    public static final String CACHE_HIT_COMPARISON_DELTA_TOKENS = "app.cache.hit.comparison.delta.tokens";

    /**
     * Cache-hit tokens predicted by the local standby matcher.
     */
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS = "app.cache.hit.comparison.local.standby.predicted.tokens";

    /**
     * Difference between actual and local standby predicted cache-hit tokens.
     */
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS = "app.cache.hit.comparison.local.standby.delta.tokens";

    /**
     * Scheduling cache-hit ratio for requests with an eventual engine result.
     */
    public static final String CACHE_HIT_COMPARISON_PREDICTED_RATIO = "app.cache.hit.comparison.predicted.ratio";

    /**
     * Actual engine cache-hit ratio for requests with an eventual engine result.
     */
    public static final String CACHE_HIT_COMPARISON_ACTUAL_RATIO = "app.cache.hit.comparison.actual.ratio";

    /**
     * Local Standby cache-hit ratio for requests with an eventual engine result.
     */
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO = "app.cache.hit.comparison.local.standby.predicted.ratio";

    /**
     * Local Standby mappings rejected because the configured capacity has been reached.
     */
    public static final String CACHE_LOCAL_STANDBY_CAPACITY_REJECTED_QPS = "app.cache.local.standby.capacity.rejected.qps";

    /**
     * Current number of Local Standby block-worker mappings.
     */
    public static final String CACHE_LOCAL_STANDBY_MAPPING_COUNT = "app.cache.local.standby.mapping.count";

    /**
     * Current cache matching source. The active source is reported as 1 and the inactive source as 0.
     */
    public static final String CACHE_MATCH_ACTIVE_SOURCE = "app.cache.match.active.source";

    /**
     * Cache matching source transitions.
     */
    public static final String CACHE_MATCH_SOURCE_CHANGE_QPS = "app.cache.match.source.change.qps";

    /**
     * Requests routed through Local Standby because KVCM is unavailable or already in fallback.
     */
    public static final String CACHE_MATCH_STANDBY_FALLBACK_QPS = "app.cache.match.standby.fallback.qps";

    /**
     * KVCM cache query retry attempts.
     */
    public static final String KVCM_QUERY_RETRY_QPS = "app.cache.kvcm.query.retry.qps";

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

    /* ------------------------ Block Hash Monitoring -------------------------- */

    /**
     * Time spent waiting in the block hash executor queue, in microseconds
     */
    public static final String BLOCK_HASH_QUEUE_WAIT_TIME_US = "app.block.hash.queue.wait.time.us";

    /**
     * Time spent calculating block hashes, in microseconds
     */
    public static final String BLOCK_HASH_EXECUTION_TIME_US = "app.block.hash.execution.time.us";

    /**
     * Block hash request result, tagged by status
     */
    public static final String BLOCK_HASH_RESULT = "app.block.hash.result";

    /**
     * Dedicated block hash thread pool status
     */
    public static final String BLOCK_HASH_THREAD_POOL_INFO = "app.block.hash.thread.pool.info";

    public static final String LOCAL_STANDBY_HASH_QUEUE_WAIT_TIME_US = "app.local.standby.hash.queue.wait.time.us";

    public static final String LOCAL_STANDBY_HASH_EXECUTION_TIME_US = "app.local.standby.hash.execution.time.us";

    public static final String LOCAL_STANDBY_HASH_RESULT = "app.local.standby.hash.result";

    public static final String LOCAL_STANDBY_HASH_THREAD_POOL_INFO = "app.local.standby.hash.thread.pool.info";

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
     * Per-engine resource water level used by FlexLB capacity control, in the range 0-100.
     * Tagged by engineIp and role. Prefill/PDFusion uses queue water level; Decode uses
     * KV-cache water level.
     */
    public static final String WORKER_RESOURCE_WATER_LEVEL = "app.worker.resource.water.level";

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
}
