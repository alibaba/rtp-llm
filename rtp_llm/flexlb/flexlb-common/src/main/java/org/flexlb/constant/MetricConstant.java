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

    public static final String ENGINE_STATUS_VISITOR_SUCCESS_QPS = "app.engine.health.check.visitor.success.qps";

    /**
     * Engine status check failure information
     */
    public static final String ENGINE_STATUS_CHECK_FAIL = "app.engine.health.check.fail";

    /** Cumulative engine-status check failure count for the process lifetime. */
    public static final String ENGINE_STATUS_CHECK_FAIL_TOTAL =
            "app.engine.health.check.fail.total";

    /** Failed WorkerStatus gRPC check latency in microseconds. */
    public static final String ENGINE_STATUS_CHECK_FAIL_RT =
            "app.engine.health.check.fail.rt";

    /**
     * Master load balancing service total QPS
     */
    public static final String ENGINE_BALANCING_MASTER_ALL_QPS = "app.engine.balancing.master.all.qps";

    public static final String ENGINE_BALANCING_MASTER_ALL_RT = "app.engine.balancing.master.all.rt";

    public static final String ENGINE_BALANCING_MASTER_SELECT_DETAIL = "app.engine.balancing.master.select.detail";

    public static final String ENGINE_BALANCING_MASTER_DISPATCH_REASON = "app.engine.balancing.master.dispatch.reason";

    /**
     * Batch dispatch size (number of requests per batch)
     */
    public static final String ENGINE_BALANCING_MASTER_BATCH_SIZE = "app.engine.balancing.master.batch.size";

    /**
     * Batch dispatch total token count per batch (sum of seqLen across picked items)
     */
    public static final String ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS =
            "app.engine.balancing.master.batch.total.tokens";

    /**
     * FlexLB scheduler inflight batch count per worker (number of dispatched-but-uncompleted batches).
     * <p>Unified metric for both prefill and decode workers, tagged by role and engineIp.
     */
    public static final String INFLIGHT_BATCH_COUNT = "app.flexlb.inflight.batch.count";

    /**
     * FlexLB scheduler inflight request count per worker (dispatched but not yet confirmed by engine).
     * <p>Unified metric for both prefill and decode workers, tagged by role and engineIp.
     * Replaces the former separate BATCH_INFLIGHT_REQUEST_COUNT (prefill) and DECODE_INFLIGHT_COUNT (decode).
     */
    public static final String INFLIGHT_REQUEST_COUNT = "app.flexlb.inflight.request.count";

    /**
     * FlexLB scheduler total load per decode worker (confirmed running + scheduler inflight)
     */
    public static final String DECODE_TOTAL_LOAD = "app.flexlb.decode.total.load";

    /**
     * FlexLB scheduler inflight KV cache reserved tokens per decode worker (local inflight reservation not yet confirmed by the engine)
     */
    public static final String DECODE_INFLIGHT_KV_RESERVED_TOKENS = "app.flexlb.decode.inflight.kv.reserved.tokens";

    /**
     * FlexLB scheduler inflight hard KV cache reserved tokens per decode worker (hard reservation that cannot be reclaimed)
     */
    public static final String DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS =
            "app.flexlb.decode.inflight.hard.kv.reserved.tokens";

    /**
     * FlexLB scheduler inflight max age (ms) — age of the oldest inflight entry, tagged by role and engineIp.
     */
    public static final String INFLIGHT_MAX_AGE_MS =
            "app.flexlb.inflight.max.age.ms";

    /**
     * FlexLB scheduler inflight TTL expired count — number of inflight requests
     * cleaned up by the TTL cleanup task. Reported as QPS, tagged by role.
     */
    public static final String INFLIGHT_TTL_EXPIRED_QPS = "app.flexlb.inflight.ttl.expired.qps";

    /**
     * Batch predicted execution time (formula estimate) in milliseconds
     */
    public static final String BATCH_PREDICTED_TIME_MS = "app.flexlb.batch.predicted.time.ms";

    /**
     * Batch actual execution time reported by the engine (NormalEngine execution, excludes queueing) in milliseconds
     */
    public static final String BATCH_ACTUAL_TIME_MS = "app.flexlb.batch.actual.time.ms";

    /**
     * Gap between actual and predicted batch execution time (actual minus predicted) in milliseconds;
     * positive means the prediction underestimated
     */
    public static final String BATCH_PREDICT_GAP_MS = "app.flexlb.batch.predict.gap.ms";

    /**
     * Dispatch-to-ACK time (from gRPC dispatch to engine EnqueueBatch acknowledgment) in milliseconds.
     * Reflects the latency of the engine accepting a batch into its queue.
     */
    public static final String DISPATCH_ACK_TIME_MS = "app.flexlb.dispatch.ack.time.ms";

    /**
     * Route+submit time (from schedule() entry to batcher offer completion) in milliseconds.
     * Measures the time spent in routing the request and enqueuing it into the per-engine batcher,
     * before the request enters the batch wait window.
     */
    public static final String ROUTE_SUBMIT_TIME_MS = "app.flexlb.route.submit.time.ms";

    /**
     * ACK-to-response time (from engine EnqueueBatch acknowledgment to schedule response sent
     * to the client) in milliseconds. Measures the latency between the engine ACKing the batch
     * and the Master sending the schedule response back to the caller.
     */
    public static final String ACK_TO_RESPONSE_TIME_MS = "app.flexlb.ack.to.response.time.ms";

    /**
     * Engine running queue time (from EP authoritative value)
     */
    public static final String ENGINE_RUNNING_QUEUE_TIME = "app.engine.health.check.running.queue.time";

    /** Local tasks known to FlexLB for a worker. */
    public static final String ENGINE_LOCAL_TASK_MAP_SIZE =
            "app.engine.health.check.local.task.map.size";

    /** Tasks dispatched by FlexLB but not yet confirmed by the worker. */
    public static final String ENGINE_IN_TRANSIT_TASK_SIZE =
            "app.engine.health.check.in.transit.task.size";

    /**
     * FlexLB scheduler inflight size — the scheduler's own inflight request count.
     * <p>Reported by BatchSchedulerReporter using role=PREFILL + engineIp="scheduler" tags.
     * Formerly kept as a separate name from the now-removed per-engine local inflight size metric
     * to avoid tag schema conflict (per-engine vs scheduler-level).
     */
    public static final String SCHEDULER_INFLIGHT_SIZE = "app.flexlb.scheduler.inflight.size";

    /**
     * FlexLB batcher queue size — number of pending (not-yet-batched) requests
     * in the per-engine WorkerBatcher queue.
     * <p>Reported by BatchSchedulerReporter with role and engineIp tags.
     * Independent metric name to avoid tag schema conflict with {@link #ROUTING_QUEUE_LENGTH}
     * (which uses type=batchQueue tag for backward compatibility).
     */
    public static final String BATCHER_QUEUE_SIZE = "app.flexlb.batcher.queue.size";

    /**
     * Engine finished task list size
     */
    public static final String ENGINE_FINISHED_TASK_LIST_SIZE = "app.engine.health.check.finished.task.list.size";

    /**
     * Engine running task info size
     */
    public static final String ENGINE_RUNNING_TASK_INFO_SIZE = "app.engine.health.check.running.task.info.size";

    public static final String ENGINE_WAITING_TASK_INFO_SIZE =
            "app.engine.health.check.waiting.task.info.size";

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

    /**
     * Engine worker info running query length variance
     */
    public static final String ENGINE_WORKER_INFO_RUNNING_QUERY_LEN_VAR = "app.engine.worker.info.running.query.len.var";

    public static final String ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_MASTER_DECISION_TO_WAITING_CONFIRM_MS =
            "app.engine.worker.status.observed.decision.to.waiting.ms";
    public static final String ENGINE_WORKER_STATUS_FLEXLB_OBSERVED_WAITING_TO_RUNNING_MS =
            "app.engine.worker.status.observed.waiting.to.running.ms";
    public static final String ENGINE_WORKER_STATUS_ENGINE_OBSERVED_WAITING_TO_RUNNING_MS =
            "app.engine.worker.status.engine.waiting.to.running.ms";
    public static final String ENGINE_WORKER_STATUS_ENGINE_OBSERVED_RECEIVED_TO_WAITING_MS =
            "app.engine.worker.status.engine.received.to.waiting.ms";
    public static final String ENGINE_WORKER_STATUS_INPUT_QUEUE_WAIT_MS =
            "app.engine.worker.status.input.queue.wait.ms";
    public static final String ENGINE_WORKER_STATUS_SCHEDULER_TO_RUNNING_MS =
            "app.engine.worker.status.scheduler.to.running.ms";
    public static final String ENGINE_WORKER_STATUS_SCHEDULER_WAIT_MS =
            "app.engine.worker.status.scheduler.wait.ms";
    public static final String ENGINE_WORKER_STATUS_REMOTE_KV_WAIT_MS =
            "app.engine.worker.status.remote.kv.wait.ms";
    public static final String ENGINE_WORKER_STATUS_RUNNING_TO_FIRST_TOKEN_MS =
            "app.engine.worker.status.running.to.first.token.ms";
    public static final String ENGINE_WORKER_STATUS_HBM_LOCAL_MATCH_TOKENS =
            "app.engine.worker.status.hbm.local.match.tokens";
    public static final String ENGINE_WORKER_STATUS_REMOTE_KV_ADDED_MATCH_TOKENS =
            "app.engine.worker.status.remote.kv.added.match.tokens";
    public static final String ENGINE_WORKER_STATUS_PREFILL_STEP_COUNT =
            "app.engine.worker.status.prefill.step.count";
    public static final String ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MIN =
            "app.engine.worker.status.prefill.nonfinal.chunk.tokens.min";
    public static final String ENGINE_WORKER_STATUS_PREFILL_NONFINAL_CHUNK_TOKENS_MAX =
            "app.engine.worker.status.prefill.nonfinal.chunk.tokens.max";

    /* ------------------------ Routing Observability -------------------------- */

    /** Failed KVCM query attempts after the configured retry budget is exhausted. */
    public static final String KVCM_QUERY_FAILURE_QPS =
            "app.cache.kvcm.query.failure.qps";

    /** KVCM retry attempts issued after an initial query failure. */
    public static final String KVCM_QUERY_RETRY_QPS =
            "app.cache.kvcm.query.retry.qps";

    /** Optimizer trace queries that could not be dispatched or returned an error. */
    public static final String OPTIMIZER_TRACE_QUERY_FAILED_QPS =
            "app.optimizer.trace.query.failed.qps";

    /** Optimizer trace queries intentionally skipped because prerequisites were unavailable. */
    public static final String OPTIMIZER_TRACE_QUERY_SKIPPED_QPS =
            "app.optimizer.trace.query.skipped.qps";

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

    public static final String CACHE_HIT_COMPARISON_PREDICTED_TOKENS =
            "app.cache.hit.comparison.predicted.tokens";
    public static final String CACHE_HIT_COMPARISON_ACTUAL_TOKENS =
            "app.cache.hit.comparison.actual.tokens";
    public static final String CACHE_HIT_COMPARISON_DELTA_TOKENS =
            "app.cache.hit.comparison.delta.tokens";
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_TOKENS =
            "app.cache.hit.comparison.local.standby.predicted.tokens";
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_DELTA_TOKENS =
            "app.cache.hit.comparison.local.standby.delta.tokens";
    public static final String CACHE_HIT_COMPARISON_PREDICTED_RATIO =
            "app.cache.hit.comparison.predicted.ratio";
    public static final String CACHE_HIT_COMPARISON_ACTUAL_RATIO =
            "app.cache.hit.comparison.actual.ratio";
    public static final String CACHE_HIT_COMPARISON_LOCAL_STANDBY_PREDICTED_RATIO =
            "app.cache.hit.comparison.local.standby.predicted.ratio";
    public static final String CACHE_KVCM_SELECTED_LOCAL_MATCH_TOKENS =
            "app.cache.kvcm.selected.local.match.tokens";
    public static final String CACHE_KVCM_SELECTED_P2P_FETCH_TOKENS =
            "app.cache.kvcm.selected.p2p.fetch.tokens";
    public static final String CACHE_KVCM_SELECTED_P2P_TOTAL_MATCH_TOKENS =
            "app.cache.kvcm.selected.p2p.total.match.tokens";
    public static final String CACHE_HIT_COMPARISON_KVCM_LOCAL_DELTA_TOKENS =
            "app.cache.hit.comparison.kvcm.local.delta.tokens";
    public static final String CACHE_HIT_COMPARISON_KVCM_P2P_TOTAL_MATCH_DELTA_TOKENS =
            "app.cache.hit.comparison.kvcm.p2p.total.match.delta.tokens";
    public static final String CACHE_LOCAL_STANDBY_CAPACITY_REJECTED_QPS =
            "app.cache.local.standby.capacity.rejected.qps";
    public static final String CACHE_LOCAL_STANDBY_MAPPING_COUNT =
            "app.cache.local.standby.mapping.count";
    public static final String CACHE_LOCAL_STANDBY_BLOCK_SIZE =
            "app.cache.local.standby.block.size";
    public static final String CACHE_MATCH_ACTIVE_SOURCE =
            "app.cache.match.active.source";
    public static final String CACHE_MATCH_SOURCE_CHANGE_QPS =
            "app.cache.match.source.change.qps";
    public static final String CACHE_MATCH_STANDBY_FALLBACK_QPS =
            "app.cache.match.standby.fallback.qps";

    /**
     * Recent cache-key hit token count for requests in the current metric bucket.
     */
    public static final String CACHE_RECENT_KEY_HIT_COUNT = "app.cache.recent.key.hit.count";

    /**
     * Recent cache-key input token count for requests in the current metric bucket.
     */
    public static final String CACHE_RECENT_KEY_TOTAL_COUNT = "app.cache.recent.key.total.count";

    /**
     * Aggregated theory cache-hit token count. Tagged by window=all.
     */
    public static final String CACHE_THEORY_HIT_COUNT = "app.cache.theory.hit.count";

    /**
     * Aggregated theory cache input-token count. Tagged by window=all.
     */
    public static final String CACHE_THEORY_TOTAL_COUNT = "app.cache.theory.total.count";

    /**
     * Aggregated theory cache-hit token ratio. Tagged by window=all.
     */
    public static final String CACHE_THEORY_HIT_RATIO = "app.cache.theory.hit.ratio";

    /**
     * Selected-worker routing cache-match hit tokens. Tagged by role.
     */
    public static final String CACHE_ROUTING_SELECTED_MATCH_HIT_TOKENS =
            "app.cache.routing.selected.match.hit.tokens";

    /**
     * Selected-worker routing cache-match input tokens. Tagged by role.
     */
    public static final String CACHE_ROUTING_SELECTED_MATCH_TOTAL_TOKENS =
            "app.cache.routing.selected.match.total.tokens";

    /**
     * Request-level maximum available-candidate cache-match hit tokens. Tagged by role.
     */
    public static final String CACHE_ROUTING_CANDIDATE_MAX_HIT_TOKENS =
            "app.cache.routing.candidate.max.hit.tokens";

    /**
     * Cache-affinity routing decisions. Tagged by role, engineIp, and decision.
     */
    public static final String CACHE_AFFINITY_DECISION = "app.cache.affinity.decision.qps";

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
     * Time spent waiting in the block hash executor queue, in microseconds.
     */
    public static final String BLOCK_HASH_QUEUE_WAIT_TIME_US = "app.block.hash.queue.wait.time.us";

    /**
     * Time spent calculating block hashes, in microseconds.
     */
    public static final String BLOCK_HASH_EXECUTION_TIME_US = "app.block.hash.execution.time.us";

    /**
     * Block hash request result, tagged by status.
     */
    public static final String BLOCK_HASH_RESULT = "app.block.hash.result";

    /**
     * Dedicated block hash thread pool status.
     */
    public static final String BLOCK_HASH_THREAD_POOL_INFO = "app.block.hash.thread.pool.info";

    /* ------------------------ gRPC Connection Pool Monitoring -------------------------- */

    /**
     * Connection count in gRPC connection pool
     */
    public static final String GRPC_CHANNEL_POOL_SIZE = "app.grpc.channel.pool.size";

    /**
     * gRPC call duration in milliseconds
     */
    public static final String GRPC_CALL_DURATION = "app.grpc.call.duration";

    /**
     * gRPC response body size in bytes
     */
    public static final String GRPC_RESPONSE_SIZE = "app.grpc.response.size";

    /**
     * gRPC call count
     */
    public static final String GRPC_CALL_COUNT = "app.grpc.call.count";

    /**
     * gRPC connection duration in microseconds
     */
    public static final String GRPC_CONNECTION_DURATION = "app.grpc.connection.duration";

    /* ------------------------ Request Queue Monitoring -------------------------- */

    /**
     * Current queue length
     */
    public static final String ROUTING_QUEUE_LENGTH = "app.routing.queue.length";

    /**
     * Wait time in milliseconds
     */
    public static final String ROUTING_QUEUE_WAIT_TIME_MS = "app.routing.queue.wait.time.ms";

    /**
     * Network transfer delay: time from client requestTimeMs to gRPC server entry, in milliseconds.
     * Reported as: grpcEntryTime - requestTimeMs
     */
    public static final String REQUEST_NETWORK_DELAY_MS = "app.request.network.delay.ms";

    /**
     * gRPC server processing time: from gRPC server entry to BalanceContext creation (startTime), in milliseconds.
     * Reported as: startTime - grpcEntryTime
     */
    public static final String GRPC_SERVER_PROCESS_MS = "app.grpc.server.process.ms";

    /** Number of input token IDs observed at the request boundary. */
    public static final String REQUEST_INPUT_IDS_COUNT = "app.request.input.ids.count";

    /** Request payload size from Content-Length when available. */
    public static final String REQUEST_BODY_BYTES = "app.request.body.bytes";

    public static final String LOCAL_STANDBY_HASH_QUEUE_WAIT_TIME_US =
            "app.local.standby.hash.queue.wait.time.us";
    public static final String LOCAL_STANDBY_HASH_EXECUTION_TIME_US =
            "app.local.standby.hash.execution.time.us";
    public static final String LOCAL_STANDBY_HASH_RESULT =
            "app.local.standby.hash.result";
    public static final String LOCAL_STANDBY_HASH_THREAD_POOL_INFO =
            "app.local.standby.hash.thread.pool.info";

    /**
     * Graceful online/offline lifecycle events
     */
    public static final String GRACEFUL_LIFECYCLE_EVENT = "app.graceful.lifecycle.event";

    /* ------------------------ Request Forwarding Monitoring -------------------------- */

    /**
     * Forward to master result QPS (status: success/failure)
     */
    public static final String FORWARD_TO_MASTER_RESULT = "app.forward.to.master.result";

    /* ------------------------ gRPC Server Executor Monitoring -------------------------- */

    /**
     * gRPC server executor active thread count (gauge)
     */
    public static final String GRPC_SERVER_EXECUTOR_ACTIVE_THREADS = "grpc.server.executor.active.threads";

    /**
     * gRPC server executor queue size (gauge)
     */
    public static final String GRPC_SERVER_EXECUTOR_QUEUE_SIZE = "grpc.server.executor.queue.size";

    /**
     * gRPC server executor current pool size (gauge)
     */
    public static final String GRPC_SERVER_EXECUTOR_POOL_SIZE = "grpc.server.executor.pool.size";

    /**
     * gRPC server executor maximum pool size (gauge)
     */
    public static final String GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE = "grpc.server.executor.max.pool.size";

    /**
     * gRPC server executor completed task count (counter — monotonically increasing)
     */
    public static final String GRPC_SERVER_EXECUTOR_COMPLETED_TASKS = "grpc.server.executor.completed.tasks";

    /**
     * gRPC server executor CallerRunsPolicy rejection count (counter — monotonically increasing)
     * <p>Note: name kept for backward compat after switching to AbortPolicy.
     */
    public static final String GRPC_SERVER_EXECUTOR_CALLER_RUNS = "grpc.server.executor.caller.runs";

    /* ------------------------ Dispatch Executor Monitoring ---------------------------- */

    /**
     * Dispatch executor active thread count (gauge)
     */
    public static final String DISPATCH_EXECUTOR_ACTIVE_THREADS = "dispatch.executor.active.threads";

    /**
     * Dispatch executor queue size (gauge)
     */
    public static final String DISPATCH_EXECUTOR_QUEUE_SIZE = "dispatch.executor.queue.size";

    /**
     * Dispatch executor current pool size (gauge)
     */
    public static final String DISPATCH_EXECUTOR_POOL_SIZE = "dispatch.executor.pool.size";

    /**
     * Dispatch executor completed task count (counter — monotonically increasing)
     */
    public static final String DISPATCH_EXECUTOR_COMPLETED_TASKS = "dispatch.executor.completed.tasks";

    /* ------------------------ Auto-TPM Priority Scheduler ---------------------------- */

    /**
     * Auto-TPM request count by priority (QPS), tags: priority
     */
    public static final String AUTO_TPM_REQUEST_COUNT = "auto_tpm.request.count";

    /**
     * Auto-TPM schedule latency in ms (timer), tags: priority, result
     */
    public static final String AUTO_TPM_SCHEDULE_LATENCY_MS = "auto_tpm.schedule.latency_ms";

    /**
     * Auto-TPM normal placement success count (QPS), tags: priority
     */
    public static final String AUTO_TPM_NORMAL_PLACEMENT_COUNT = "auto_tpm.normal_placement.count";

    /**
     * Auto-TPM eviction plan generation count (QPS), tags: priority, case, result
     */
    public static final String AUTO_TPM_EVICTION_PLAN_COUNT = "auto_tpm.eviction_plan.count";

    /**
     * Auto-TPM eviction plan commit count (QPS), tags: priority, case, result
     */
    public static final String AUTO_TPM_EVICTION_COMMIT_COUNT = "auto_tpm.eviction_commit.count";

    /**
     * Auto-TPM evicted victim count (QPS), tags: victim_priority, incoming_priority, stage, case
     */
    public static final String AUTO_TPM_VICTIM_COUNT = "auto_tpm.victim.count";

    /**
     * Auto-TPM optimistic plan commit conflict count (QPS), tags: case
     */
    public static final String AUTO_TPM_PLAN_CONFLICT_COUNT = "auto_tpm.plan_conflict.count";

    /**
     * Auto-TPM prefill batcher queue depth (gauge), tags: endpoint
     */
    public static final String AUTO_TPM_PREFILL_QUEUE_DEPTH = "auto_tpm.prefill.queue_depth";

    /**
     * Auto-TPM evicted victim KV tokens released (timer), tags: victim_priority, stage
     */
    public static final String AUTO_TPM_VICTIM_KV_TOKENS = "auto_tpm.victim.kv_tokens";

    /**
     * Auto-TPM decode reserved (shadow, engine-unconfirmed) request count (gauge), tags: endpoint
     */
    public static final String AUTO_TPM_DECODE_RESERVED_COUNT = "auto_tpm.decode.reserved.count";

    /**
     * Auto-TPM decode shadow hard KV reserved tokens (gauge), tags: endpoint
     */
    public static final String AUTO_TPM_DECODE_SHADOW_KV_RESERVED = "auto_tpm.decode.shadow_kv_reserved";

    /**
     * Auto-TPM TTFT approximation in ms (timer), tags: priority.
     * Approximated as request arrival → schedule completion on the Master;
     * true TTFT (first token on the engine) is not observable here (§19.2).
     */
    public static final String AUTO_TPM_TTFT_MS = "auto_tpm.ttft_ms";

    /**
     * Auto-TPM priority preemption count (QPS), tags: stage
     * (prefill_queued / decode_reserved).
     */
    public static final String AUTO_TPM_PRIORITY_PREEMPT_COUNT = "auto_tpm.priority_preempt.count";

    /**
     * Auto-TPM decode confirmed running request count (gauge), tags: endpoint.
     * Dashboard migration note on the value semantics:
     * <ul>
     *   <li>Phase 4 and earlier: equalled confirmedRunningCount, i.e. the
     *       accepted and running layers merged into one gauge.</li>
     *   <li>Phase 5+: counts ONLY engine-reported {@code RUNNING} tasks; the
     *       accepted layer moved to {@link #AUTO_TPM_DECODE_ACCEPTED_COUNT}.
     *       The former merged value = running.count + accepted.count.</li>
     * </ul>
     */
    public static final String AUTO_TPM_DECODE_RUNNING_COUNT = "auto_tpm.decode.running.count";

    /**
     * Auto-TPM decode accepted-not-running (engine KV-allocated) request
     * count (gauge), tags: endpoint. Introduced by the Phase 5 layered view:
     * before Phase 5 this layer was folded into
     * {@link #AUTO_TPM_DECODE_RUNNING_COUNT}; from Phase 5 on, the former
     * merged value = running.count + accepted.count (dashboard migration).
     */
    public static final String AUTO_TPM_DECODE_ACCEPTED_COUNT = "auto_tpm.decode.accepted.count";

    /**
     * Auto-TPM engine cancel requests issued (QPS), tags: endpoint,
     * priority (victim's normalized priority).
     * Counted when an accepted-eviction commit fires a Cancel RPC.
     */
    public static final String AUTO_TPM_CANCEL_REQUEST_COUNT = "auto_tpm.cancel.request.count";

    /**
     * Auto-TPM engine cancel release confirmations (QPS), tags: endpoint,
     * priority (victim's normalized priority; "0" when unavailable).
     * Counted when a cancelled victim's release is confirmed — either inside
     * the commit wait window or later via the WorkerStatus settle path.
     */
    public static final String AUTO_TPM_CANCEL_CONFIRM_COUNT = "auto_tpm.cancel.confirm.count";

    /**
     * Auto-TPM engine cancel wait-window timeouts (QPS), tags: endpoint,
     * priority (the incoming request whose eviction plan failed).
     * The plan failed; the victim stays CANCEL_REQUESTED until WorkerStatus
     * settles it.
     */
    public static final String AUTO_TPM_CANCEL_TIMEOUT_COUNT = "auto_tpm.cancel.timeout.count";

    /**
     * Auto-TPM cancel initiations (QPS), tags: priority (normalized priority
     * of the cancelled request, raw 1-100 value — matches the rest of the
     * auto_tpm family; "0" = priority unavailable), reason
     * (PRIORITY_PREEMPTED / USER_CANCELLED / DEADLINE_EXCEEDED). Counted
     * once per cancel intent injected into the EngineCancelChannel: the
     * preemption path and the Frontend-initiated cancel path.
     */
    public static final String AUTO_TPM_CANCEL_QPS = "auto_tpm.cancel.qps";

    /**
     * Auto-TPM plan age — snapshot/plan build to successful commit, ms
     * (redesign N3 §3.8 placement_plan_age_ms), tags: priority.
     * Quantifies how stale a lockfree commit's plan view was.
     */
    public static final String AUTO_TPM_PLAN_AGE_MS = "auto_tpm.plan_age_ms";

    /**
     * Auto-TPM decode engine-facing load (gauge): confirmed + dispatched
     * reservations, excluding queued-phase shadow entries (redesign N2/§3.8
     * decode_shadow_load vs decode_engine_running), tags: endpoint. Compare
     * against {@link #AUTO_TPM_DECODE_RESERVED_COUNT} to monitor root cause C.
     */
    public static final String AUTO_TPM_DECODE_ENGINE_LOAD = "auto_tpm.decode.engine_load";

    /**
     * Auto-TPM inflight settle misses (QPS): a finishYielded/PreemptedById
     * found no inflight entry (review P2-2), tags: kind (yielded/preempted).
     * Harmless in isolation, but a burst points at a registration/cleanup
     * race — alert-worthy where a warn log is not.
     */
    public static final String AUTO_TPM_INFLIGHT_SETTLE_MISS = "auto_tpm.inflight_settle_miss.count";
}
