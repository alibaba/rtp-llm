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
     * FlexLB decode engine-accepted tasks currently in the WAITING phase
     * ({@code DecodeEndpoint#decodeEngineWaitingCount}). Decode role only.
     */
    public static final String DECODE_ENGINE_WAITING_COUNT = "app.flexlb.decode.engine.waiting.count";

    /**
     * FlexLB decode engine-accepted tasks currently in the LOADING phase (remote KV loading)
     * ({@code DecodeEndpoint#decodeEngineLoadingCount}). Decode role only.
     */
    public static final String DECODE_ENGINE_LOADING_COUNT = "app.flexlb.decode.engine.loading.count";

    /**
     * FlexLB decode engine-accepted tasks currently in the RUNNING phase
     * ({@code DecodeEndpoint#decodeEngineRunningCount}). Decode role only.
     */
    public static final String DECODE_ENGINE_RUNNING_COUNT = "app.flexlb.decode.engine.running.count";

    /**
     * FlexLB prefill unconfirmed entry count — dispatched, not yet acknowledged by
     * the engine ({@code PrefillEndpoint#prefillActiveRequestCount} minus
     * {@code PrefillEndpoint#prefillEngineOwnedCount}). Prefill/PDFusion roles only.
     */
    public static final String PREFILL_INFLIGHT_ENTRIES_COUNT = "app.flexlb.prefill.inflight.entries.count";

    /**
     * FlexLB prefill engine-owned task count, observed from engine reports
     * ({@code PrefillEndpoint#prefillEngineOwnedCount}). Prefill/PDFusion roles only.
     */
    public static final String PREFILL_ENGINE_WORK_COUNT = "app.flexlb.prefill.engine.work.count";

    /**
     * FlexLB decode unconfirmed reservation count — reserved locally, not yet accepted
     * by the engine ({@code DecodeEndpoint#decodeInflightCount}). Decode role only.
     */
    public static final String DECODE_INFLIGHT_REQUESTS_COUNT = "app.flexlb.decode.inflight.requests.count";

    /**
     * FlexLB decode engine-owned task count
     * ({@code DecodeEndpoint#decodeEngineWorkCount}). Decode role only.
     */
    public static final String DECODE_ENGINE_WORK_COUNT = "app.flexlb.decode.engine.work.count";

    /**
     * FlexLB scheduler inflight KV cache reserved tokens per decode worker (local inflight reservation not yet confirmed by the engine)
     */
    public static final String DECODE_INFLIGHT_KV_RESERVED_TOKENS = "app.flexlb.decode.inflight.kv.reserved.tokens";

    /**
     * FlexLB scheduler inflight hard KV cache reserved tokens per decode worker (layer-1 Σ kvTokens,
     * seqLen-only hard demand used for hard-capacity filtering). The non-suffixed
     * {@code app.flexlb.decode.inflight.kv.reserved.tokens} carries the expected (seqLen + maxNewTokens) view.
     */
    public static final String DECODE_INFLIGHT_KV_RESERVED_HARD_TOKENS =
            "app.flexlb.decode.inflight.kv.reserved.hard.tokens";

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
     * Prefill estimated queue wait time in milliseconds (EP authoritative,
     * {@code PrefillEndpoint#prefillEstimatedWaitTimeMs}). Prefill/PDFusion roles only.
     */
    public static final String ENGINE_PREFILL_WAIT_TIME_MS = "app.engine.health.check.prefill.wait.time.ms";

    /**
     * Prefill engine-accepted tasks currently in the WAITING phase
     * ({@code PrefillEndpoint#prefillEngineWaitingCount}). Prefill/PDFusion roles only.
     */
    public static final String ENGINE_PREFILL_WAITING_COUNT = "app.engine.health.check.prefill.engine.waiting.count";

    /**
     * Prefill engine-accepted tasks currently in the RUNNING phase
     * ({@code PrefillEndpoint#prefillEngineRunningCount}). Prefill/PDFusion roles only.
     */
    public static final String ENGINE_PREFILL_RUNNING_COUNT = "app.engine.health.check.prefill.engine.running.count";

    /**
     * Active task count (EP authoritative). Decode: confirmed running + inflight
     * ({@code DecodeEndpoint#decodeTotalLoad}); other roles: engine-reported running
     * task count ({@code SimpleWorkerEndpoint#runningTaskCount}). Differentiated by role tag.
     */
    public static final String ENGINE_ACTIVE_TASK_COUNT = "app.engine.health.check.active.task.count";

    // ========== FlexLB state ledger metrics ==========

    /** Shadow event-pump counter: one report per engine observation fed into the shadow StateLedger. */
    public static final String SHADOW_EVENT = "app.flexlb.shadow.event";

    /** Shadow pipeline error counter (catch-all Throwable inside shadow calls; never affects the main path). */
    public static final String SHADOW_ERROR = "app.flexlb.shadow.error";

    /** Terminal-state diff: old-path terminal state and shadow-ledger terminal state disagree for the same requestId. */
    public static final String SHADOW_DIFF_TERMINAL_STATE = "app.flexlb.shadow.diff.terminal.state";

    /** Terminal-reason diff: shadow TerminalReason is outside the equivalent-reason set of the old terminal state. */
    public static final String SHADOW_DIFF_TERMINAL_REASON = "app.flexlb.shadow.diff.terminal.reason";

    /** Old path reached a terminal state but the shadow ledger never did (within the diff window). */
    public static final String SHADOW_DIFF_TERMINAL_MISSING_ON_NEW = "app.flexlb.shadow.diff.terminal.missing.on.new";

    /** Shadow ledger reached a terminal state but the old path never did (within the diff window). */
    public static final String SHADOW_DIFF_TERMINAL_MISSING_ON_OLD = "app.flexlb.shadow.diff.terminal.missing.on.old";

    // ========== FlexLB state ledger observability metrics ==========
    // Reporter: LedgerMetricsReporter (flexlb-sync), 5s sampling tick.
    // Two-level aggregation (cardinality containment): global untagged metrics below
    // plus tagged phase/endpoint/reason series — full endpoint x phase matrix is
    // deliberately NOT published.

    /** P-side active entry count (global level; GAUGE). */
    public static final String STATE_LEDGER_PREFILL_ACTIVE = "app.flexlb.state.ledger.prefill.active";

    /** D-side active entry count (global level; GAUGE). */
    public static final String STATE_LEDGER_DECODE_ACTIVE = "app.flexlb.state.ledger.decode.active";

    /** P-side tombstone watermark — retained terminal entries (GAUGE). */
    public static final String STATE_LEDGER_PREFILL_TOMBSTONES = "app.flexlb.state.ledger.prefill.tombstones";

    /** D-side tombstone watermark — retained terminal entries (GAUGE). */
    public static final String STATE_LEDGER_DECODE_TOMBSTONES = "app.flexlb.state.ledger.decode.tombstones";

    /** Late engine events absorbed by tombstones after settlement (cumulative QPS). */
    public static final String STATE_LEDGER_LATE_EVENTS = "app.flexlb.state.ledger.late.events";

    /** Late local cancels absorbed by tombstones after settlement (cumulative QPS). */
    public static final String STATE_LEDGER_LATE_CANCELS = "app.flexlb.state.ledger.late.cancels";

    /** Unknown-target engine events (running + finished, cumulative QPS) — ledger coverage gap signal. */
    public static final String STATE_LEDGER_UNKNOWN_EVENTS = "app.flexlb.state.ledger.unknown.events";

    /** Fast-path settle wins (engine finished / local settle / cancel propagation — the "normal channel wins" leg of the overtake tri-split, cumulative QPS). */
    public static final String STATE_LEDGER_FASTPATH_SETTLES = "app.flexlb.state.ledger.fastpath.settles";

    /** Phase-advance CAS losers (the "overtaken" leg of the overtake tri-split, cumulative QPS). */
    public static final String STATE_LEDGER_OVERTAKEN = "app.flexlb.state.ledger.overtaken";

    /** Counter drift entries from the last auditAndDrift reconciliation (GAUGE; 0 = clean). */
    public static final String STATE_LEDGER_DRIFT_ENTRIES = "app.flexlb.state.ledger.drift.entries";

    /** Janitor evidence-channel (F2) absence-presumed-death settlements (cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_VANISHED_SETTLES = "app.flexlb.state.ledger.janitor.vanished.settles";

    /** Janitor time-channel (F3) TTL settlements (cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_TTL_SETTLES = "app.flexlb.state.ledger.janitor.ttl.settles";

    /** Janitor force-channel (F4) hard-cap settlements (cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_HARDCAP_SETTLES = "app.flexlb.state.ledger.janitor.hardcap.settles";

    /** Janitor settle attempts lost to the fast path (the "fallback channel lost" leg of the overtake tri-split, cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_LOST_TO_FASTPATH = "app.flexlb.state.ledger.janitor.lost.to.fastpath";

    /** Janitor fence-hold skips (guard-rail 3 exemptions, cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_FENCE_HOLDS = "app.flexlb.state.ledger.janitor.fence.holds";

    /** Janitor internal error counter (catch-all; cumulative QPS). */
    public static final String STATE_LEDGER_JANITOR_ERRORS = "app.flexlb.state.ledger.janitor.errors";

    /** Tracked endpoint count in the ledger endpoint index (GAUGE). */
    public static final String STATE_LEDGER_ENDPOINT_COUNT = "app.flexlb.state.ledger.endpoint.count";

    /** Per-endpoint active-entry distribution P50 (P+D combined pool aggregate, GAUGE). */
    public static final String STATE_LEDGER_ENDPOINT_ACTIVE_P50 = "app.flexlb.state.ledger.endpoint.active.p50";

    /** Per-endpoint active-entry distribution P95 (P+D combined pool aggregate, GAUGE). */
    public static final String STATE_LEDGER_ENDPOINT_ACTIVE_P95 = "app.flexlb.state.ledger.endpoint.active.p95";

    /** Per-endpoint active-entry distribution max (P+D combined pool aggregate, GAUGE). */
    public static final String STATE_LEDGER_ENDPOINT_ACTIVE_MAX = "app.flexlb.state.ledger.endpoint.active.max";

    /**
     * Phase population histogram (global level; GAUGE; tags: side=P|D, phase).
     * Endpoint x phase matrix is intentionally NOT published (cardinality).
     */
    public static final String STATE_LEDGER_PHASE_POPULATION = "app.flexlb.state.ledger.phase.population";

    /**
     * Phase residence age quantile (GAUGE; tags: side, phase, quantile=p50|p95).
     * Bucket-upper-bound estimate from the cumulative sampling distribution
     * (conservatively high); right-censored in-residence age.
     */
    public static final String STATE_LEDGER_PHASE_AGE_MS = "app.flexlb.state.ledger.phase.age.ms";

    /**
     * Controlled reason counter (cumulative QPS; tags: kind=settle|cleanup|transition|terminal, reason).
     * Full enum coverage per kind — dead-enum prevention is asserted by tests.
     */
    public static final String STATE_LEDGER_REASON_COUNT = "app.flexlb.state.ledger.reason.count";

    /**
     * Per-endpoint active entry count (endpoint-fine level; GAUGE; tags: side=P|D, endpoint).
     * Series count = tracked endpoints (same order as the engine health-check metrics).
     */
    public static final String STATE_LEDGER_ENDPOINT_ACTIVE = "app.flexlb.state.ledger.endpoint.active";

    /**
     * FlexLB batcher queue size — number of pending (not-yet-batched) requests
     * in the per-engine WorkerBatcher queue.
     * <p>Reported by BatchSchedulerReporter with role and engineIp tags.
     * Independent metric name to avoid tag schema conflict with {@link #ROUTING_QUEUE_LENGTH}
     * (which uses type=batchQueue tag for backward compatibility).
     */
    public static final String BATCHER_QUEUE_SIZE = "app.flexlb.batcher.queue.size";

    /**
     * FlexLB batch queue wait time (enqueue to dispatch) in milliseconds — batch path only.
     * <p>Reported by BatchSchedulerReporter as a TIMER with role and engineIp tags.
     * Independent metric name to avoid type/semantic conflict with
     * {@link #ROUTING_QUEUE_WAIT_TIME_MS} (non-batch routing queue wait, GAUGE,
     * registered by RoutingQueueReporter).
     */
    public static final String BATCH_QUEUE_WAIT_TIME_MS = "app.flexlb.batch.queue.wait.time.ms";

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

    /**
     * Engine worker load variance across workers of the same role.
     * Input is the role-specific endpoint load — Prefill: estimated wait time (ms);
     * Decode: active task count; other roles: running task count — so the unit
     * differs per role tag and values are only comparable within one role.
     */
    public static final String ENGINE_WORKER_INFO_LOAD_VAR = "app.engine.worker.info.load.var";

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
     * Network transfer delay: time from client requestTimeMs to gRPC server entry, in milliseconds.
     * Reported as: grpcEntryTime - requestTimeMs
     */
    public static final String REQUEST_NETWORK_DELAY_MS = "app.request.network.delay.ms";

    /**
     * gRPC server processing time: from gRPC server entry to BalanceContext creation (startTime), in milliseconds.
     * Reported as: startTime - grpcEntryTime
     */
    public static final String GRPC_SERVER_PROCESS_MS = "app.grpc.server.process.ms";

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

    /* ------------------------ Unified Cross-Path Metrics (Phase 5) -------------------------- */

    /**
     * Request success QPS — shared metric reported by all three scheduling paths
     * (BATCH, QUEUE, DIRECT), tagged with {@link #TAG_PATH}.
     */
    public static final String REQUEST_SUCCESS_QPS = "flexlb.request.success.qps";

    /**
     * Request failure QPS — shared metric, tagged with {@link #TAG_PATH} and {@link #TAG_CODE}.
     */
    public static final String REQUEST_FAILURE_QPS = "flexlb.request.failure.qps";

    /**
     * Request timeout QPS — shared metric, tagged with {@link #TAG_PATH}.
     */
    public static final String REQUEST_TIMEOUT_QPS = "flexlb.request.timeout.qps";

    /**
     * Request cancel QPS — shared metric, tagged with {@link #TAG_PATH}.
     */
    public static final String REQUEST_CANCEL_QPS = "flexlb.request.cancel.qps";

    /**
     * Time to first token in milliseconds (TIMER) — shared metric, tagged with {@link #TAG_PATH}.
     */
    public static final String REQUEST_TTFT_MS = "flexlb.request.ttft.ms";

    /**
     * Inflight size gauge — shared metric, tagged with {@link #TAG_PATH}.
     */
    public static final String INFLIGHT_SIZE = "flexlb.inflight.size";

    /* ------------------------ Unified Tag Constants -------------------------- */

    /**
     * Tag key for the scheduling path (BATCH / QUEUE / DIRECT).
     */
    public static final String TAG_PATH = "path";

    /**
     * Tag key for the engine role (PREFILL / DECODE / PDFUSION).
     */
    public static final String TAG_ROLE = "role";

    /**
     * Tag key for the engine IP address.
     */
    public static final String TAG_ENGINE_IP = "engineIp";

    /**
     * Tag key for the error code (used on failure metrics).
     */
    public static final String TAG_CODE = "code";

    /* ------------------------ Path Value Constants -------------------------- */

    /**
     * Path value for the batch scheduling path.
     */
    public static final String PATH_BATCH = "BATCH";

    /**
     * Path value for the queue scheduling path.
     */
    public static final String PATH_QUEUE = "QUEUE";

    /**
     * Path value for the direct scheduling path.
     */
    public static final String PATH_DIRECT = "DIRECT";
}
