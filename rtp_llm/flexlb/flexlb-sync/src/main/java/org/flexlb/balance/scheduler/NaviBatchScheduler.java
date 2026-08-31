package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NaviBatchSchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiConsumer;

/**
 * Global attention-batch scheduler that ports the navi_sched cost-based
 * scheduler into FlexLB as the {@code NAVI_BATCH} scheduling mode.
 *
 * <p>The scheduler collects arriving requests into one global attbatch window
 * and then jointly assigns the whole window to eligible Prefill endpoints with
 * a single {@link NaviPgdOptimizer} run. It differs from the per-request
 * {@code CostBasedPrefillStrategy} in that the placement of every request is
 * decided together (projected-gradient descent over the shared cost matrix),
 * so co-located same-node batching emerges from the optimizer rather than from
 * greedy per-request selection.
 *
 * <p>Window flushing follows the navi three-trigger contract:
 * <ul>
 *   <li>the buffer reaches {@code naviBatchMaxCount} — flush immediately;</li>
 *   <li>the first request of an empty buffer arms a {@code naviBatchWindowMs}
 *       timer — flush when it fires;</li>
 *   <li>a defensive overflow guard flushes when the buffer grows beyond twice
 *       the configured maximum;</li>
 *   <li>with L2 capacity gating enabled, an engine slot-free signal (observed
 *       {@code available_concurrency} rising to a positive value) flushes
 *       immediately. The signal is a trigger, never a decision domain: the
 *       window timer stays armed as the worst-case backstop, so the flush
 *       gap is bounded by {@code naviBatchWindowMs} when no signal fires.</li>
 * </ul>
 *
 * <p>L2 capacity gating (off by default, {@code naviCapacityGatingEnabled})
 * additionally shrinks each round's PGD feasible domain to endpoints that
 * currently look able to accept work (see {@link #hasFreeCapacity}). Full
 * endpoints simply do not join that round's joint optimization — smooth
 * feasible-domain contraction instead of all-or-nothing weight zeroing. If
 * every eligible endpoint is observed at capacity the window is requeued
 * into the one and only buffer (no shadow queue) and retries on the next
 * trigger, bounded by the {@code naviCapacityStallLimitMs} starvation
 * valve which forces the full domain.
 *
 * <p>All optimize-and-dispatch work is serialized on a single-thread executor,
 * which keeps the {@link NaviPgdOptimizer} instance thread-confined (it is not
 * safe for concurrent use).
 *
 * <p>Delivery integration follows the intake2 ownership model: optional batch
 * dispatch crosses the {@link BatchSubmissionPort} transport boundary with one
 * {@link BatchSubmissionPort.Command} per optimized node group, and transport
 * outcomes arrive through the submission observer callback; the fallback
 * route-decision path answers with plain {@link ServerStatus} metadata exactly
 * like a DIRECT route. Decode selection goes through
 * {@link ConfiguredLoadBalanceSelector}; the returned {@link SelectedRole}
 * generation pin is released immediately because a navi route decision carries
 * no endpoint ownership (the engine-side PD mechanism owns decode placement).
 */
@Component
public class NaviBatchScheduler {

    private static final String REASON = "navi_batch";

    private final ConfigService configService;
    private final EndpointRegistry endpointRegistry;
    private final CacheAwareService cacheAwareService;
    private final BatchSubmissionPort batchSubmissionPort;
    private final ConfiguredLoadBalanceSelector decodeSelector;

    private final BatchIdGenerator batchIdGenerator;
    private final ScheduledExecutorService flushExecutor;

    /** Thread-confined to {@link #flushExecutor}; never shared across threads. */
    private final NaviPgdOptimizer optimizer = new NaviPgdOptimizer();

    private final ReentrantLock lock = new ReentrantLock();
    private List<PendingRequest> buffer = new ArrayList<>();
    private ScheduledFuture<?> pendingFlush;
    private long oldestArrivalNanos;
    private volatile boolean closed;
    /** Cumulative P0-1 degrade events (decode unavailable -> prefill-only). */
    private final AtomicLong decodeDegradeCount = new AtomicLong();
    /**
     * L2 capacity gating: master-side count of EnqueueBatch batches submitted
     * to an endpoint but not yet at a transport terminal outcome. Navi batches
     * bypass PrefillState's reserveBatch ledger, so this is the scheduler's
     * own O(1) inflight view, keyed by endpoint ip:port.
     */
    private final ConcurrentHashMap<String, AtomicInteger> inflightBatches =
            new ConcurrentHashMap<>();
    /**
     * L2 capacity gating: last observed {@code availableConcurrency} per
     * endpoint, used for O(1) slot-free edge detection on the status signal.
     */
    private final ConcurrentHashMap<String, Long> lastAvailableConcurrency =
            new ConcurrentHashMap<>();
    /**
     * L2 capacity gating: coalescing flag so a burst of status signals merges
     * into at most one queued flush task on the single-thread executor.
     */
    private final AtomicBoolean signalFlushPending = new AtomicBoolean();
    /** Cumulative feasible-domain requeues (every eligible endpoint observed full). */
    private final AtomicLong capacityRequeueCount = new AtomicLong();

    @Autowired
    public NaviBatchScheduler(ConfigService configService,
                              EndpointRegistry endpointRegistry,
                              CacheAwareService cacheAwareService,
                              BatchSubmissionPort batchSubmissionPort,
                              ConfiguredLoadBalanceSelector decodeSelector) {
        this.configService = configService;
        this.endpointRegistry = endpointRegistry;
        this.cacheAwareService = cacheAwareService;
        this.batchSubmissionPort = batchSubmissionPort;
        this.decodeSelector = decodeSelector;
        this.batchIdGenerator = new BatchIdGenerator(detectLocalIp(), 0);
        AtomicLong threadIndex = new AtomicLong();
        ThreadFactory threadFactory = runnable -> {
            Thread thread = new Thread(runnable,
                    "navi-batch-scheduler-" + threadIndex.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
        this.flushExecutor = Executors.newSingleThreadScheduledExecutor(threadFactory);
    }

    /** One buffered request awaiting a global attbatch placement decision. */
    private record PendingRequest(BalanceContext ctx,
                                  CompletableFuture<Response> future,
                                  long arrivalNanos) {
    }

    // ==================== Public API ====================

    /**
     * Engine status signal for the L2 capacity closed loop. Called (outside
     * any worker-status lock, from the status polling callback) every time a
     * prefill endpoint's committed observation is republished; the scheduler
     * keeps only the last observed {@code availableConcurrency} per endpoint
     * and turns a 0 → positive edge (a freed slot) into an immediate flush of
     * the buffered window — without waiting for the window timer.
     *
     * <p>The whole path is O(1): one map read/write, one CAS, at most one
     * task submission. Signal bursts coalesce through
     * {@link #signalFlushPending}; with gating disabled the call returns
     * right after the config read.
     */
    public void onEngineObservationPublished(
            PrefillEndpoint endpoint,
            WorkerStatus.EngineObservation observation) {
        if (endpoint == null || observation == null) {
            return;
        }
        if (!currentConfig().isNaviCapacityGatingEnabled()) {
            return;
        }
        Long available = observation.availableConcurrency();
        if (available == null || available <= 0L) {
            // No free slot observed (or the engine cannot report one):
            // remember the non-positive value so the next rise is an edge.
            lastAvailableConcurrency.put(endpoint.ipPort(), 0L);
            return;
        }
        Long previous = lastAvailableConcurrency.put(
                endpoint.ipPort(), available);
        if (previous == null || previous <= 0L) {
            requestSignalFlush();
        }
    }

    /**
     * Submit one request into the global attbatch window. The returned future
     * completes with the route decision (selected Prefill target) once the
     * window flushes and the optimizer places the request.
     */
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        if (ctx == null || ctx.getRequest() == null) {
            future.complete(Response.error(StrategyErrorType.INVALID_REQUEST));
            return future;
        }
        PendingRequest pending = new PendingRequest(ctx, future, System.nanoTime());
        List<PendingRequest> toFlush = null;
        lock.lock();
        try {
            if (closed) {
                future.complete(Response.error(StrategyErrorType.BATCH_DISPATCH_FAILED));
                return future;
            }
            if (buffer.isEmpty()) {
                oldestArrivalNanos = pending.arrivalNanos();
            }
            buffer.add(pending);
            int size = buffer.size();
            int maxCount = Math.max(1, currentConfig().getNaviBatchMaxCount());
            if (size >= maxCount || size > 2 * maxCount) {
                toFlush = swapBufferLocked();
            } else if (size == 1) {
                scheduleTimerLocked(currentConfig().getNaviBatchWindowMs());
            }
        } finally {
            lock.unlock();
        }
        if (toFlush != null) {
            dispatchToExecutor(toFlush);
        }
        return future;
    }

    @PreDestroy
    public void shutdown() {
        List<PendingRequest> drained;
        lock.lock();
        try {
            closed = true;
            drained = swapBufferLocked();
        } finally {
            lock.unlock();
        }
        if (!drained.isEmpty()) {
            failAll(drained, StrategyErrorType.BATCH_DISPATCH_FAILED);
        }
        flushExecutor.shutdown();
    }

    // ==================== Window management ====================

    /** Timer callback: flush whatever accumulated within the collection window. */
    private void onWindowElapsed() {
        flushBufferNow();
    }

    /**
     * Coalesce a capacity signal into at most one queued flush task. The
     * flag is released when the task starts running, so signals arriving
     * while a flush is in flight queue exactly one follow-up; signals
     * arriving with an empty buffer are no-ops inside the task.
     */
    private void requestSignalFlush() {
        if (!signalFlushPending.compareAndSet(false, true)) {
            return;
        }
        try {
            flushExecutor.execute(() -> {
                signalFlushPending.set(false);
                flushBufferNow();
            });
        } catch (RuntimeException rejected) {
            signalFlushPending.set(false);
        }
    }

    /**
     * Swap out and optimize the whole current buffer. Runs inline on the
     * flush-executor thread (both callers — window timer and capacity
     * signal — are already there), preserving window ordering.
     */
    private void flushBufferNow() {
        List<PendingRequest> toFlush;
        lock.lock();
        try {
            if (buffer.isEmpty()) {
                return;
            }
            toFlush = swapBufferLocked();
        } finally {
            lock.unlock();
        }
        // Already on the flush executor thread; run inline to preserve ordering.
        doOptimizeAndDispatch(toFlush);
    }

    private List<PendingRequest> swapBufferLocked() {
        if (pendingFlush != null) {
            pendingFlush.cancel(false);
            pendingFlush = null;
        }
        List<PendingRequest> previous = buffer;
        buffer = new ArrayList<>();
        oldestArrivalNanos = 0L;
        return previous;
    }

    private void scheduleTimerLocked(long windowMs) {
        long delay = Math.max(0L, windowMs);
        try {
            pendingFlush = flushExecutor.schedule(
                    this::onWindowElapsed, delay, TimeUnit.MILLISECONDS);
        } catch (RuntimeException rejected) {
            // Executor already shutting down; fail the buffered work eagerly.
            List<PendingRequest> drained = swapBufferLocked();
            if (!drained.isEmpty()) {
                failAll(drained, StrategyErrorType.BATCH_DISPATCH_FAILED);
            }
        }
    }

    private void dispatchToExecutor(List<PendingRequest> batch) {
        try {
            flushExecutor.execute(() -> doOptimizeAndDispatch(batch));
        } catch (RuntimeException rejected) {
            failAll(batch, StrategyErrorType.BATCH_DISPATCH_FAILED);
        }
    }

    private NaviBatchSchedulerConfig currentConfig() {
        FlexlbConfig config = configService.loadBalanceConfig();
        return config.isNaviBatch()
                ? config.naviBatchScheduler()
                : new NaviBatchSchedulerConfig();
    }

    // ==================== Optimize + dispatch (flush-executor thread) ====================

    private void doOptimizeAndDispatch(List<PendingRequest> batch) {
        if (batch == null || batch.isEmpty()) {
            return;
        }
        try {
            FlexlbConfig config = configService.loadBalanceConfig();
            NaviBatchSchedulerConfig cfg = config.isNaviBatch()
                    ? config.naviBatchScheduler()
                    : new NaviBatchSchedulerConfig();

            // Average request length of this window (tokens); the O(1)
            // pricing basis for every node's engine queue estimate below.
            long windowTotalTokens = 0L;
            for (PendingRequest pending : batch) {
                Request request = pending.ctx().getRequest();
                windowTotalTokens += request == null
                        ? 1L : Math.max(1L, request.getSeqLen());
            }
            long windowAvgTokens = Math.max(1L, windowTotalTokens / batch.size());

            // 1. Collect eligible Prefill endpoints (alive + learning
            //    predictor) and their navi latency parameters. With L2
            //    capacity gating enabled the PGD feasible domain shrinks to
            //    endpoints that currently look able to accept work — full
            //    endpoints do not join this round's joint optimization at
            //    all (smooth feasible-domain contraction, never weight
            //    zeroing). See #collectNodeCandidates / #hasFreeCapacity.
            boolean capacityGating = cfg.isNaviCapacityGatingEnabled();
            Integer inflightCap = capacityGating
                    ? dispatcherInflightCap(config) : null;
            NodeCandidates candidates = collectNodeCandidates(
                    windowAvgTokens, capacityGating, inflightCap);
            if (candidates.nodes.isEmpty()) {
                boolean anyAtCapacity =
                        capacityGating && candidates.capacityFullCount > 0;
                if (!anyAtCapacity) {
                    failAll(batch, StrategyErrorType.NO_AVAILABLE_WORKER);
                    return;
                }
                // Every eligible endpoint is observed at capacity: skip this
                // flush round and requeue the window into the buffer — the
                // requests stay in the one and only buffer, no new waiting
                // area. The stall valve below bounds the requeue chain.
                long stalledMs = (System.nanoTime() - batch.get(0).arrivalNanos())
                        / 1_000_000L;
                if (stalledMs < Math.max(0L, cfg.getNaviCapacityStallLimitMs())) {
                    long requeues = capacityRequeueCount.incrementAndGet();
                    if (requeues == 1L || requeues % 100L == 0L) {
                        Logger.info("flexlb_navi_capacity_requeue requests={} "
                                        + "stalled_ms={} full_endpoints={} "
                                        + "stall_limit_ms={} total_requeues={}",
                                batch.size(), stalledMs,
                                candidates.capacityFullCount,
                                Math.max(0L, cfg.getNaviCapacityStallLimitMs()),
                                requeues);
                    }
                    requeueBatch(batch);
                    return;
                }
                // Stall valve: the oldest requeued request has waited past the
                // configured limit, so a stale or misleading capacity
                // observation must not starve it — force the full endpoint
                // domain (pre-L2 behavior) for this round.
                Logger.warn("flexlb_navi_capacity_stall_forced requests={} "
                                + "stalled_ms={} full_endpoints={}: "
                                + "forcing full feasible domain",
                        batch.size(), stalledMs, candidates.capacityFullCount);
                candidates = collectNodeCandidates(windowAvgTokens, false, null);
                if (candidates.nodes.isEmpty()) {
                    failAll(batch, StrategyErrorType.NO_AVAILABLE_WORKER);
                    return;
                }
            }

            int nodeCount = candidates.nodes.size();
            int requestCount = batch.size();

            // 2. Build the optimizer inputs. cacheHitTokens is node-major:
            //    index = nodeIndex * requestCount + requestIndex.
            List<PrefillEndpoint> nodes = candidates.nodes;
            double[][] latencyParameters = candidates.params.toArray(new double[0][]);
            double[] queueWaitMs = new double[nodeCount];
            for (int n = 0; n < nodeCount; n++) {
                queueWaitMs[n] = candidates.queueWait.get(n);
            }
            long[] requestTokenCounts = new long[requestCount];
            long[] cacheHitTokens = new long[nodeCount * requestCount];
            for (int r = 0; r < requestCount; r++) {
                Request request = batch.get(r).ctx().getRequest();
                requestTokenCounts[r] = Math.max(1L, request.getSeqLen());
                // Cache matching mirrors CostBasedPrefillStrategy: engine
                // ip-port -> prefix match length. No group filter — the navi
                // window spans every active prefill endpoint.
                Map<String, Integer> matches =
                        cacheAwareService.findMatchingEngines(
                                request.getBlockCacheKeys(),
                                RoleType.PREFILL, null);
                for (int n = 0; n < nodeCount; n++) {
                    cacheHitTokens[n * requestCount + r] =
                            cacheHitOf(nodes.get(n), matches, request);
                }
            }

            // Formal observation of the optimizer's per-node queue-wait
            // inputs — the designated observability channel for NAVI
            // optimizer input (stress-test forensics depend on it).
            // queue_wait_ms is what the optimizer consumes; engine_ms /
            // ledger_ms expose the two components merged by the max() above;
            // waiting is the engine-reported waitingQueryLen per node and
            // avg_tokens the window-scalar pricing basis behind engine_ms.
            // capacity_gated / capacity_full expose the L2 feasible-domain
            // contraction: whether gating shaped this round and how many
            // eligible endpoints it removed.
            // Rate note: one INFO line per optimize-and-dispatch call —
            // ceiling ~33 lines/s at naviBatchWindowMs=30 under continuous
            // full load, each line well under ~400 chars; logback's
            // 50MB×5-day rolling with a 2GB total cap bounds the volume.
            // Kept at INFO deliberately: this is an experimental scheduler
            // and the line is its only optimizer-input observability; if it
            // ever moves to production traffic, demote to debug or sample.
            StringBuilder queueWaitDiag = new StringBuilder(nodeCount * 8);
            StringBuilder engineMsDiag = new StringBuilder(nodeCount * 8);
            StringBuilder ledgerMsDiag = new StringBuilder(nodeCount * 4);
            StringBuilder waitingDiag = new StringBuilder(nodeCount * 4);
            for (int n = 0; n < nodeCount; n++) {
                if (n > 0) {
                    queueWaitDiag.append(',');
                    engineMsDiag.append(',');
                    ledgerMsDiag.append(',');
                    waitingDiag.append(',');
                }
                queueWaitDiag.append((long) queueWaitMs[n]);
                engineMsDiag.append(candidates.engineMs.get(n));
                ledgerMsDiag.append(candidates.ledgerMs.get(n));
                waitingDiag.append(candidates.waiting.get(n));
            }
            Logger.info("flexlb_navi_queue_wait nodes={} requests={} "
                            + "avg_tokens={} queue_wait_ms={} engine_ms={} "
                            + "ledger_ms={} waiting={} capacity_gated={} "
                            + "capacity_full={}",
                    nodeCount, requestCount, windowAvgTokens, queueWaitDiag,
                    engineMsDiag, ledgerMsDiag, waitingDiag, capacityGating,
                    candidates.capacityFullCount);

            // 3. Run the joint PGD assignment.
            optimizer.configure(
                    cfg.getNaviBatchLambda(),
                    cfg.getNaviBatchAlpha(),
                    cfg.getNaviBatchAlphaDecay(),
                    cfg.getNaviBatchMinAlpha(),
                    cfg.getNaviBatchMaxLoopCount(),
                    cfg.getNaviBatchTimeBudgetUs());
            NaviPgdOptimizer.OptimizeResult result = optimizer.optimize(
                    nodeCount, requestCount, latencyParameters, queueWaitMs,
                    requestTokenCounts, cacheHitTokens);
            if (result == null) {
                failAll(batch, StrategyErrorType.NO_AVAILABLE_WORKER);
                return;
            }

            // 4. Group requests by their assigned node (same-node co-batching).
            int[] selected = result.selectedNodeIndexes();
            Map<Integer, List<Integer>> groups = new LinkedHashMap<>();
            for (int r = 0; r < requestCount; r++) {
                int nodeIndex = selected[r];
                if (nodeIndex < 0 || nodeIndex >= nodeCount) {
                    // Optimizer left this request unplaced; surface a retryable error.
                    batch.get(r).future().complete(
                            Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
                    continue;
                }
                groups.computeIfAbsent(nodeIndex, ignored -> new ArrayList<>()).add(r);
            }

            // 5. Dispatch each group to its target with one shared batch id.
            boolean batchDispatch = config.getDispatcher() instanceof BatchDispatcherConfig
                    && batchSubmissionPort != null;
            for (Map.Entry<Integer, List<Integer>> group : groups.entrySet()) {
                PrefillEndpoint endpoint = nodes.get(group.getKey());
                List<Integer> requestIndexes = group.getValue();
                long batchId = batchIdGenerator.nextBatchId();
                boolean dispatched = false;
                if (batchDispatch) {
                    dispatched = tryDispatchAsBatch(
                            endpoint, batchId, batch, requestIndexes,
                            requestCount, cacheHitTokens, group.getKey());
                }
                if (!dispatched) {
                    completeRouteDecisions(
                            endpoint, batchId, batch, requestIndexes,
                            requestCount, cacheHitTokens, group.getKey());
                }
            }
        } catch (Throwable failure) {
            Logger.warn("NAVI_BATCH optimize/dispatch failed for {} requests",
                    batch.size(), failure);
            failAll(batch, StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    // ==================== L2 capacity feasible domain ====================

    /** Optimizer-input candidates for one flush round (flush-executor thread). */
    private static final class NodeCandidates {
        final List<PrefillEndpoint> nodes = new ArrayList<>();
        final List<double[]> params = new ArrayList<>();
        final List<Double> queueWait = new ArrayList<>();
        final List<Long> ledgerMs = new ArrayList<>();
        final List<Long> engineMs = new ArrayList<>();
        final List<Long> waiting = new ArrayList<>();
        /** Eligible endpoints removed by capacity gating this round. */
        int capacityFullCount;
    }

    /**
     * Collect the PGD feasible domain for one flush round. When
     * {@code applyCapacityGating} is set, endpoints failing
     * {@link #hasFreeCapacity} are counted and skipped: they do not join
     * this round's joint optimization at all. O(eligible endpoints) with
     * O(1) work per endpoint — no per-request or per-task traversal.
     */
    private NodeCandidates collectNodeCandidates(long windowAvgTokens,
                                                 boolean applyCapacityGating,
                                                 Integer inflightCap) {
        NodeCandidates candidates = new NodeCandidates();
        for (PrefillEndpoint endpoint
                : endpointRegistry.snapshotPrefillEndpoints().values()) {
            if (endpoint.getStatus() == null
                    || !endpoint.getStatus().isActiveGeneration()) {
                continue;
            }
            PrefillTimePredictor predictor = endpoint.getPredictor();
            if (!(predictor instanceof LearningPredictor learning)) {
                continue;
            }
            double[] weights = learning.weightsSnapshot();
            if (weights == null) {
                continue;
            }
            WorkerStatus.EngineObservation observation =
                    endpoint.getStatus().committedEngineObservation();
            if (applyCapacityGating
                    && !hasFreeCapacity(endpoint.ipPort(), observation, inflightCap)) {
                candidates.capacityFullCount++;
                continue;
            }
            // Queue wait merges two independent measures of the work
            // already sitting on this endpoint. ledgerMs is the
            // per-endpoint work ledger: navi batches bypass
            // reserveBatch/reserveRoute, so under a navi-only deployment
            // the ledger's committed snapshot goes empty (unknown engine
            // requests make totalRemainingWorkMs() absent) and reads as
            // zero — the defect fixed here. engineMs is the engine's
            // directly reported waiting count priced into milliseconds
            // through the same navi latency model the optimizer uses
            // (see #engineQueueWaitEstimateMs); it is ledger-independent,
            // O(1) in queue depth, and stays observable under navi-only
            // load. Both estimate the same backlog, so taking the max
            // avoids double counting either measure while staying
            // conservative; in a mixed deployment the larger of ledgered
            // and engine-reported backlog wins.
            long ledgerMs = endpoint.getLoadMetric().orElse(0L);
            long engineMs = engineQueueWaitEstimateMs(
                    observation, weights, windowAvgTokens);
            long waitMs = Math.max(ledgerMs, engineMs);
            candidates.nodes.add(endpoint);
            candidates.params.add(weights);
            candidates.queueWait.add((double) Math.max(0L, waitMs));
            candidates.ledgerMs.add(Math.max(0L, ledgerMs));
            candidates.engineMs.add(engineMs);
            candidates.waiting.add(observation == null
                    ? 0L : Math.max(0L, observation.waitingQueryLen()));
        }
        return candidates;
    }

    /**
     * Free-capacity verdict for one endpoint, the intersection (AND) of the
     * two capacity signals — the conservative combination:
     *
     * <ul>
     *   <li>Master side (primary): this scheduler's own EnqueueBatch
     *       inflight ledger against
     *       {@code maxInflightBatchesPerPrefillWorker}. Real-time and exact,
     *       but blind to work the engine took through other paths (e.g.
     *       route-decision traffic driven straight at the engine).</li>
     *   <li>Engine side (secondary): the committed
     *       {@code availableConcurrency} observation (~20 ms polling
     *       freshness; the mock engine reports
     *       {@code max_prefill_concurrency - running prefill batches}).
     *       {@code null} means the engine does not report the field, in
     *       which case only the master ledger decides.</li>
     * </ul>
     *
     * <p>Both lookups are O(1) map reads; the endpoint's waiting-queue depth
     * is deliberately not traversed (hot-path red line).
     */
    private boolean hasFreeCapacity(String endpointKey,
                                    WorkerStatus.EngineObservation observation,
                                    Integer inflightCap) {
        if (inflightCap != null && inflightCap > 0) {
            AtomicInteger inflight = inflightBatches.get(endpointKey);
            if (inflight != null && inflight.get() >= inflightCap) {
                return false;
            }
        }
        if (observation != null && observation.availableConcurrency() != null) {
            return observation.availableConcurrency() > 0L;
        }
        return true;
    }

    /** Master-side inflight batch cap from the batch dispatcher config. */
    private static Integer dispatcherInflightCap(FlexlbConfig config) {
        return config.getDispatcher() instanceof BatchDispatcherConfig batch
                ? batch.maxInflightDeliveriesPerPrefillWorker()
                : null;
    }

    /**
     * Return a requeued window to the front of the buffer (its requests
     * arrived earlier than anything submitted meanwhile) and re-arm the
     * window timer so the worst-case retry gap stays one window period; a
     * slot-free signal can still preempt the timer at any moment. The
     * requests never leave the one and only buffer — no shadow queue.
     */
    private void requeueBatch(List<PendingRequest> batch) {
        boolean failInstead = false;
        lock.lock();
        try {
            if (closed) {
                failInstead = true;
            } else {
                if (pendingFlush != null) {
                    pendingFlush.cancel(false);
                    pendingFlush = null;
                }
                List<PendingRequest> merged =
                        new ArrayList<>(batch.size() + buffer.size());
                merged.addAll(batch);
                merged.addAll(buffer);
                buffer = merged;
                long earliest = batch.get(0).arrivalNanos();
                oldestArrivalNanos = oldestArrivalNanos == 0L
                        ? earliest : Math.min(oldestArrivalNanos, earliest);
                scheduleTimerLocked(
                        Math.max(0L, currentConfig().getNaviBatchWindowMs()));
            }
        } finally {
            lock.unlock();
        }
        if (failInstead) {
            failAll(batch, StrategyErrorType.BATCH_DISPATCH_FAILED);
        }
    }

    // ==================== Route-decision completion ====================

    private void completeRouteDecisions(PrefillEndpoint endpoint,
                                        long batchId,
                                        List<PendingRequest> batch,
                                        List<Integer> requestIndexes,
                                        int requestCount,
                                        long[] cacheHitTokens,
                                        int nodeIndex) {
        for (int requestIndex : requestIndexes) {
            PendingRequest pending = batch.get(requestIndex);
            long cacheHit = cacheHitTokens[nodeIndex * requestCount + requestIndex];
            ServerStatus prefill = buildServerStatus(
                    endpoint, pending.ctx().getRequestId(), cacheHit);

            // Select decode endpoint for PD separation. P0-1: decode
            // selection failure (typically decode saturation under overload)
            // degrades to prefill-only routing — the v2 overload-tolerant
            // semantics — instead of rejecting; the engine-side PD mechanism
            // absorbs decode placement.
            ServerStatus decode = selectDecode(
                    pending.ctx(), endpoint.getStatus().getGroup());
            if (decode == null) {
                long degraded = decodeDegradeCount.incrementAndGet();
                if (degraded == 1L || degraded % 100L == 0L) {
                    Logger.warn("NAVI_BATCH decode selection unavailable; degrading to "
                            + "prefill-only route, total degrade events={}: request_id={}",
                            degraded, pending.ctx().getRequestId());
                }
            }

            Response response = new Response();
            response.setSuccess(true);
            List<ServerStatus> statuses = decode != null
                    ? List.of(prefill, decode) : List.of(prefill);
            response.setServerStatus(statuses);
            pending.future().complete(response);
            if (Logger.isDebugEnabled()) {
                Logger.debug("NAVI_BATCH route decision request_id={} target={} "
                                + "batch_id={} hit_cache={} decode={}",
                        pending.ctx().getRequestId(), endpoint.ipPort(),
                        batchId, cacheHit,
                        decode != null ? decode.getServerIp() + ":" + decode.getHttpPort() : "none");
            }
        }
    }

    // ==================== Optional BATCH_ENQUEUE dispatch ====================

    /**
     * Best-effort EnqueueBatch dispatch for the BATCH dispatcher mode. Returns
     * {@code false} (degrade to a route decision) when the batch cannot be
     * constructed or the transport has no submission capacity, so every
     * request still receives a placement.
     */
    private boolean tryDispatchAsBatch(PrefillEndpoint endpoint,
                                       long batchId,
                                       List<PendingRequest> batch,
                                       List<Integer> requestIndexes,
                                       int requestCount,
                                       long[] cacheHitTokens,
                                       int nodeIndex) {
        List<ScheduledRequest> items = new ArrayList<>(requestIndexes.size());
        List<PendingRequest> members = new ArrayList<>(requestIndexes.size());
        try {
            for (int requestIndex : requestIndexes) {
                PendingRequest pending = batch.get(requestIndex);
                if (pending.ctx().getGenerateInputPb() == null) {
                    return false;
                }
                long cacheHit = cacheHitTokens[nodeIndex * requestCount + requestIndex];
                ServerStatus prefill = buildServerStatus(
                        endpoint, pending.ctx().getRequestId(), cacheHit);

                // Select decode endpoint for PD separation. Decode
                // unavailable degrades the whole group to the route-decision
                // path (completeRouteDecisions degrades per-request to
                // prefill-only instead of rejecting).
                ServerStatus decode = selectDecode(
                        pending.ctx(), endpoint.getStatus().getGroup());
                if (decode == null) {
                    return false;
                }
                DecodeEndpoint decodeEp = resolveDecodeEndpoint(decode);

                Response routeResponse = new Response();
                routeResponse.setSuccess(true);
                routeResponse.setServerStatus(List.of(prefill, decode));
                ScheduledRequest item = new ScheduledRequest(
                        pending.ctx(), pending.future(), routeResponse,
                        prefill, decode, endpoint, decodeEp, null,
                        System.currentTimeMillis());
                items.add(item);
                members.add(pending);
            }
        } catch (RuntimeException constructionFailure) {
            Logger.debug("NAVI_BATCH batch build failed, using route decisions",
                    constructionFailure);
            return false;
        }
        CapacityBoundary.Attempt<BatchSubmissionPort.PreparedSubmission> attempt;
        try {
            attempt = batchSubmissionPort.tryPrepareSubmission();
        } catch (RuntimeException prepareFailure) {
            Logger.debug("NAVI_BATCH dispatcher prepare failed, using route decisions",
                    prepareFailure);
            return false;
        }
        if (!(attempt instanceof
                CapacityBoundary.Attempt.Accepted<BatchSubmissionPort.PreparedSubmission> accepted)) {
            // Admission unavailable (or failed): degrade to route decisions so
            // every request still receives a placement.
            Logger.debug("NAVI_BATCH dispatcher admission unavailable, using route decisions");
            return false;
        }
        BatchSubmissionPort.PreparedSubmission prepared = accepted.value();
        // L2 capacity ledger: the batch is accounted as inflight before the
        // submit call, so synchronous observer callbacks (which settle the
        // ledger on the last member) never decrement below zero. A submit
        // failure rolls the increment back exactly once (idempotent settle).
        String endpointKey = endpoint.ipPort();
        inflightBatches.computeIfAbsent(endpointKey, ignored -> new AtomicInteger())
                .incrementAndGet();
        NaviDispatchObserver observer = new NaviDispatchObserver(
                members, endpoint, cacheHitTokens, requestCount, nodeIndex,
                requestIndexes, endpointKey);
        boolean submitted = false;
        try {
            BatchSubmissionPort.Command command = new BatchSubmissionPort.Command(
                    List.copyOf(items), batchId, 0L,
                    new DeliveryMetadata(REASON, 0));
            prepared.submitBatch(command, observer);
            submitted = true;
            return true;
        } catch (RuntimeException submitFailure) {
            observer.abandonInflight();
            Logger.debug("NAVI_BATCH dispatcher submit failed, using route decisions",
                    submitFailure);
            return false;
        } finally {
            if (!submitted) {
                // Resolve the unused preparation without changing business
                // outcome; after a successful submit close is a no-op.
                try {
                    prepared.close();
                } catch (RuntimeException closeFailure) {
                    Logger.debug("NAVI_BATCH prepared submission close failed",
                            closeFailure);
                }
            }
        }
    }

    /** Completes buffered futures from EnqueueBatch transport outcomes. */
    private final class NaviDispatchObserver
            implements BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> {
        private final List<PendingRequest> members;
        private final PrefillEndpoint endpoint;
        private final long[] cacheHitTokens;
        private final int requestCount;
        private final int nodeIndex;
        private final List<Integer> requestIndexes;
        /** L2 capacity ledger: endpoint key and settle-once state. */
        private final String endpointKey;
        private final AtomicInteger remainingMembers;
        private final AtomicBoolean inflightSettled;

        private NaviDispatchObserver(List<PendingRequest> members,
                                     PrefillEndpoint endpoint,
                                     long[] cacheHitTokens,
                                     int requestCount,
                                     int nodeIndex,
                                     List<Integer> requestIndexes,
                                     String endpointKey) {
            this.members = members;
            this.endpoint = endpoint;
            this.cacheHitTokens = cacheHitTokens;
            this.requestCount = requestCount;
            this.nodeIndex = nodeIndex;
            this.requestIndexes = requestIndexes;
            this.endpointKey = endpointKey;
            this.remainingMembers = new AtomicInteger(members.size());
            this.inflightSettled = new AtomicBoolean();
        }

        private int memberIndexOf(ScheduledRequest item) {
            for (int i = 0; i < members.size(); i++) {
                if (members.get(i).ctx() == item.ctx()) {
                    return i;
                }
            }
            return -1;
        }

        private long cacheHitOf(int memberIndex) {
            int requestIndex = requestIndexes.get(memberIndex);
            return cacheHitTokens[nodeIndex * requestCount + requestIndex];
        }

        /**
         * Roll the pre-submit inflight increment back when submission
         * failed before ownership moved to the dispatcher.
         */
        private void abandonInflight() {
            settleInflightOnce();
        }

        /** Decrement the endpoint inflight ledger exactly once per batch. */
        private void settleInflightOnce() {
            if (!inflightSettled.compareAndSet(false, true)) {
                return;
            }
            AtomicInteger counter = inflightBatches.get(endpointKey);
            if (counter != null) {
                counter.decrementAndGet();
            }
        }

        @Override
        public void accept(ScheduledRequest exactItem,
                           SlotDeliveryPort.Completion completion) {
            if (!(exactItem instanceof ScheduledRequest item)) {
                settleInflightOnce();
                return;
            }
            int memberIndex = memberIndexOf(item);
            if (memberIndex < 0) {
                // Defensive: an unknown item cannot be waited for, so settle
                // directly rather than leaking the inflight increment.
                settleInflightOnce();
                return;
            }
            PendingRequest pending = members.get(memberIndex);
            if (completion instanceof SlotDeliveryPort.Completion.Delivered) {
                ServerStatus prefill = buildServerStatus(
                        endpoint, item.requestId(), cacheHitOf(memberIndex));
                Response response = new Response();
                response.setSuccess(true);
                response.setEnqueuedByMaster(true);
                ServerStatus decode = item.decode();
                List<ServerStatus> statuses = decode != null
                        ? List.of(prefill, decode) : List.of(prefill);
                response.setServerStatus(statuses);
                pending.future().complete(response);
            } else {
                // Failed / TimedOut / Uncertain all surface as one retryable
                // batch-dispatch failure for the buffered request.
                pending.future().complete(
                        Response.error(StrategyErrorType.BATCH_DISPATCH_FAILED));
            }
            if (remainingMembers.decrementAndGet() == 0) {
                settleInflightOnce();
            }
        }
    }

    // ==================== Shared helpers ====================

    /**
     * Estimate one endpoint's queue wait in <b>milliseconds</b> from the
     * engine's directly reported waiting count — O(1) in queue depth.
     *
     * <p>Source: {@code committedEngineObservation().waitingQueryLen()} — the
     * engine's direct status report (≈20 ms polling freshness), which
     * bypasses the per-endpoint work ledger and therefore still observes a
     * queue under a navi-only deployment. Per-task queue detail (per-task
     * inputLengths) is deliberately not traversed:
     * the first implementation of this fix walked the task map on every
     * flush window, and that O(backlog) cost on the single-threaded flush
     * hot path collapsed flush throughput at high queue depths (measured:
     * flush gap p50 32→223 ms, window throughput 780→141 req/s, 17.56% of
     * requests never scheduled at the 600 QPS tier), so the aggregate form
     * below replaced it.
     *
     * <p>Unit conversion (hard requirement: milliseconds, never a raw
     * count): single-batch drain model. The {@code waitingQueryLen} queued
     * requests are synthesised into one batch of average window requests —
     * per-request linear cost is computed once from
     * {@code windowAvgTokens} at cacheHit 0 (conservative: queued work is
     * assumed fully un-cached), scaled by the waiting count, and mapped
     * through {@link NaviPrefillModel#calculateLatencyAndDerivative} with
     * this node's own learned weights — the same weights the optimizer
     * uses for placement. Model constants (bias, non-linear baseline) are
     * included exactly once (sum-then-latency, same shape the optimizer
     * itself evaluates), so the result is the modelled drain time of the
     * current queue: the FIFO wait a newly dispatched batch would sit
     * behind. Running work is not counted — it is not observable from the
     * status snapshot and is left for a later refinement; the avg-token
     * approximation replaces per-task inputLength detail.
     *
     * <p>Limits: {@code waitingQueryLen <= 0}, a null observation, or a
     * non-positive {@code windowAvgTokens} (cannot happen for a non-empty
     * window, but guarded) all yield 0 — the queue is then unobservable from
     * this endpoint and the ledger signal remains the only wait input.
     *
     * @param observation    the engine's committed observation; may be null
     * @param params         this endpoint's learned 9-parameter weights
     * @param windowAvgTokens average request length (tokens) of the current
     *                        flush window; the O(1) pricing basis
     */
    static long engineQueueWaitEstimateMs(WorkerStatus.EngineObservation observation,
                                          double[] params,
                                          long windowAvgTokens) {
        if (observation == null || params == null
                || params.length < NaviPrefillModel.LINEAR_PARAMETER_COUNT + 3) {
            return 0L;
        }
        long waitingCount = Math.max(0L, observation.waitingQueryLen());
        if (waitingCount <= 0L || windowAvgTokens <= 0L) {
            return 0L;
        }
        double perRequestCost = NaviPrefillModel.calculateRequestLinearCost(
                params, windowAvgTokens, 0L);
        double estimatedMs = NaviPrefillModel.calculateLatencyAndDerivative(
                waitingCount * perRequestCost, params)[0];
        return clampModelMs(estimatedMs);
    }

    /** Clamp a modelled latency to a finite, non-negative millisecond count. */
    private static long clampModelMs(double latencyMs) {
        if (!Double.isFinite(latencyMs) || latencyMs <= 0.0) {
            return 0L;
        }
        return latencyMs >= Long.MAX_VALUE
                ? Long.MAX_VALUE : Math.round(latencyMs);
    }

    private ServerStatus buildServerStatus(PrefillEndpoint endpoint,
                                           long requestId,
                                           long cacheHit) {
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(cacheHit);
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(RoleType.PREFILL);
        status.setRequestId(requestId);
        status.setGroup(endpoint.getStatus().getGroup());
        status.setServerIp(endpoint.getIp());
        status.setHttpPort(endpoint.getHttpPort());
        status.setGrpcPort(CommonUtils.toGrpcPort(endpoint.getHttpPort()));
        status.setDpRank(endpoint.getStatus().getDpRank());
        status.setDebugInfo(debugInfo);
        return status;
    }

    /**
     * Cache-hit tokens for one (endpoint, request) pair, mirroring
     * {@code CostBasedPrefillStrategy}: prefix blocks times block size, clamped
     * below the request length so a full match never zeroes out compute.
     */
    private static long cacheHitOf(PrefillEndpoint endpoint,
                                   Map<String, Integer> matches,
                                   Request request) {
        if (matches == null || matches.isEmpty() || request == null) {
            return 0L;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return 0L;
        }
        Integer prefixMatchLength = matches.get(endpoint.ipPort());
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return 0L;
        }
        WorkerStatus status = endpoint.getStatus();
        long blockSize = request.getCacheKeyBlockSize();
        if (blockSize <= 0L && status.getCacheStatus() != null) {
            blockSize = status.getCacheStatus().getBlockSize();
        }
        if (blockSize <= 0L) {
            return 0L;
        }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(
                    blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            rawHit = seqLen;
        }
        if (rawHit >= seqLen) {
            return Math.max(0L, seqLen - blockSize);
        }
        return Math.max(0L, rawHit);
    }

    private void failAll(List<PendingRequest> batch, StrategyErrorType errorType) {
        for (PendingRequest pending : batch) {
            pending.future().complete(Response.error(errorType));
        }
    }

    /**
     * Select a decode endpoint through the configured decode selector.
     * Returns a successful {@link ServerStatus} or {@code null} if selection
     * fails or no decode strategy is available (non-PD deployment).
     *
     * <p>The selection's generation pin is released immediately: a navi route
     * decision (and the optional batch dispatch it degrades from) carries no
     * decode ownership — {@link ServerStatus} stays response metadata and the
     * engine-side PD mechanism owns decode placement.
     */
    private ServerStatus selectDecode(BalanceContext ctx,
                                      String prefillGroup) {
        try {
            try (SelectedRole selected =
                    decodeSelector.select(ctx, RoleType.DECODE, prefillGroup)) {
                ServerStatus decode = selected.serverStatus();
                return decode != null && decode.isSuccess() ? decode : null;
            }
        } catch (RuntimeException selectionFailure) {
            Logger.debug("NAVI_BATCH decode selection unavailable: {}",
                    selectionFailure.getMessage());
            return null;
        }
    }

    private DecodeEndpoint resolveDecodeEndpoint(ServerStatus decode) {
        String decodeIpPort = decode.getServerIp() + ":" + decode.getHttpPort();
        return endpointRegistry.getDecode(decodeIpPort);
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (Exception unresolved) {
            return "127.0.0.1";
        }
    }
}
