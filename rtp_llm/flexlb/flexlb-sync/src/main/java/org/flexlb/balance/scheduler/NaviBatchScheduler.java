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
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
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
 *       the configured maximum.</li>
 * </ul>
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

            // 1. Collect eligible Prefill endpoints (alive + learning predictor)
            //    and their navi latency parameters.
            List<PrefillEndpoint> nodes = new ArrayList<>();
            List<double[]> nodeParams = new ArrayList<>();
            List<Double> nodeQueue = new ArrayList<>();
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
                // Navi batches bypass the per-endpoint work ledger, so the
                // committed-snapshot metric reflects only concurrent non-navi
                // ownership. An unobservable metric reads as zero wait, which
                // matches the pre-rebase NAVI_BATCH behaviour: its wait
                // estimate also stayed zero under a navi-only deployment.
                long waitMs = endpoint.getLoadMetric().orElse(0L);
                nodes.add(endpoint);
                nodeParams.add(weights);
                nodeQueue.add((double) Math.max(0L, waitMs));
            }

            int nodeCount = nodes.size();
            int requestCount = batch.size();
            if (nodeCount == 0) {
                failAll(batch, StrategyErrorType.NO_AVAILABLE_WORKER);
                return;
            }

            // 2. Build the optimizer inputs. cacheHitTokens is node-major:
            //    index = nodeIndex * requestCount + requestIndex.
            double[][] latencyParameters = nodeParams.toArray(new double[0][]);
            double[] queueWaitMs = new double[nodeCount];
            for (int n = 0; n < nodeCount; n++) {
                queueWaitMs[n] = nodeQueue.get(n);
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
        boolean submitted = false;
        try {
            BatchSubmissionPort.Command command = new BatchSubmissionPort.Command(
                    List.copyOf(items), batchId, 0L,
                    new DeliveryMetadata(REASON, 0));
            prepared.submitBatch(command, new NaviDispatchObserver(
                    members, endpoint, cacheHitTokens, requestCount, nodeIndex,
                    requestIndexes));
            submitted = true;
            return true;
        } catch (RuntimeException submitFailure) {
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

        private NaviDispatchObserver(List<PendingRequest> members,
                                     PrefillEndpoint endpoint,
                                     long[] cacheHitTokens,
                                     int requestCount,
                                     int nodeIndex,
                                     List<Integer> requestIndexes) {
            this.members = members;
            this.endpoint = endpoint;
            this.cacheHitTokens = cacheHitTokens;
            this.requestCount = requestCount;
            this.nodeIndex = nodeIndex;
            this.requestIndexes = requestIndexes;
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

        @Override
        public void accept(ScheduledRequest exactItem,
                           SlotDeliveryPort.Completion completion) {
            if (!(exactItem instanceof ScheduledRequest item)) {
                return;
            }
            int memberIndex = memberIndexOf(item);
            if (memberIndex < 0) {
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
        }
    }

    // ==================== Shared helpers ====================

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
