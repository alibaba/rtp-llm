package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.withSettings;

/**
 * na130_4 congested-queue candidate filter tests for
 * {@link CostBasedPrefillStrategy}: a prefill endpoint whose batcher queue
 * depth is at or beyond {@code flexlbCongestedQueueRatio ×
 * flexlbBatchQueueMaxSize} must not be a routing candidate, and when every
 * feasible endpoint is congested the existing least-loaded fallback must
 * still select one — routing never fails closed.
 *
 * <p>Also covers the na130_4 engine-wait signal: the engine-reported
 * {@code waitingQueryLen} (~20ms sync) both penalizes the Round-1 score
 * ({@code flexlbEngineWaitPenaltyEnabled}) and hard-filters candidates
 * ({@code flexlbEngineWaitHardFilterEnabled}), with the same
 * least-loaded fallback semantics.
 *
 * <p>Real {@link PrefillEndpoint}s are used so the filter reads the live
 * {@code WorkerBatcher.queueSize()}. Queue drain is pinned by the
 * fixed-window backpressure gate: {@code flexlbBatchFixedMaxInflightBatches=1}
 * plus one committed zero-cost inflight batch parks the batcher thread in
 * {@code processQueue} step 1, before any batch_full dispatch, so the queue
 * depth stays under test control. Auto-TPM is on (the filter is gated on it
 * like the depth term); queue depths are kept symmetric wherever the filter
 * itself must be the only deciding factor.
 */
class CostBasedPrefillStrategyCongestionTest {

    private static final int CONGESTED_PORT = 62_001;
    private static final int IDLE_PORT = 62_002;
    /** ceil(100 × 0.8) — congestion threshold under flexlbBatchQueueMaxSize=100. */
    private static final int CONGESTION_THRESHOLD = 80;

    private FlexlbConfig config;
    private EndpointRegistry endpointRegistry;
    private PrefillEndpoint congested;
    private PrefillEndpoint idle;
    private CostBasedPrefillStrategy strategy;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        // The congested-queue filter is gated on autoTpmEnabled like the
        // depth term, so exercise it in the Auto-TPM context it ships for.
        config.setAutoTpmEnabled(true);
        // Keep both engines resource-available regardless of queue depth
        config.setPrefillQueueSizeThreshold(1_000_000L);
        // Pin the batcher threads: one committed inflight batch >= the
        // backpressure limit parks processQueue before batch_full dispatch
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        // Small queue cap so a congested engine needs only 80 queued items
        config.setFlexlbBatchQueueMaxSize(100);
        // No enqueue deadline: assertions rely only on the backpressure park
        // and must not observe deadline-driven drops
        config.setFlexlbBatchEnqueueDeadlineMs(0);

        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(config);
        BatchSchedulerReporter reporter =
                Mockito.mock(BatchSchedulerReporter.class, withSettings().stubOnly());
        FlexlbBatchScheduler scheduler =
                Mockito.mock(FlexlbBatchScheduler.class, withSettings().stubOnly());
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        congested = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ipPort(CONGESTED_PORT), workerStatus(CONGESTED_PORT));
        idle = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ipPort(IDLE_PORT), workerStatus(IDLE_PORT));
        // One zero-cost inflight batch per engine engages the backpressure park
        congested.commitBatch(1L, 0L, List.of());
        idle.commitBatch(1L, 0L, List.of());

        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);
        ResourceMeasureFactory resourceMeasureFactory =
                new ResourceMeasureFactory(List.of(new PrefillResourceMeasure(configService)));
        EngineHealthReporter healthReporter =
                Mockito.mock(EngineHealthReporter.class, withSettings().stubOnly());
        strategy = new CostBasedPrefillStrategy(
                engineWorkerStatus, new EmptyCacheAwareService(), resourceMeasureFactory, healthReporter);
    }

    @AfterEach
    void tearDown() {
        if (endpointRegistry != null) {
            endpointRegistry.close();
        }
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    @Test
    void engine_at_threshold_depth_is_excluded_from_candidates() {
        // queueSize == ceil(100×0.8): the ">=" boundary fires the filter
        fillQueue(congested, CONGESTION_THRESHOLD);
        assertEquals(CONGESTION_THRESHOLD, congested.getBatcher().queueSize());

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            assertNotEquals(CONGESTED_PORT, selected.getHttpPort(),
                    "engine at congestion threshold must never be selected while an idle one exists");
        }
    }

    @Test
    void engine_below_threshold_remains_candidate() {
        // 79 < 80: neither engine is congested, both stay in the candidate set
        fillQueue(congested, CONGESTION_THRESHOLD - 1);
        fillQueue(idle, CONGESTION_THRESHOLD - 1);

        boolean congestedSeen = false;
        boolean idleSeen = false;
        for (int i = 0; i < 200; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            congestedSeen |= selected.getHttpPort() == CONGESTED_PORT;
            idleSeen |= selected.getHttpPort() == IDLE_PORT;
        }
        assertTrue(congestedSeen, "below-threshold engine must stay a routing candidate");
        assertTrue(idleSeen, "the sibling engine must stay a routing candidate too");
    }

    @Test
    void all_engines_congested_falls_back_without_error() {
        // Both engines at/beyond the threshold: the survivor set empties and
        // the existing least-loaded fallback must still return one endpoint
        // instead of NO_AVAILABLE_WORKER.
        fillQueue(congested, 100);
        fillQueue(idle, 90);
        assertEquals(100, congested.getBatcher().queueSize());
        assertEquals(90, idle.getBatcher().queueSize());

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess(),
                    "all-congested cluster must fall back to a selectable engine");
            int port = selected.getHttpPort();
            assertTrue(port == CONGESTED_PORT || port == IDLE_PORT);
        }
    }

    @Test
    void disabled_filter_keeps_congested_engine_as_candidate() {
        // Gate off: legacy behavior — a queue-pinned engine remains a plain
        // candidate (regression protection for the switch). Both engines sit
        // at the same depth so the Auto-TPM depth term is symmetric and the
        // disabled filter is the only difference from the excluded case.
        config.setFlexlbCongestedQueueFilterEnabled(false);
        fillQueue(congested, 100);
        fillQueue(idle, 100);

        boolean congestedSeen = false;
        for (int i = 0; i < 200; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            congestedSeen |= selected.getHttpPort() == CONGESTED_PORT;
        }
        assertTrue(congestedSeen,
                "with the filter disabled the congested engine must be selectable again");
    }

    @Test
    void engine_wait_penalty_pushes_high_wait_engine_below_sibling() {
        // 100 engine-side waiting streams × default 20ms = 2000ms score
        // penalty — the only score asymmetry between two otherwise identical
        // engines (batcher queues empty, symmetric waits/predictions). The
        // pendingCount asymmetry (100 vs 0, avg 50) stays under the hotspot
        // bound (100 < 3 × 50) so the penalty term alone must decide.
        reportWaitingQueryLen(congested, 100);

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            assertNotEquals(CONGESTED_PORT, selected.getHttpPort(),
                    "a high engine-side wait must push the engine below its low-wait sibling");
        }
    }

    @Test
    void engine_at_wait_threshold_is_excluded_from_candidates() {
        // waitingQueryLen == 128 (explicit flexlbEngineWaitHardFilterThreshold
        // — the default moved to 256, so the boundary is pinned explicitly
        // instead of riding the default): the ">=" boundary fires the hard
        // filter ("ENGINE_WAIT_FILTERED"). Both batcher queues stay empty so
        // the congested-queue filter does not interfere; pendingCount 128 vs
        // avg 64 stays under hotspot (128 < 3 × 64) so the hard filter is
        // the only deciding factor.
        config.setFlexlbEngineWaitHardFilterThreshold(128);
        reportWaitingQueryLen(congested, 128);

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            assertNotEquals(CONGESTED_PORT, selected.getHttpPort(),
                    "engine at the engine-wait threshold must never be selected while an idle one exists");
        }
    }

    @Test
    void engine_wait_switches_off_restore_legacy_candidate_behavior() {
        // Both engine-wait gates off: the reported wait is ignored entirely
        // and the two engines (queues symmetric-empty) stay interchangeable —
        // both must be selected across iterations (score-tie randomization on
        // by default). Regression protection for the two switches.
        config.setFlexlbEngineWaitPenaltyEnabled(false);
        config.setFlexlbEngineWaitHardFilterEnabled(false);
        reportWaitingQueryLen(congested, 100);

        boolean congestedSeen = false;
        boolean idleSeen = false;
        for (int i = 0; i < 200; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            congestedSeen |= selected.getHttpPort() == CONGESTED_PORT;
            idleSeen |= selected.getHttpPort() == IDLE_PORT;
        }
        assertTrue(congestedSeen,
                "with both engine-wait switches off the high-wait engine must stay selectable");
        assertTrue(idleSeen, "the sibling engine must stay selectable too");
    }

    @Test
    void reported_waiting_query_len_is_clamped_non_negative() {
        reportWaitingQueryLen(congested, -5);
        assertEquals(0L, congested.getReportedWaitingQueryLen(),
                "a negative engine-reported waitingQueryLen must clamp to 0");
    }

    @Test
    void pending_offer_penalty_pushes_route_committed_engine_below_sibling() {
        // R1 (205 pileup): 100 route-committed but not-yet-offered requests
        // × default 50ms = 5000ms Round-1 penalty — the only score asymmetry
        // between two otherwise identical engines. Each select below also
        // records one pending offer on the winner (the buildServerStatus
        // hook), adding at most 50 × 50ms = 2500ms to the idle sibling —
        // still strictly below 5000ms, so the choice stays deterministic.
        for (long requestId = 10_000; requestId < 10_100; requestId++) {
            congested.recordPendingOffer(requestId);
        }

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            assertNotEquals(CONGESTED_PORT, selected.getHttpPort(),
                    "route-committed pending offers must push the engine below its idle sibling");
        }
    }

    @Test
    void pending_offer_penalty_gate_off_restores_tie_and_stops_recording() {
        // Gate off: the reservations are ignored by the score and select no
        // longer records new ones — legacy behavior (regression protection
        // for the switch).
        config.setFlexlbPendingOfferPenaltyEnabled(false);
        for (long requestId = 10_000; requestId < 10_100; requestId++) {
            congested.recordPendingOffer(requestId);
        }

        boolean congestedSeen = false;
        boolean idleSeen = false;
        for (int i = 0; i < 200; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            congestedSeen |= selected.getHttpPort() == CONGESTED_PORT;
            idleSeen |= selected.getHttpPort() == IDLE_PORT;
        }
        assertTrue(congestedSeen,
                "with the gate off the reserved engine must stay selectable");
        assertTrue(idleSeen, "the sibling engine must stay selectable too");
        assertEquals(0, idle.getPendingOfferCount(),
                "with the gate off select must not record pending offers");
    }

    @Test
    void select_records_pending_offer_until_batcher_offer_releases_it() {
        // One select reserves the route→offer blind window on the winner;
        // the batcher offer of the same requestId is the hand-over that
        // releases it.
        ServerStatus selected = strategy.select(context(9_000L), RoleType.PREFILL, null);
        assertTrue(selected.isSuccess());
        PrefillEndpoint winner =
                selected.getHttpPort() == CONGESTED_PORT ? congested : idle;
        assertEquals(1, winner.getPendingOfferCount(),
                "select must reserve the route→offer blind window");

        long now = System.currentTimeMillis();
        assertTrue(winner.getBatcher().tryOffer(item(9_000L, 30, now + 60_000, now, 128)));
        assertEquals(0, winner.getPendingOfferCount(),
                "the batcher offer must release the reservation");
    }

    @Test
    void roll_back_releases_pending_offer_reservation() {
        congested.recordPendingOffer(7_777L);
        strategy.rollBack(congested, 7_777L);
        assertEquals(0, congested.getPendingOfferCount(),
                "a dead route decision must release its pending-offer reservation");
    }

    @Test
    void engine_untracked_penalty_pushes_untracked_busy_engine_below_sibling() {
        // S3 (205 pileup): 100 engine-side active tasks the local ledger does
        // not track × default 20ms = 2000ms Round-1 penalty. The pendingCount
        // asymmetry (100 vs 0, avg 50) stays under the hotspot bound
        // (100 < 3 × 50). The R1 gate is off so the select-side pending-offer
        // recording cannot erode the asymmetry across iterations.
        config.setFlexlbPendingOfferPenaltyEnabled(false);
        reportEngineOnlyRunningTasks(congested, 100);
        assertEquals(100, congested.getEngineUntrackedRequestCount());

        for (int i = 0; i < 50; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            assertNotEquals(CONGESTED_PORT, selected.getHttpPort(),
                    "engine-untracked active tasks must push the engine below its idle sibling");
        }
    }

    @Test
    void engine_untracked_penalty_gate_off_restores_tie() {
        // Both fix-C gates off: untracked engine work is ignored by the score
        // and the two engines stay interchangeable — legacy behavior
        // (regression protection for the switch).
        config.setFlexlbPendingOfferPenaltyEnabled(false);
        config.setFlexlbEngineUntrackedPenaltyEnabled(false);
        reportEngineOnlyRunningTasks(congested, 100);

        boolean congestedSeen = false;
        boolean idleSeen = false;
        for (int i = 0; i < 200; i++) {
            ServerStatus selected = strategy.select(context(i), RoleType.PREFILL, null);
            assertTrue(selected.isSuccess());
            congestedSeen |= selected.getHttpPort() == CONGESTED_PORT;
            idleSeen |= selected.getHttpPort() == IDLE_PORT;
        }
        assertTrue(congestedSeen,
                "with the gate off the untracked-busy engine must stay selectable");
        assertTrue(idleSeen, "the sibling engine must stay selectable too");
    }

    // ==================== helpers ====================

    /**
     * Inject an engine-side sync whose running task details are all unknown
     * to the local batch ledger, driving
     * {@code PrefillEndpoint.engineUntrackedRequestCount} to {@code count} —
     * the S3 scoring input (other-master re-routes / scalar lower bound).
     */
    private static void reportEngineOnlyRunningTasks(PrefillEndpoint endpoint, int count) {
        Map<String, TaskInfo> running = new HashMap<>();
        for (long requestId = 50_000; requestId < 50_000 + count; requestId++) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setBatchId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            running.put(String.valueOf(requestId), task);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    /**
     * Inject an engine-side worker-status sync (the ~20ms path used by
     * {@code PrefillEndpoint.onWorkerStatusUpdate}) carrying the given
     * {@code waitingQueryLen}, mirroring the injection style of
     * {@code PrefillEndpointTest#realPendingCountFallsBackToEngineQueryLengthScalars}.
     */
    private static void reportWaitingQueryLen(PrefillEndpoint endpoint, long waitingQueryLen) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of());
        response.setWaitingQueryLen(waitingQueryLen);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    private static void fillQueue(PrefillEndpoint endpoint, int count) {
        long now = System.currentTimeMillis();
        for (long requestId = 1; requestId <= count; requestId++) {
            assertTrue(endpoint.getBatcher().tryOffer(
                    item(requestId, 30, now + 60_000, now, 128)));
        }
    }

    private static BatchItem item(long requestId, int priority, long deadlineMs,
                                  long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs, deadlineMs));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }

    private BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1_024L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private static String ipPort(int port) {
        return "127.0.0.1:" + port;
    }

    private static WorkerStatus workerStatus(int port) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setRole(RoleType.PREFILL);
        status.setGroup("congestion-test");
        status.setAlive(true);
        status.setRunningTaskList(new java.util.HashMap<>());
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(256);
        cacheStatus.setAvailableKvCache(1_000_000L);
        status.setCacheStatus(cacheStatus);
        return status;
    }

    private static final class EmptyCacheAwareService implements CacheAwareService {
        @Override
        public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
                                                        RoleType roleType,
                                                        String group) {
            return Map.of();
        }

        @Override
        public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
            return null;
        }
    }
}
