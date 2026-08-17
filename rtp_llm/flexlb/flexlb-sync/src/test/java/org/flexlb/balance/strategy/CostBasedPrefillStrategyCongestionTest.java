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
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

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

    // ==================== helpers ====================

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
