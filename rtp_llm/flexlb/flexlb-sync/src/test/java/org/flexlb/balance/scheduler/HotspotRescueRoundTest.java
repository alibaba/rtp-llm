package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Decision-loop rescue rounds: gating, filters, caps and cadence. */
class HotspotRescueRoundTest {
    private static final String SOURCE_ADDRESS = "10.0.0.1:8080";

    private FlexlbConfig config;
    private RoutingConfig.HotspotRescueConfig rescue;
    private DefaultRouter router;
    private RequestRegistry lifecycle;
    private PrefillEndpoint source;
    private GlobalQueueCoordinator coordinator;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        RoutingConfig.TokenBalanceConfig balance = new RoutingConfig.TokenBalanceConfig();
        rescue = new RoutingConfig.HotspotRescueConfig();
        rescue.setMigrationIntervalMs(3_600_000L);
        balance.setHotspotRescue(rescue);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        router = mock(DefaultRouter.class);
        lifecycle = mock(RequestRegistry.class);
        when(lifecycle.migrateQueuedPrefillRoute(any(), any())).thenReturn(true);
        source = mock(PrefillEndpoint.class);
        when(source.ipPort()).thenReturn(SOURCE_ADDRESS);
    }

    @AfterEach
    void close() {
        if (coordinator != null) {
            coordinator.close();
        }
    }

    private void startCoordinator() {
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        coordinator = new GlobalQueueCoordinator(service, router,
                mock(BatchSchedulerReporter.class), mock(EvictionManager.class),
                lifecycle, new PlacementAvailability());
    }

    private void stubOnePool(boolean idleTarget) {
        when(router.hotspotRescuePools(any())).thenReturn(List.of(
                new CostBasedPrefillStrategy.HotspotRescuePool(null, List.of(source), idleTarget)));
        when(router.hasCacheEligibleRescueTarget(any(), any(), any())).thenReturn(true);
    }

    private ScheduledRequest item(long id, long seqLen, long ageMs) {
        BalanceContext context = new BalanceContext(config);
        Request request = new Request();
        request.setRequestId(id);
        request.setSeqLen(seqLen);
        context.setRequest(request);
        context.setStartTime(System.currentTimeMillis() - ageMs);
        context.setWorkerEnqueueSequence(id);
        ScheduledRequest candidate = mock(ScheduledRequest.class);
        when(candidate.requestId()).thenReturn(id);
        when(candidate.seqLen()).thenReturn(seqLen);
        when(candidate.ctx()).thenReturn(context);
        when(candidate.future()).thenReturn(new CompletableFuture<>());
        when(candidate.requestExpired(anyLong())).thenReturn(false);
        return candidate;
    }

    private void queuedItems(ScheduledRequest... items) {
        when(source.captureQueueSnapshot()).thenReturn(
                new WorkerBatcher.QueueSnapshot(SOURCE_ADDRESS, 1L, List.of(items)));
    }

    @Test
    void disabledByDefaultNeverRunsARescueRound() {
        // tokenBalance without hotspotRescue: the round must not even consult pools.
        config.getRouter().getRoles().getPrefill().getTokenBalance().setHotspotRescue(null);
        startCoordinator();
        verify(router, after(300).never()).hotspotRescuePools(any());
        verify(lifecycle, after(100).never()).migrateQueuedPrefillRoute(any(), any());
    }

    @Test
    void aRoundWithoutIdleTargetsNeverWithdraws() {
        stubOnePool(false);
        queuedItems(item(1, 100, 6_000L));
        startCoordinator();
        verify(router, timeout(1_000)).hotspotRescuePools(any());
        verify(lifecycle, after(300).never()).migrateQueuedPrefillRoute(any(), any());
    }

    @Test
    void theRoundFiltersYoungLargeExpiredDoneAndAlreadyMigratedRequests() {
        stubOnePool(true);
        ScheduledRequest young = item(1, 100, 1_000L);
        ScheduledRequest large = item(2, 32_769, 6_000L);
        ScheduledRequest expired = item(3, 100, 6_000L);
        when(expired.requestExpired(anyLong())).thenReturn(true);
        ScheduledRequest done = item(4, 100, 6_000L);
        done.future().complete(new org.flexlb.dao.loadbalance.Response());
        ScheduledRequest migrated = item(5, 100, 6_000L);
        migrated.ctx().setPrefillMigrationSource("10.0.0.9:8080");
        ScheduledRequest eligible = item(6, 100, 6_000L);
        queuedItems(young, large, expired, done, migrated, eligible);
        startCoordinator();
        verify(lifecycle, timeout(1_000)).migrateQueuedPrefillRoute(source, eligible);
        verify(lifecycle, after(300).times(1)).migrateQueuedPrefillRoute(any(), any());
    }

    @Test
    void oneRoundMigratesAtMostMaxMigrationsPerRoundShortestFirst() {
        rescue.setMaxMigrationsPerRound(1);
        stubOnePool(true);
        ScheduledRequest longer = item(1, 400, 6_000L);
        ScheduledRequest shortest = item(2, 100, 6_000L);
        queuedItems(longer, shortest);
        startCoordinator();
        verify(lifecycle, timeout(1_000)).migrateQueuedPrefillRoute(source, shortest);
        verify(lifecycle, after(300).times(1)).migrateQueuedPrefillRoute(any(), any());
    }

    @Test
    void roundsRepeatAtTheMigrationIntervalWithoutAccumulating() {
        rescue.setMigrationIntervalMs(200L);
        stubOnePool(true);
        queuedItems(item(1, 100, 6_000L));
        startCoordinator();
        // Only one withdrawal per round even though the loop wakes every 100ms.
        verify(lifecycle, timeout(2_000).times(2)).migrateQueuedPrefillRoute(eq(source), any());
    }
}
