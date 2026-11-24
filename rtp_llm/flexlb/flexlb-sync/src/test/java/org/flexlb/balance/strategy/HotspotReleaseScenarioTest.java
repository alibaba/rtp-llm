package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Hotspot selection and recovery with real endpoints and committed status reports. */
class HotspotReleaseScenarioTest {

    private static final String LEADER_IP = "10.0.0.2";
    private static final String IDLE_IP = "10.0.0.1";

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private PrefillEndpoint leader;
    private PrefillEndpoint idle;
    private final HotspotRedirectLimiter budget = new HotspotRedirectLimiter(() -> 0L);
    private CostBasedPrefillStrategy strategy;
    private CacheAwareService cache;
    private long requestId;
    private long governorTimeMs;
    private final PrefillHotspots hotspots = new PrefillHotspots(() -> MILLISECONDS.toNanos(governorTimeMs));

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        // The hotspot's 30 running streams count as unknown Engine work and
        // would trip both the queue-admission gate (threshold 10) and the
        // per-worker route-slot limit (2); relax both so the scenario
        // exercises hotspot detection, not the admission gates.
        RoutingConfig.TokenBalanceConfig tokenBalance =
                new RoutingConfig.TokenBalanceConfig();
        tokenBalance.setEngineQueueAdmissionThreshold(null);
        config.getRouter().getRoles().getPrefill().setTokenBalance(tokenBalance);
        config.getDispatcher().setMaxInflightPerPrefillWorker(100);
        config.getRouter().getRoles().getPrefill()
                .getHotspotRelease().setEnabled(true);
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setSustainedPressureMs(5_000L);
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);

        // The hot leader reports 30 running streams; the idle
        // worker reports none. Only the leader holds the session prefix.
        WorkerStatus leaderSource = worker(LEADER_IP, 30L);
        publish(RoleType.PREFILL, LEADER_IP + ":8080", leaderSource);
        publish(RoleType.PREFILL, IDLE_IP + ":8080", worker(IDLE_IP, 0L));
        Map<String, PrefillEndpoint> prefillEndpoints = registry.snapshotPrefillEndpoints();
        leader = prefillEndpoints.get(LEADER_IP + ":8080");
        idle = prefillEndpoints.get(IDLE_IP + ":8080");

        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), eq(RoleType.PREFILL), any())).thenReturn(
                Map.of(LEADER_IP + ":8080", 5));
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class), budget, hotspots);
        requestId = 1_000L;
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void sustainedHotspotDriftsThenRePinsOnFirstSampleAfterRelief() {
        try (var backlog = leader.planTokenWork(1_500L)) {
            assertEquals(LEADER_IP, selectedIp());
            driveGovernor(6L);
            assertTrue(sampledHotspots().contains(leader));
            assertEquals(IDLE_IP, selectedIp());

            republishRunning(leader, 0L);
            governorTimeMs += 100L;
            assertFalse(sampledHotspots().contains(leader));
            assertEquals(LEADER_IP, selectedIp());
        }
    }

    @Test
    void transientPressureBelowConfiguredDurationKeepsAffinity() {
        try (var backlog = leader.planTokenWork(1_500L)) {
            driveGovernor(4L);
            assertFalse(sampledHotspots().contains(leader));
            assertEquals(LEADER_IP, selectedIp());
        }
    }

    @Test
    void healthyBusyFleetKeepsAffinity() {
        republishRunning(leader, 40L);
        republishRunning(idle, 30L);
        try (var backlog = leader.planTokenWork(1_500L)) {
            driveGovernor(12L);
            assertFalse(sampledHotspots().contains(leader));
            assertEquals(LEADER_IP, selectedIp());
        }
    }

    /** One selection round; the request is a 50% hit on the leader. */
    private String selectedIp() {
        Request request = new Request();
        request.setRequestId(requestId++);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setCacheKeyBlockSize(100L);
        request.setBlockCacheKeys(List.of());
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        PlacementResult<SelectedRole, ?> result =
                strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (SelectedRole selected = result.value()) {
            return selected.serverStatus().getServerIp();
        }
    }

    /** Drive production governor rounds with a monotonic test clock. */
    private void driveGovernor(long seconds) {
        for (long round = 0; round < seconds * 10L; round++) {
            governorTimeMs += 100L;
            sampledHotspots();
        }
    }

    private java.util.Set<PrefillEndpoint> sampledHotspots() {
        return hotspots.sample(RoleType.PREFILL, null, registry.prefillRoutingSnapshot(RoleType.PREFILL),
                config.getRouter().getRoles().getPrefill().getHotspotRelease());
    }

    /** Republish one endpoint's running-stream count at a fresh status version. */
    private static void republishRunning(PrefillEndpoint endpoint, long runningStreams) {
        WorkerStatus status = endpoint.getStatus();
        WorkerStatusResponse response = StrategyTestSupport.response(
                RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setRunningQueryLen(runningStreams);
        Runnable publishEvents;
        status.lock.lock();
        try {
            var observation = WorkerStatusTestFixtures.observation(status, response);
            publishEvents = endpoint.applyPreparedStatus(status, WorkerStatusTestFixtures.prepare(status, observation));
        } finally {
            status.lock.unlock();
        }
        publishEvents.run();
    }

    private void publish(RoleType role, String address, WorkerStatus source) {
        StrategyTestSupport.publishEndpoint(registry, role, address, source);
    }

    private static WorkerStatus worker(String ip, long runningStreams) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
        WorkerStatusResponse response = StrategyTestSupport.response(
                RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setRunningQueryLen(runningStreams);
        StrategyTestSupport.publish(status, response);
        return status;
    }
}
