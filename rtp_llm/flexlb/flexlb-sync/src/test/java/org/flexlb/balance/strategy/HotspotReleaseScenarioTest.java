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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * End-to-end hotspot release scenarios over real endpoints and committed
 * status reports: a sustained HOTSPOT opens the line against the
 * fleet-relative setpoint, drifts the request to balanced ranking, and
 * re-pins after relief; a transient RAMP burst below the sustained window,
 * and a deep-queue but balanced large-batch fleet, must never open the line.
 */
class HotspotReleaseScenarioTest {

    private static final String LEADER_IP = "10.0.0.2";
    private static final String IDLE_IP = "10.0.0.1";

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private PrefillEndpoint leader;
    private PrefillEndpoint idle;
    private CostBasedPrefillStrategy strategy;
    private CacheAwareService cache;
    private long requestId;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        // The hotspot's 30 waiting queries count as unknown Engine work and
        // would trip both the queue-admission gate (threshold 10) and the
        // per-worker route-slot limit (2); relax both so the scenario
        // exercises the release line itself, not the admission gates.
        RoutingConfig.TokenBalanceConfig tokenBalance =
                new RoutingConfig.TokenBalanceConfig();
        tokenBalance.setEngineQueueAdmissionThreshold(null);
        config.getRouter().getRoles().getPrefill().setTokenBalance(tokenBalance);
        config.getDispatcher().setMaxInflightPerPrefillWorker(100);
        config.getRouter().getRoles().getPrefill()
                .getHotspotRelease().setEnabled(true);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);

        // The hot leader reports a sustained waiting queue of 30; the idle
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
                new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
        requestId = 1_000L;
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    /**
     * HOTSPOT: the line only climbs after the sustained window, the released
     * request drifts off the congested leader, and relief re-pins it once
     * the leader drains.
     */
    @Test
    void sustainedHotspotDriftsThenRePinsAfterRelief() {
        PrefillTokenWorkload.Reservation backlog = leader.planTokenWork(1_500L);
        try {
            assertEquals(LEADER_IP, selectedIp(), "closed line must pin the cache leader");

            // 12s of sustained waiting-queue pressure: 5s gate, then climbing
            // against the fleet-relative setpoint (median 15 -> setpoint 22,
            // so 1.2*(30-22) = 9.6pp/s for the remaining 7s).
            driveGovernor(12L);
            assertTrue(leader.hotspotReleaseLine() > 50.0,
                    "sustained pressure with an idle worker must open the line");
            assertEquals(IDLE_IP, selectedIp(),
                    "a released leader must lose to lower uncached work");

            // The leader drains: relief decays the line back to closed.
            republishWaiting(leader, 0L);
            driveGovernor(30L);
            assertEquals(0.0, leader.hotspotReleaseLine(), 1e-9);
            assertEquals(LEADER_IP, selectedIp(),
                    "a drained leader must win the pin again");
        } finally {
            backlog.close();
        }
    }

    /**
     * RAMP: balanced-fleet transient queue bumps below the sustained window
     * must never open the line, so pinning is preserved exactly.
     */
    @Test
    void transientRampPressureNeverOpensTheLine() {
        PrefillTokenWorkload.Reservation backlog = leader.planTokenWork(1_500L);
        try {
            // 4s of pressure: below the 5s sustained window.
            driveGovernor(4L);
            assertEquals(0.0, leader.hotspotReleaseLine(), 1e-9,
                    "a transient burst must not climb the line");
            assertEquals(LEADER_IP, selectedIp(),
                    "without a real hotspot the pin must stay untouched");
        } finally {
            backlog.close();
        }
    }

    /**
     * Large-batch fleet: every worker runs a deep, healthy queue (leader 40,
     * peer 30). The fleet-relative setpoint (median 35 -> 52) sits above the
     * leader's own depth, so sustained pressure must not open the line —
     * congestion is judged against the fleet, not the absolute floor.
     */
    @Test
    void healthyBusyFleetNeverOpensTheLine() {
        republishWaiting(leader, 40L);
        republishWaiting(idle, 30L);
        PrefillTokenWorkload.Reservation backlog = leader.planTokenWork(1_500L);
        try {
            driveGovernor(12L);
            assertEquals(0.0, leader.hotspotReleaseLine(), 1e-9,
                    "a busy-but-balanced fleet must stay pinned");
            assertEquals(LEADER_IP, selectedIp(),
                    "the cache leader must keep the pin under healthy depth");
        } finally {
            backlog.close();
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

    /** Advance the pool governor over injected timestamps ending at now. */
    private void driveGovernor(long seconds) {
        long now = System.currentTimeMillis();
        long rounds = seconds * 10L;
        for (long round = 1; round <= rounds; round++) {
            long at = now - rounds * 100L + round * 100L;
            long setpoint = CostBasedPrefillStrategy.hotspotReleaseSetpoint(
                    config.getRouter().getRoles().getPrefill().getHotspotRelease(),
                    new long[] {
                            leader.getStatus().committedEngineObservation().waitingQueryLen(),
                            idle.getStatus().committedEngineObservation().waitingQueryLen()});
            leader.governHotspotRelease(setpoint, true, at);
            idle.governHotspotRelease(setpoint, true, at);
        }
    }

    /** Republish one endpoint's waiting-queue depth at a fresh status version. */
    private static void republishWaiting(PrefillEndpoint endpoint, long waitingQueryLen) {
        WorkerStatus status = endpoint.getStatus();
        WorkerStatusResponse response = StrategyTestSupport.response(
                RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setWaitingQueryLen(waitingQueryLen);
        StrategyTestSupport.publish(status, response);
    }

    private void publish(RoleType role, String address, WorkerStatus source) {
        StrategyTestSupport.publishEndpoint(registry, role, address, source);
    }

    private static WorkerStatus worker(String ip, long waitingQueryLen) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
        WorkerStatusResponse response = StrategyTestSupport.response(
                RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setWaitingQueryLen(waitingQueryLen);
        StrategyTestSupport.publish(status, response);
        return status;
    }
}
