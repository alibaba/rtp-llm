package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.EngineType;
import org.flexlb.exception.ServiceDiscoveryException;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.RateLimitedWarn;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EngineSyncRunnerTest {

    private final String modelName = "test-model";

    private Map<String, WorkerStatus> workerStatusMap;

    @Mock
    private WorkerAddressService workerAddressService;

    @Mock
    private ExecutorService statusCheckExecutor;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private EngineGrpcService engineGrpcService;

    private final RoleType roleType = RoleType.PDFUSION;

    @Mock
    private CacheAwareService localKvCacheAwareManager;

    private final long syncRequestTimeoutMs = 5000L;

    @Mock
    private LongAdder syncCount;

    private final long syncEngineStatusInterval = 20L;

    private EngineSyncRunner engineSyncRunner;

    /** Fresh per test so the grace clock never leaks across methods (it used to be a static map). */
    private Map<String, Long> lastDiscoverySuccessUs;

    @BeforeEach
    void setUp() {
        workerStatusMap = new ConcurrentHashMap<>();
        lastDiscoverySuccessUs = new ConcurrentHashMap<>();

        engineSyncRunner = newRunner(EngineType.LLM);
    }

    private EngineSyncRunner newRunner(EngineType engineType) {
        return newRunner(engineType, 300_000L);
    }

    private EngineSyncRunner newRunner(EngineType engineType, long discoveryFailureGraceMs) {
        return new EngineSyncRunner(
                modelName,
                workerStatusMap,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                engineType,
                discoveryFailureGraceMs,
                lastDiscoverySuccessUs,
                new RateLimitedWarn(1, TimeUnit.SECONDS)
        );
    }

    private WorkerHost host(String ip, int port) {
        return new WorkerHost(ip, port, port + 1, port + 5, "site-a", "group-a");
    }

    @Test
    void should_not_submit_tasks_when_no_workers_exist() {
        // Execute
        engineSyncRunner.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void embedding_engine_marks_workers_alive_without_probing() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950), host("10.0.0.2", 23950)));

        newRunner(EngineType.EMBEDDING).run();

        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
        assertEquals(2, workerStatusMap.size());
        WorkerStatus status = workerStatusMap.get("10.0.0.1:23950");
        assertNotNull(status);
        assertTrue(status.isAlive());
        assertEquals("group-a", status.getGroup());
        assertEquals("site-a", status.getSite());
        assertEquals(roleType.getCode(), status.getRole(),
                "role must use the RoleType code format that RoleType.matches() consumers expect");
        assertTrue(status.getStatusLastUpdateTime().get() > 0);
    }

    @Test
    void embedding_engine_removes_worker_dropped_by_discovery() {
        WorkerStatus stale = new WorkerStatus();
        stale.setIp("10.0.0.9");
        stale.setPort(23950);
        stale.setAlive(true);
        stale.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", stale);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)));

        newRunner(EngineType.EMBEDDING).run();

        assertFalse(workerStatusMap.containsKey("10.0.0.9:23950"));
        assertTrue(workerStatusMap.containsKey("10.0.0.1:23950"));
    }

    @Test
    void empty_discovery_result_never_wipes_a_known_fleet() {
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.9");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", alive);

        // A prior successful round establishes the discovery-grace baseline, then discovery hands
        // back an empty list — which a client that swallows a failed lookup reports exactly like a
        // fleet that scaled to zero. The known fleet must survive it.
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenReturn(List.of());

        EngineSyncRunner runner = newRunner(EngineType.EMBEDDING);
        runner.run();
        long beforeUs = System.nanoTime() / 1000;
        runner.run();

        assertTrue(alive.isAlive(),
                "an empty discovery list is indistinguishable from a swallowed lookup failure, so it "
                        + "must not mark a known embedding fleet dead");
        assertTrue(workerStatusMap.containsKey("10.0.0.9:23950"),
                "an empty discovery list must not physically remove known workers either");
        assertTrue(alive.getStatusLastUpdateTime().get() >= beforeUs,
                "within the grace window the retained workers' staleness clock is refreshed, so the "
                        + "independent ExpirationCleaner does not evict a fleet discovery cannot confirm");
    }

    @Test
    void empty_discovery_result_beyond_grace_lets_workers_age_out() {
        EngineSyncRunner zeroGraceRunner = newRunner(EngineType.EMBEDDING, 0L);
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.9");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", alive);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenReturn(List.of());

        zeroGraceRunner.run();
        // The successful round stamps the clock itself (embedding liveness comes from the discovery
        // list); reset it so the assertion observes only what the empty round does.
        alive.getStatusLastUpdateTime().set(1L);
        zeroGraceRunner.run();

        assertEquals(1L, alive.getStatusLastUpdateTime().get(),
                "once the empty results outlast the grace window the staleness clock stops being "
                        + "refreshed, so a fleet that genuinely scaled to zero still ages out");
    }

    @Test
    void embedding_engine_still_marks_dead_a_worker_dropped_from_a_non_empty_list() {
        WorkerStatus dropped = new WorkerStatus();
        dropped.setIp("10.0.0.9");
        dropped.setPort(23950);
        dropped.setAlive(true);
        dropped.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        workerStatusMap.put("10.0.0.9:23950", dropped);

        // Partial shrinkage is unambiguous — discovery answered, and this worker is not in the
        // answer. Only the all-or-nothing empty result is treated as untrustworthy.
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)));

        newRunner(EngineType.EMBEDDING).run();

        assertFalse(dropped.isAlive(),
                "embedding has no probe fallback: a worker gone from a non-empty discovery list must "
                        + "stop being routable immediately");
    }

    @Test
    void rejected_status_submit_resets_in_progress_flags() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)));
        when(statusCheckExecutor.submit(any(Runnable.class)))
                .thenThrow(new java.util.concurrent.RejectedExecutionException("queue full"));

        newRunner(EngineType.LLM).run();

        WorkerStatus ws = workerStatusMap.get("10.0.0.1:23950");
        assertNotNull(ws);
        assertFalse(ws.getStatusCheckInProgress().get(),
                "a rejected submit must reset the in-progress flag so the worker is retried next round");
        assertFalse(ws.getCacheCheckInProgress().get(),
                "a rejected cache-check submit must also reset its flag");
    }

    @Test
    void embedding_engine_marks_worker_dead_immediately_before_removal_threshold() {
        WorkerStatus stale = new WorkerStatus();
        stale.setIp("10.0.0.9");
        stale.setPort(23950);
        stale.setAlive(true);
        // Fresh update time: within max(3 * interval, 1s), so physical removal must not
        // trigger yet — but routability must drop right away.
        stale.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        workerStatusMap.put("10.0.0.9:23950", stale);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)));

        newRunner(EngineType.EMBEDDING).run();

        assertTrue(workerStatusMap.containsKey("10.0.0.9:23950"),
                "physical removal stays thresholded to tolerate discovery flaps");
        assertFalse(stale.isAlive(),
                "a worker missing from discovery must be non-routable immediately, not after the removal threshold");
        assertTrue(workerStatusMap.get("10.0.0.1:23950").isAlive());
    }

    @Test
    void embedding_engine_keeps_workers_alive_when_discovery_fails() {
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.9");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        workerStatusMap.put("10.0.0.9:23950", alive);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenThrow(new ServiceDiscoveryException(
                        BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, "vipserver down", null));

        newRunner(EngineType.EMBEDDING).run();

        assertTrue(alive.isAlive(),
                "a failed discovery round must keep the previous alive state — only a successful "
                        + "round may mark embedding workers dead");
        assertTrue(workerStatusMap.containsKey("10.0.0.9:23950"));
    }

    @Test
    void embedding_discovery_failure_within_grace_refreshes_staleness_clock_so_workers_survive_expiration() {
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.9");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", alive);

        // A prior successful round establishes the discovery-grace baseline; the immediately
        // following failure is well within the grace window. Embedding workers have no probe, so
        // the clock refresh is the only thing keeping ExpirationCleaner from evicting the fleet.
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenThrow(new ServiceDiscoveryException(
                        BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, "vipserver down", null));

        EngineSyncRunner runner = newRunner(EngineType.EMBEDDING);
        runner.run();
        long beforeUs = System.nanoTime() / 1000;
        runner.run();

        assertTrue(workerStatusMap.containsKey("10.0.0.9:23950"));
        assertTrue(alive.getStatusLastUpdateTime().get() >= beforeUs,
                "a discovery failure within the grace window must refresh the retained embedding workers' "
                        + "staleness clock so the independent ExpirationCleaner does not evict a healthy fleet");
    }

    @Test
    void embedding_worker_already_marked_dead_is_not_kept_fresh_during_the_gap() {
        WorkerStatus dead = new WorkerStatus();
        dead.setIp("10.0.0.9");
        dead.setPort(23950);
        dead.setAlive(false);
        dead.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", dead);
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.1");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.1:23950", alive);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)))
                .thenThrow(new ServiceDiscoveryException(
                        BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, "vipserver down", null));

        EngineSyncRunner runner = newRunner(EngineType.EMBEDDING);
        runner.run();
        runner.run();

        assertEquals(1L, dead.getStatusLastUpdateTime().get(),
                "a worker an earlier authoritative round already marked dead must age out normally — "
                        + "the gap ride-out only protects workers that were alive when discovery broke");
    }

    @Test
    void llm_empty_discovery_list_within_grace_keeps_probing_known_workers_without_touching_membership() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenReturn(List.of());

        engineSyncRunner.run();
        Long baselineUs = lastDiscoverySuccessUs.get(modelName + "/" + roleType);
        assertNotNull(baselineUs, "the successful round must stamp the grace baseline");

        WorkerStatus known = workerStatusMap.get("10.0.0.9:23950");
        assertNotNull(known);
        // The mocked executor never runs the probes, so complete them by hand — otherwise the
        // in-progress flags from the first round would make the gap round skip its submissions.
        known.getStatusCheckInProgress().set(false);
        known.getCacheCheckInProgress().set(false);
        known.getStatusLastUpdateTime().set(1L);

        engineSyncRunner.run();

        verify(statusCheckExecutor, org.mockito.Mockito.times(4)).submit(any(Runnable.class));
        assertTrue(workerStatusMap.containsKey("10.0.0.9:23950"),
                "membership stays frozen during the gap — no removals");
        assertEquals(1, workerStatusMap.size(), "membership stays frozen during the gap — no additions");
        assertEquals(baselineUs, lastDiscoverySuccessUs.get(modelName + "/" + roleType),
                "an empty result is an outage, not a success — it must not restart the grace clock");
        assertEquals(1L, known.getStatusLastUpdateTime().get(),
                "LLM workers get no artificial clock refresh: the probes themselves refresh it on "
                        + "success, so a worker that died mid-outage fails its probes and ages out");
    }

    @Test
    void llm_discovery_failure_within_grace_keeps_probing_instead_of_refreshing_clocks() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenThrow(new ServiceDiscoveryException(
                        BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, "vipserver down", null));

        engineSyncRunner.run();
        WorkerStatus known = workerStatusMap.get("10.0.0.9:23950");
        assertNotNull(known);
        known.getStatusCheckInProgress().set(false);
        known.getCacheCheckInProgress().set(false);
        known.getStatusLastUpdateTime().set(1L);

        engineSyncRunner.run();

        verify(statusCheckExecutor, org.mockito.Mockito.times(4)).submit(any(Runnable.class));
        assertEquals(1L, known.getStatusLastUpdateTime().get(),
                "gRPC probing is the LLM health signal during a discovery gap — the sync loop must "
                        + "not overwrite the staleness clock the probes maintain");
    }

    @Test
    void discovery_failure_beyond_grace_stops_refreshing_so_workers_can_age_out() {
        EngineSyncRunner zeroGraceRunner = newRunner(EngineType.LLM, 0L);
        WorkerStatus alive = new WorkerStatus();
        alive.setIp("10.0.0.9");
        alive.setPort(23950);
        alive.setAlive(true);
        alive.getStatusLastUpdateTime().set(1L);
        workerStatusMap.put("10.0.0.9:23950", alive);

        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.9", 23950)))
                .thenThrow(new ServiceDiscoveryException(
                        BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, "vipserver down", null));

        zeroGraceRunner.run();
        zeroGraceRunner.run();

        assertEquals(1L, alive.getStatusLastUpdateTime().get(),
                "once a discovery outage outlasts the grace window the staleness clock must stop being "
                        + "refreshed, so ExpirationCleaner can eventually evict workers a broken discovery "
                        + "can no longer confirm");
    }

    @Test
    void embedding_engine_skips_variance_reporting() {
        // Embedding workers are never probed, so stepLatency/runningQueueTime stay 0 and
        // the variance is identically 0 — reporting it would only pollute monitoring.
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950), host("10.0.0.2", 23950)));

        newRunner(EngineType.EMBEDDING).run();

        verify(engineHealthReporter, never())
                .reportLatencyMetric(any(), any(), org.mockito.Mockito.anyDouble(), org.mockito.Mockito.anyDouble());
    }

    @Test
    void llm_engine_still_submits_probe_runners() {
        when(workerAddressService.getEngineWorkerList(modelName, roleType))
                .thenReturn(List.of(host("10.0.0.1", 23950)));

        engineSyncRunner.run();

        verify(statusCheckExecutor, org.mockito.Mockito.times(2)).submit(any(Runnable.class));
        WorkerStatus status = workerStatusMap.get("10.0.0.1:23950");
        assertNotNull(status);
        assertFalse(status.isAlive());
    }
}
