package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerLifecycleState;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class EngineSyncRunnerTest {

    private final String modelName = "test-model";

    @Mock
    private Map<String, WorkerStatus> workerStatusMap;

    @Mock
    private WorkerAddressService workerAddressService;

    @Mock
    private ExecutorService statusCheckExecutor;

    @Mock
    private EngineHealthReporter engineHealthReporter;

    @Mock
    private EngineGrpcService engineGrpcService;

    @Mock
    private RoleType roleType;

    @Mock
    private CacheAwareService localKvCacheAwareManager;

    private final long syncRequestTimeoutMs = 5000L;

    @Mock
    private LongAdder syncCount;

    private final long syncEngineStatusInterval = 20L;

    private EngineSyncRunner engineSyncRunner;

    @BeforeEach
    void setUp() {
        workerStatusMap = new ConcurrentHashMap<>();

        engineSyncRunner = new EngineSyncRunner(
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
                null,
                null
        );
    }

    @Test
    void should_not_submit_tasks_when_no_workers_exist() {
        // Execute
        engineSyncRunner.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_handle_null_worker_status_gracefully() {
        // Setup - create runner with null map
        EngineSyncRunner runnerWithNullMap = new EngineSyncRunner(
                modelName,
                new ConcurrentHashMap<>(),
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                roleType,
                localKvCacheAwareManager,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                null,
                null
        );

        // Execute
        runnerWithNullMap.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_start_new_worker_expiration_window_at_discovery_time() {
        String ipPort = "127.0.0.1:8080";
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerStatusMap, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.VIT,
                localKvCacheAwareManager, syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, null, null);

        runner.run();

        WorkerStatus status = workerStatusMap.get(ipPort);
        assertTrue(status.getDiscoveryLastSeenTime().get() > 0);
        assertEquals(-1L, status.getStatusLastUpdateTime().get());
        assertEquals(WorkerLifecycleState.PROBING, status.getLifecycleState());
    }

    @Test
    void should_remove_status_and_endpoint_when_service_discovery_is_empty() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointRegistry registry = new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.PREFILL);
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setAlive(true);
        status.getDiscoveryLastSeenTime().set(System.nanoTime() / 1000 - 2_000_000L);
        status.getDiscoveryUpdateIntervalUs().set(20_000L);
        statuses.put(ipPort, status);
        registry.ensureEndpoint(RoleType.PREFILL, ipPort, status);
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of());

        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, statuses, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                localKvCacheAwareManager, syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, null, registry);
        runner.run();

        assertFalse(status.isAlive());
        assertFalse(statuses.containsKey(ipPort));
        assertNull(registry.get(RoleType.PREFILL, ipPort));
        registry.close();
    }

    @Test
    void newer_discovery_observation_fences_overlapping_missing_retirement()
            throws InterruptedException {
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EndpointRegistry registry = new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.PREFILL);
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setAlive(true);
        status.getDiscoveryLastSeenTime().set(System.nanoTime() / 1000 - 2_000_000L);
        status.getDiscoveryUpdateIntervalUs().set(20_000L);
        statuses.put(ipPort, status);
        registry.ensureEndpoint(RoleType.PREFILL, ipPort, status);
        Mockito.when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of());
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, statuses, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.PREFILL,
                localKvCacheAwareManager, syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, null, registry);

        Thread missingRound = new Thread(runner, "missing-discovery-round");
        status.lock.lock();
        try {
            missingRound.start();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (!status.lock.hasQueuedThread(missingRound) && System.nanoTime() < deadline) {
                Thread.onSpinWait();
            }
            assertTrue(status.lock.hasQueuedThread(missingRound),
                    "missing round did not reach the generation retirement check");
            assertTrue(status.recordDiscoverySeen(System.nanoTime() / 1000));
        } finally {
            status.lock.unlock();
        }
        missingRound.join(TimeUnit.SECONDS.toMillis(2));

        assertFalse(missingRound.isAlive());
        assertSame(status, statuses.get(ipPort));
        assertTrue(status.isReady());
        assertSame(status, registry.get(RoleType.PREFILL, ipPort).getStatus());
        registry.close();
    }

    @Test
    void should_publish_discovered_role_before_status_check_is_submitted() {
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName,
                workerStatusMap,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                localKvCacheAwareManager,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                null,
                null
        );
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(new WorkerHost("127.0.0.1", 61000)));

        runner.run();

        assertEquals(RoleType.PREFILL, workerStatusMap.get("127.0.0.1:61000").getRole());
        verify(statusCheckExecutor, times(1)).submit(any(Runnable.class));
    }

    @Test
    void repeated_sync_keeps_same_probing_generation_and_never_publishes_endpoint() {
        String ipPort = "127.0.0.1:8080";
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(WorkerHost.of("127.0.0.1", 8080)));
        ConfigService configService = Mockito.mock(ConfigService.class);
        EndpointRegistry registry = new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName, workerStatusMap, workerAddressService, statusCheckExecutor,
                engineHealthReporter, engineGrpcService, RoleType.VIT,
                localKvCacheAwareManager, syncRequestTimeoutMs, syncCount,
                syncEngineStatusInterval, null, registry);

        runner.run();
        WorkerStatus firstGeneration = workerStatusMap.get(ipPort);
        runner.run();

        assertSame(firstGeneration, workerStatusMap.get(ipPort));
        assertEquals(WorkerLifecycleState.PROBING, firstGeneration.getLifecycleState());
        assertFalse(firstGeneration.isAlive());
        assertNull(registry.get(RoleType.VIT, ipPort));
        verify(statusCheckExecutor, times(1)).submit(any(Runnable.class));
        registry.close();
    }
}
