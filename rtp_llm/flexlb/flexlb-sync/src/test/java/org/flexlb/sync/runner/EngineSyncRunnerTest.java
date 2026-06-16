package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.address.WorkerAddressService.WorkerDiscoveryResult;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertFalse;
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

    private final RoleType roleType = RoleType.PREFILL;

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
                syncEngineStatusInterval
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
                syncEngineStatusInterval
        );

        // Execute
        runnerWithNullMap.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_mark_cached_worker_unavailable_when_service_discovery_is_empty() {
        WorkerStatus staleWorker = freshAliveWorker("10.0.0.1", 8080);
        workerStatusMap.put("10.0.0.1:8080", staleWorker);
        when(workerAddressService.getEngineWorkerDiscoveryResult(modelName, roleType))
                .thenReturn(WorkerDiscoveryResult.success(List.of()));

        engineSyncRunner.run();

        assertFalse(staleWorker.isAlive());
        assertTrue(workerStatusMap.containsKey("10.0.0.1:8080"));
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_directly_check_cached_worker_when_service_discovery_empty_result_is_unreliable() {
        WorkerStatus staleWorker = freshAliveWorker("10.0.0.1", 8080);
        workerStatusMap.put("10.0.0.1:8080", staleWorker);
        when(workerAddressService.getEngineWorkerDiscoveryResult(modelName, roleType))
                .thenReturn(WorkerDiscoveryResult.failure());

        engineSyncRunner.run();

        assertTrue(staleWorker.isAlive());
        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }

    @Test
    void should_mark_only_missing_cached_worker_unavailable() {
        WorkerStatus staleWorker = freshAliveWorker("10.0.0.1", 8080);
        WorkerStatus activeWorker = freshAliveWorker("10.0.0.2", 8080);
        workerStatusMap.put("10.0.0.1:8080", staleWorker);
        workerStatusMap.put("10.0.0.2:8080", activeWorker);
        when(workerAddressService.getEngineWorkerDiscoveryResult(modelName, roleType))
                .thenReturn(WorkerDiscoveryResult.success(
                        List.of(WorkerHost.of("10.0.0.2", 8080))));

        engineSyncRunner.run();

        assertFalse(staleWorker.isAlive());
        assertTrue(activeWorker.isAlive());
    }

    @Test
    void should_not_mark_missing_cached_worker_unavailable_when_partial_discovery_result_is_unreliable() {
        WorkerStatus staleWorker = freshAliveWorker("10.0.0.1", 8080);
        WorkerStatus activeWorker = freshAliveWorker("10.0.0.2", 8080);
        workerStatusMap.put("10.0.0.1:8080", staleWorker);
        workerStatusMap.put("10.0.0.2:8080", activeWorker);
        when(workerAddressService.getEngineWorkerDiscoveryResult(modelName, roleType))
                .thenReturn(new WorkerDiscoveryResult(
                        List.of(WorkerHost.of("10.0.0.2", 8080)),
                        false));

        engineSyncRunner.run();

        assertTrue(staleWorker.isAlive());
        assertTrue(activeWorker.isAlive());
        verify(statusCheckExecutor, times(4)).submit(any(Runnable.class));
    }

    private static WorkerStatus freshAliveWorker(String ip, int port) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(port);
        workerStatus.setAlive(true);
        workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        workerStatus.getStatusUpdateIntervalUs().set(1_000_000L);
        return workerStatus;
    }
}
