package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.Set;
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
                syncEngineStatusInterval
        );
    }

    @Test
    void should_not_submit_tasks_when_no_workers_exist() {
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(), Set.of()));

        // Execute
        engineSyncRunner.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_handle_null_worker_status_gracefully() {
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(), Set.of()));

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
    void should_remove_cached_workers_for_unavailable_discovery_group() {
        WorkerStatus grayWorker = new WorkerStatus();
        grayWorker.setGroup("gray");
        grayWorker.setAlive(true);
        grayWorker.getStatusLastUpdateTime().set(1L);
        grayWorker.getStatusUpdateIntervalUs().set(1L);
        workerStatusMap.put("10.0.0.1:8080", grayWorker);

        WorkerStatus stableWorker = new WorkerStatus();
        stableWorker.setGroup("stable");
        workerStatusMap.put("10.0.0.2:8080", stableWorker);

        WorkerHost stableHost = new WorkerHost("10.0.0.2", 8080, 8081, 8085, "site", "stable");
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(stableHost), Set.of("gray")));

        engineSyncRunner.run();

        assertFalse(workerStatusMap.containsKey("10.0.0.1:8080"));
        assertTrue(workerStatusMap.containsKey("10.0.0.2:8080"));
    }

    @Test
    void should_remove_never_synced_cached_workers_after_first_seen_threshold() {
        WorkerStatus grayWorker = new WorkerStatus();
        grayWorker.setGroup("gray");
        grayWorker.setAlive(true);
        grayWorker.getStatusFirstSeenTimeUs().set(1L);
        workerStatusMap.put("10.0.0.1:8080", grayWorker);

        WorkerHost stableHost = new WorkerHost("10.0.0.2", 8080, 8081, 8085, "site", "stable");
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(stableHost), Set.of("gray")));

        engineSyncRunner.run();

        assertFalse(workerStatusMap.containsKey("10.0.0.1:8080"));
        assertTrue(workerStatusMap.containsKey("10.0.0.2:8080"));
    }

    @Test
    void should_mark_unavailable_group_workers_before_removal_threshold() {
        WorkerStatus grayWorker = new WorkerStatus();
        grayWorker.setGroup("gray");
        grayWorker.setAlive(true);
        workerStatusMap.put("10.0.0.1:8080", grayWorker);

        WorkerStatus stableWorker = new WorkerStatus();
        stableWorker.setGroup("stable");
        workerStatusMap.put("10.0.0.2:8080", stableWorker);

        WorkerHost stableHost = new WorkerHost("10.0.0.2", 8080, 8081, 8085, "site", "stable");
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(stableHost), Set.of("gray")));

        engineSyncRunner.run();

        assertTrue(workerStatusMap.containsKey("10.0.0.1:8080"));
        assertFalse(workerStatusMap.get("10.0.0.1:8080").isAlive());
        assertFalse(workerStatusMap.get("10.0.0.1:8080").getResourceAvailable().get());
        assertTrue(workerStatusMap.containsKey("10.0.0.2:8080"));
        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }

    @Test
    void should_keep_and_refresh_cached_workers_for_discovery_failed_group() {
        WorkerStatus grayWorker = new WorkerStatus();
        grayWorker.setGroup("gray");
        grayWorker.setAlive(true);
        grayWorker.getStatusLastUpdateTime().set(1L);
        grayWorker.getStatusUpdateIntervalUs().set(1L);
        workerStatusMap.put("10.0.0.1:8080", grayWorker);

        WorkerStatus stableWorker = new WorkerStatus();
        stableWorker.setGroup("stable");
        workerStatusMap.put("10.0.0.2:8080", stableWorker);

        WorkerHost stableHost = new WorkerHost("10.0.0.2", 8080, 8081, 8085, "site", "stable");
        when(workerAddressService.getEngineWorkers(modelName, roleType))
                .thenReturn(new WorkerAddressService.EngineWorkerList(List.of(stableHost), Set.of(), Set.of("gray")));

        engineSyncRunner.run();

        assertTrue(workerStatusMap.containsKey("10.0.0.1:8080"));
        assertTrue(workerStatusMap.get("10.0.0.1:8080").isAlive());
        assertTrue(workerStatusMap.containsKey("10.0.0.2:8080"));
        verify(statusCheckExecutor, times(4)).submit(any(Runnable.class));
    }
}
