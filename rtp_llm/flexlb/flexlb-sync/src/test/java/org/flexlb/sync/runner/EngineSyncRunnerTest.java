package org.flexlb.sync.runner;

import org.flexlb.cache.match.CacheAwareService;
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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    private CacheAwareService cacheAwareService;

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
                cacheAwareService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false
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
                cacheAwareService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false
        );

        // Execute
        runnerWithNullMap.run();

        // Verify
        verify(statusCheckExecutor, never()).submit(any(Runnable.class));
    }

    @Test
    void should_only_submit_worker_status_check_when_kvcmIsEnabled() {
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(java.util.List.of(WorkerHost.of("127.0.0.1", 8080)));

        EngineSyncRunner kvcmRunner = new EngineSyncRunner(
                modelName,
                new ConcurrentHashMap<>(),
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                cacheAwareService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                true
        );

        kvcmRunner.run();

        verify(statusCheckExecutor, times(1)).submit(any(Runnable.class));
    }

    @Test
    void submitsCacheStatusCheckForLogicalWorkerWhenKvcmIsDisabled() {
        WorkerHost engine0 = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002,
                "site-a", "group-a", "deployment-a", 0, 2, "service-a");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(java.util.List.of(engine0));
        EngineSyncRunner localSyncRunner = new EngineSyncRunner(
                modelName,
                new ConcurrentHashMap<>(),
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                cacheAwareService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                false);

        localSyncRunner.run();

        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }

    @Test
    void createsIndependentStatusEntriesForEachLogicalEngine() {
        WorkerHost engine0 = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002,
                "site-a", "group-a", "deployment-a", 0, 2, "service-a");
        WorkerHost engine1 = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18003,
                "site-a", "group-a", "deployment-a", 1, 2, "service-a");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(java.util.List.of(engine0, engine1));
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        EngineSyncRunner runner = new EngineSyncRunner(
                modelName,
                statuses,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                RoleType.PREFILL,
                cacheAwareService,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval,
                true);

        runner.run();

        assertEquals(2, statuses.size());
        assertTrue(statuses.containsKey("127.0.0.1:8080@0"));
        assertTrue(statuses.containsKey("127.0.0.1:8080@1"));
        assertEquals(0, statuses.get("127.0.0.1:8080@0").getEngineIndex());
        assertEquals(1, statuses.get("127.0.0.1:8080@1").getEngineIndex());
        assertEquals(2, statuses.get("127.0.0.1:8080@1").getMultiEngineNum());
        verify(statusCheckExecutor, times(2)).submit(any(Runnable.class));
    }
}
