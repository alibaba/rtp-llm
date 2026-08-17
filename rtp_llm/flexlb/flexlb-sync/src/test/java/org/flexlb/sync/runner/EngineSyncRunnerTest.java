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
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doThrow;
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
    void should_submit_only_worker_status_check_for_vit() {
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(host));

        createRunner(RoleType.VIT).run();

        ArgumentCaptor<Runnable> taskCaptor = ArgumentCaptor.forClass(Runnable.class);
        verify(statusCheckExecutor).submit(taskCaptor.capture());
        assertTrue(taskCaptor.getValue() instanceof GrpcWorkerStatusRunner);
        WorkerStatus workerStatus = workerStatusMap.get(host.getIpPort());
        assertFalse(workerStatus.getCacheCheckInProgress().get());
    }

    @Test
    void should_submit_worker_and_cache_status_checks_for_prefill() {
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.PREFILL))
                .thenReturn(List.of(host));

        createRunner(RoleType.PREFILL).run();

        ArgumentCaptor<Runnable> taskCaptor = ArgumentCaptor.forClass(Runnable.class);
        verify(statusCheckExecutor, times(2)).submit(taskCaptor.capture());
        assertTrue(taskCaptor.getAllValues().stream()
                .anyMatch(GrpcWorkerStatusRunner.class::isInstance));
        assertTrue(taskCaptor.getAllValues().stream()
                .anyMatch(GrpcCacheStatusCheckRunner.class::isInstance));
    }

    @Test
    void should_reset_status_check_flag_when_executor_rejects_task() {
        WorkerHost host = new WorkerHost("127.0.0.1", 8080, "test-site");
        when(workerAddressService.getEngineWorkerList(modelName, RoleType.VIT))
                .thenReturn(List.of(host));
        doThrow(new RejectedExecutionException("executor stopped"))
                .when(statusCheckExecutor).submit(any(Runnable.class));

        createRunner(RoleType.VIT).run();

        assertFalse(workerStatusMap.get(host.getIpPort()).getStatusCheckInProgress().get());
    }

    private EngineSyncRunner createRunner(RoleType actualRoleType) {
        return new EngineSyncRunner(
                modelName,
                workerStatusMap,
                workerAddressService,
                statusCheckExecutor,
                engineHealthReporter,
                engineGrpcService,
                actualRoleType,
                localKvCacheAwareManager,
                syncRequestTimeoutMs,
                syncCount,
                syncEngineStatusInterval
        );
    }
}
