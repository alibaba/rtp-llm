package org.flexlb.sync.runner;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
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

    @BeforeEach
    void setUp() {
        workerStatusMap = new ConcurrentHashMap<>();

        engineSyncRunner = newRunner(EngineType.LLM);
    }

    private EngineSyncRunner newRunner(EngineType engineType) {
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
                engineType
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
    void should_handle_null_worker_status_gracefully() {
        // Setup - create runner with null map
        EngineSyncRunner runnerWithNullMap = newRunner(EngineType.LLM);

        // Execute
        runnerWithNullMap.run();

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
