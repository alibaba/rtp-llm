package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);
    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    @Test
    void should_callGrpcServiceAndApplyWorkerMetadata_when_runnerExecutes() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus workerStatus = status(8080);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setAvailableConcurrency(10)
                .setRunningQueryLen(5)
                .setWaitingQueryLen(3)
                .setStepLatencyMs(100)
                .setIterateCount(20)
                .setDpSize(2)
                .setTpSize(4)
                .setStatusVersion(100)
                .setAlive(true)
                .setAvailableKvCache(800)
                .setTotalKvCache(1000)
                .setBlockSize(64)
                .setBlockHashLookaheadTokens(1)
                .setCacheMatchRollbackBlocks(1)
                .setKvCacheGroupMode(
                        EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .build();
        whenStatus(response);

        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.PREFILL, "test-group",
                workerStatus, Map.of(ipPort, workerStatus), engineHealthReporter,
                engineGrpcService, 20L, null, null, Runnable::run).run();

        verify(engineGrpcService).getWorkerStatusAsync(
                "127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
        assertEquals(64, workerStatus.getCacheStatus().getBlockSize());
        assertEquals(800, workerStatus.getAvailableKvCacheTokens().get());
        assertEquals(200, workerStatus.getUsedKvCacheTokens().get());
        assertEquals(1, workerStatus.getBlockHashLookaheadTokens());
        assertEquals(1, workerStatus.getCacheMatchRollbackBlocks());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, workerStatus.getKvCacheGroupMode());
    }

    @Test
    void should_refresh_task_lifecycle_when_status_version_is_unchanged() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus workerStatus = status(8080);
        workerStatus.getStatusVersion().set(100L);
        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(123L);
        localTask.setInputLength(64_000);
        localTask.setPredictedPrefixLength(48_000);
        workerStatus.putLocalTask("123", localTask);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setInputLength(64_000)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(100L)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();
        whenStatus(response);

        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.PREFILL, "test-group",
                workerStatus, Map.of(ipPort, workerStatus), engineHealthReporter,
                engineGrpcService, 20L, null, null, Runnable::run).run();

        assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(16_000, workerStatus.getInTransitAndWaitingUncachedTokens());
        assertEquals(1, workerStatus.getWaitingTaskList().size());
    }

    @Test
    void should_ignore_status_callback_from_expired_generation() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus expired = status(8080);
        WorkerStatus current = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, current);
        EndpointRegistry registry = registry();
        WorkerEndpoint currentEndpoint = registry.ensureEndpoint(RoleType.VIT, ipPort, current);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(100L)
                .setAlive(true)
                .build();
        whenStatus(response);

        new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                expired, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run).run();

        assertSame(currentEndpoint, registry.get(RoleType.VIT, ipPort));
        assertSame(current, currentEndpoint.getStatus());
        assertEquals(-1L, expired.getStatusVersion().get());
        registry.close();
    }

    @Test
    void should_remove_endpoint_after_consecutive_grpc_failures() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus workerStatus = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, workerStatus);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, workerStatus);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                workerStatus, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();
        runner.run();
        runner.run();

        assertFalse(workerStatus.isAlive());
        assertNull(registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void should_restore_endpoint_when_same_version_worker_recovers() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus workerStatus = status(8080);
        workerStatus.getStatusVersion().set(100L);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, workerStatus);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, workerStatus);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                workerStatus, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);
        runner.run();
        runner.run();
        runner.run();

        EngineRpcService.WorkerStatusPB recovered = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.VIT.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_VIT)
                .setStatusVersion(100L)
                .setAlive(true)
                .build();
        whenStatus(recovered);
        runner.run();

        assertTrue(workerStatus.isAlive());
        assertSame(workerStatus, registry.get(RoleType.VIT, ipPort).getStatus());
        registry.close();
    }

    private void whenStatus(EngineRpcService.WorkerStatusPB response) {
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response));
    }

    private static EndpointRegistry registry() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
    }

    private static WorkerStatus status(int port) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setRole(RoleType.VIT);
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(port);
        workerStatus.setGrpcPort(port + 1);
        workerStatus.setAlive(true);
        return workerStatus;
    }
}
