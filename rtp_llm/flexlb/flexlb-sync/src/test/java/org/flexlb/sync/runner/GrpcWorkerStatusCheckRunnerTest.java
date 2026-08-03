package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
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

/**
 * Tests for {@link GrpcWorkerStatusRunner}.
 *
 * <p>Key API changes since original implementation:
 * <ul>
 *   <li>Proto field {@code is_waiting} replaced by {@code TaskPhase phase}</li>
 *   <li>{@code WorkerStatus.runningTaskList} replaces old {@code waitingTaskList + localTaskMap}</li>
 *   <li>Constructor requires {@code FlexlbBatchScheduler + EndpointRegistry} (nullable)</li>
 *   <li>Task list refresh only occurs when status version advances (not on equal version)</li>
 * </ul>
 */
class GrpcWorkerStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    @Test
    void should_callGrpcServiceAndVerifyInteraction_when_runnerExecutes() {
        // Arrange
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";
        String group = "test-group";

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
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
                .build();

        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(workerStatusPB));

        // Act — pass null for FlexlbBatchScheduler and EndpointRegistry (not needed in unit test)
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, Map.of(ipPort, workerStatus),
                engineHealthReporter, engineGrpcService, 20L, null, null, Runnable::run);
        runner.run();

        // Assert — gRPC port is derived from HTTP port 8080 → 8081
        verify(engineGrpcService).getWorkerStatusAsync("127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void should_not_update_task_list_when_status_version_is_unchanged() {
        // When the gRPC response version equals the local version, the status update
        // is skipped — including the runningTaskList refresh. This avoids unnecessary
        // state churn when the engine hasn't changed.
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";
        String group = "test-group";

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.getStatusVersion().set(100L);

        // Use TaskPhasePB instead of the removed is_waiting field
        EngineRpcService.TaskInfoPB taskInfo = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setInputLength(100)
                .setPhase(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setRoleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(100L)
                .setAlive(true)
                .addRunningTaskInfo(taskInfo)
                .build();

        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(CompletableFuture.completedFuture(workerStatusPB));

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, Map.of(ipPort, workerStatus),
                engineHealthReporter, engineGrpcService, 20L, null, null, Runnable::run);
        runner.run();

        // Version not advanced → runningTaskList should NOT be populated from response
        assertNull(workerStatus.getRunningTaskList(),
                "runningTaskList should not be updated when status version is unchanged");
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
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(response));

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                expired, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);
        runner.run();

        assertSame(currentEndpoint, registry.get(RoleType.VIT, ipPort));
        assertSame(current, currentEndpoint.getStatus());
        assertEquals(-1L, expired.getStatusVersion().get());
        registry.close();
    }

    @Test
    void should_remove_endpoint_after_consecutive_grpc_failures() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(8080);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, status);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                status, statuses, engineHealthReporter, engineGrpcService,
                20L, null, registry, Runnable::run);

        runner.run();
        runner.run();
        runner.run();

        assertFalse(status.isAlive());
        assertNull(registry.get(RoleType.VIT, ipPort));
        registry.close();
    }

    @Test
    void should_restore_endpoint_when_same_version_worker_recovers() {
        String ipPort = "127.0.0.1:8080";
        WorkerStatus status = status(8080);
        status.getStatusVersion().set(100L);
        Map<String, WorkerStatus> statuses = new ConcurrentHashMap<>();
        statuses.put(ipPort, status);
        EndpointRegistry registry = registry();
        registry.ensureEndpoint(RoleType.VIT, ipPort, status);
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("unavailable")));
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", ipPort, "test-site", RoleType.VIT, "test-group",
                status, statuses, engineHealthReporter, engineGrpcService,
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
        when(engineGrpcService.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(CompletableFuture.completedFuture(recovered));

        runner.run();

        assertTrue(status.isAlive());
        assertSame(status, registry.get(RoleType.VIT, ipPort).getStatus());
        registry.close();
    }

    private static EndpointRegistry registry() {
        ConfigService configService = Mockito.mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new EndpointRegistry(
                configService, () -> null, Mockito.mock(BatchSchedulerReporter.class));
    }

    private static WorkerStatus status(int port) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(RoleType.VIT);
        status.setIp("127.0.0.1");
        status.setPort(port);
        status.setGrpcPort(port + 1);
        status.setAlive(true);
        return status;
    }
}
