package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
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

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        // Act — pass null for FlexlbBatchScheduler and EndpointRegistry (not needed in unit test)
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, engineHealthReporter, engineGrpcService, 20L, null, null);
        runner.run();

        // Assert — gRPC port is derived from HTTP port 8080 → 8081
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
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

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, engineHealthReporter, engineGrpcService, 20L, null, null);
        runner.run();

        // Version not advanced → runningTaskList should NOT be populated from response
        assertNull(workerStatus.getRunningTaskList(),
                "runningTaskList should not be updated when status version is unchanged");
    }
}
