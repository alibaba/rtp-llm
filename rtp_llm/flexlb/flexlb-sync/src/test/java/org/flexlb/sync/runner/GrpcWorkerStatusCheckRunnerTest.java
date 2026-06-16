package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
                .setRole(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
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

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(), org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        // Act
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, engineHealthReporter, engineGrpcService, 20, null, null);
        runner.run();

        // Assert
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void should_refreshTaskLists_when_statusVersionIsNotUpdated() {
        String modelName = "test-model";
        String ipPort = "127.0.0.1:8080";
        String site = "test-site";
        String group = "test-group";

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.getStatusVersion().set(100L);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(123L)
                .setInputLength(100)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100L)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(), org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, ipPort, site,
                RoleType.PREFILL,
                group, workerStatus, engineHealthReporter, engineGrpcService, 20);
        runner.run();

        assertEquals(1, workerStatus.getWaitingTaskList().size());
        assertTrue(workerStatus.getWaitingTaskList().containsKey("123"));
    }
}
