package org.flexlb.sync.runner;

import io.grpc.Status;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.times;
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
                .setRole("test-role")
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
                group, workerStatus, engineHealthReporter, engineGrpcService, 20);
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

    @Test
    void should_useLongerTimeoutForVitStatusCheck() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("VIT")
                .setStatusVersion(1)
                .setAlive(true)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", "127.0.0.1:8080", "test-site", RoleType.VIT,
                "test-group", workerStatus, engineHealthReporter, engineGrpcService,
                20, 1000, true);
        runner.run();

        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 8081, -1L, 1000L, RoleType.VIT);
    }

    @Test
    void should_notShortenGlobalTimeoutForVitStatusCheck() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setRole("VIT")
                        .setStatusVersion(1)
                        .setAlive(true)
                        .build());

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", "127.0.0.1:8080", "test-site", RoleType.VIT,
                "test-group", workerStatus, engineHealthReporter, engineGrpcService,
                5000, 1000, true);
        runner.run();

        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 8081, -1L, 5000L, RoleType.VIT);
    }

    @Test
    void should_keepLastVitAliveStateWhenStatusCheckTimesOut() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(Status.DEADLINE_EXCEEDED.asRuntimeException());

        createRunner(RoleType.VIT, workerStatus).run();

        assertTrue(workerStatus.isAlive());
        verify(engineHealthReporter).reportStatusCheckerFail(
                "test-model", BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, RoleType.VIT);
        verify(engineHealthReporter, times(1)).reportStatusCheckerFail(
                anyString(), any(BalanceStatusEnum.class), any(RoleType.class));
    }

    @Test
    void should_markVitDeadWhenTimeoutRetentionIsDisabled() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(Status.DEADLINE_EXCEEDED.asRuntimeException());

        createRunner(RoleType.VIT, workerStatus, false).run();

        assertFalse(workerStatus.isAlive());
    }

    @Test
    void should_markPrefillDeadWhenStatusCheckTimesOut() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(Status.DEADLINE_EXCEEDED.asRuntimeException());

        createRunner(RoleType.PREFILL, workerStatus).run();

        assertFalse(workerStatus.isAlive());
    }

    @Test
    void should_markVitDeadWhenStatusCheckFailsWithoutTimeout() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(new RuntimeException("connection refused"));

        createRunner(RoleType.VIT, workerStatus).run();

        assertFalse(workerStatus.isAlive());
    }

    @Test
    void should_applyExplicitDeadStatusFromVit() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("VIT")
                .setStatusVersion(1)
                .setAlive(false)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        createRunner(RoleType.VIT, workerStatus).run();

        assertFalse(workerStatus.isAlive());
    }

    @Test
    void should_notTrustDeadlineTokenInNonGrpcErrorMessage() {
        WorkerStatus workerStatus = aliveWorkerStatus();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(new RuntimeException("worker said DEADLINE_EXCEEDED but connection failed"));

        createRunner(RoleType.VIT, workerStatus).run();

        assertFalse(workerStatus.isAlive());
    }

    private WorkerStatus aliveWorkerStatus() {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        workerStatus.setAlive(true);
        return workerStatus;
    }

    private GrpcWorkerStatusRunner createRunner(RoleType roleType, WorkerStatus workerStatus) {
        return createRunner(roleType, workerStatus, true);
    }

    private GrpcWorkerStatusRunner createRunner(
            RoleType roleType, WorkerStatus workerStatus, boolean retainVitAliveOnTimeout) {
        return new GrpcWorkerStatusRunner(
                "test-model",
                "127.0.0.1:8080",
                "test-site",
                roleType,
                "test-group",
                workerStatus,
                engineHealthReporter,
                engineGrpcService,
                20,
                1000,
                retainVitAliveOnTimeout);
    }
}
