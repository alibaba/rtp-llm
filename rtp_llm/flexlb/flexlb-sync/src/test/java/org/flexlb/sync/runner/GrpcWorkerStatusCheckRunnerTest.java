package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.pv.CacheHitComparisonPvLog;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.KvCacheGroupMode;
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
        String site = "test-site";
        String group = "test-group";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, site, group, "deployment-a");

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
                .setAvailableKvCache(800)
                .setTotalKvCache(1000)
                .setBlockSize(64)
                .setBlockHashLookaheadTokens(1)
                .setKvCacheGroupMode(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .build();

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(), org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        // Act
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, host,
                RoleType.PREFILL,
                workerStatus, engineHealthReporter, engineGrpcService, 20);
        runner.run();

        // Assert
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 18002, -1L, 20L, RoleType.PREFILL);
        assertEquals(64, workerStatus.getCacheStatus().getBlockSize());
        assertEquals(800, workerStatus.getAvailableKvCacheTokens().get());
        assertEquals(200, workerStatus.getUsedKvCacheTokens().get());
        assertEquals(1, workerStatus.getBlockHashLookaheadTokens());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, workerStatus.getKvCacheGroupMode());
    }

    @Test
    void shouldReportCacheHitComparison_whenActualHitFirstBecomesValid() {
        // Arrange
        String modelName = "test-model";
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        localTask.setInputLength(200);
        localTask.setPrefixLength(100);
        localTask.setPredictedPrefixLength(100);
        localTask.setCacheMatchSource("KVCM");
        workerStatus.putLocalTask(requestId, localTask);

        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputLength(200)
                .setPrefixLength(120)
                .setPrefixLengthValid(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setStatusVersion(100)
                .setAlive(true)
                .setAvailableKvCache(800)
                .setTotalKvCache(1000)
                .setBlockSize(64)
                .addRunningTaskInfo(runningTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(), org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenReturn(workerStatusPB);

        // Act
        new GrpcWorkerStatusRunner(
                modelName, host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20).run();

        // Assert
        CacheHitComparisonPvLog expected = new CacheHitComparisonPvLog(
                "cache_hit_comparison", requestId, "KVCM", "PREFILL", "test-group", "127.0.0.1", 8080,
                "running", 200, 64, 100, 120, 20);
        assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream().anyMatch(invocation -> {
            Object[] arguments = invocation.getArguments();
            return invocation.getMethod().getName().equals("reportCacheHitComparisonMetrics")
                    && arguments.length == 2
                    && modelName.equals(arguments[0])
                    && expected.equals(arguments[1]);
        }));
    }
}
