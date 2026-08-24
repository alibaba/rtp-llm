package org.flexlb.sync.runner;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcWorkerStatusCheckRunnerTest {

    private final EngineGrpcService engineGrpcService = Mockito.mock(EngineGrpcService.class);

    private final EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);

    private final CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);

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
                .setCacheMatchRollbackBlocks(1)
                .setKvCacheGroupMode(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA)
                .build();

        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(), org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        // Act
        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                modelName, host,
                RoleType.PREFILL,
                workerStatus, engineHealthReporter, engineGrpcService, 20, cacheAwareService);
        runner.run();

        // Assert
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 18002, -1L, 20L, RoleType.PREFILL);
        assertEquals(64, workerStatus.getCacheStatus().getBlockSize());
        assertEquals(800, workerStatus.getAvailableKvCacheTokens().get());
        assertEquals(200, workerStatus.getUsedKvCacheTokens().get());
        assertEquals(1, workerStatus.getBlockHashLookaheadTokens());
        assertEquals(1, workerStatus.getCacheMatchRollbackBlocks());
        assertEquals(KvCacheGroupMode.WITH_MAMBA, workerStatus.getKvCacheGroupMode());
    }

    @Test
    void shouldReportTimeoutFailureLatency_whenWorkerStatusGrpcTimesOut() {
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class)))
                .thenThrow(new RuntimeException("DEADLINE_EXCEEDED"));

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        verify(engineHealthReporter).reportStatusCheckerFail(
                "test-model", BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT,
                "127.0.0.1@0", RoleType.PREFILL);
        assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream().anyMatch(invocation -> {
            Object[] arguments = invocation.getArguments();
            return invocation.getMethod().getName().equals("reportStatusCheckFailureLatency")
                    && arguments.length == 5
                    && "test-model".equals(arguments[0])
                    && BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT.equals(arguments[1])
                    && "127.0.0.1@0".equals(arguments[2])
                    && RoleType.PREFILL.equals(arguments[3])
                    && (long) arguments[4] >= 0;
        }));
    }

    @Test
    void shouldKeepAliveUnchangedWhenWorkerStatusIsNotInitialized() {
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18003, "test-site", "test-group", "deployment-a",
                1, 2);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setAlive(true);
        EngineRpcService.WorkerStatusPB uninitialized = EngineRpcService.WorkerStatusPB.newBuilder()
                .setAlive(true)
                .setStatusVersion(0)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(uninitialized);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        assertTrue(workerStatus.isAlive());
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 18003, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void shouldKeepAliveUnchangedWhenStatusResponseHandlingFails() {
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18003, "test-site", "test-group", "deployment-a",
                1, 2);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setAlive(true);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(1)
                .setAlive(false)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(response);
        Mockito.doThrow(new RuntimeException("metrics unavailable"))
                .when(engineHealthReporter)
                .reportStatusCheckRemoteInfo(anyString(), anyString(), anyString(), anyLong());

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        assertTrue(workerStatus.isAlive());
    }

    @Test
    void shouldUpdateAliveFromExplicitWorkerStatusResponse() {
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18003, "test-site", "test-group", "deployment-a",
                1, 2);
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setAlive(true);
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(1)
                .setAlive(false)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(response);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        assertFalse(workerStatus.isAlive());
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

        CacheHitFeedback expected = new CacheHitFeedback(
                "cache_hit_comparison", requestId, "KVCM", "PREFILL", "test-group", "127.0.0.1", 8080,
                "running", 200, 64, 100, 120, 20);
        CacheHitComparisonResult unifiedComparison = new CacheHitComparisonResult(
                "cache_hit_comparison", requestId, "KVCM", "PREFILL", "test-group",
                "127.0.0.1:8080@0",
                "127.0.0.1@0",
                "running", 200,
                new CacheHitComparisonResult.Actual(120),
                new CacheHitComparisonResult.HitComparison(100, 20),
                new CacheHitComparisonResult.HitComparison(80, 40),
                null);
        when(cacheAwareService.buildCacheHitComparison(expected))
                .thenReturn(CompletableFuture.completedFuture(unifiedComparison));

        // Act
        new GrpcWorkerStatusRunner(
                modelName, host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        // Assert
        assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream().anyMatch(invocation -> {
            Object[] arguments = invocation.getArguments();
            return invocation.getMethod().getName().equals("reportCacheHitComparisonMetrics")
                    && arguments.length == 2
                    && modelName.equals(arguments[0])
                    && unifiedComparison.equals(arguments[1]);
        }));
    }

    @Test
    void shouldReportFinishedPrefillTasksForPdfusionButNotDecode() {
        String requestId = "finished-request";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        EngineRpcService.TaskInfoPB finishedTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputQueueEnqueueTimeMs(1_000)
                .setInputQueueDrainTimeMs(1_100)
                .setFirstTokenTimeMs(1_200)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PDFUSION.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addFinishedTaskList(finishedTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        ch.qos.logback.classic.Logger pvLogger =
                (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        ListAppender<ILoggingEvent> pvEvents = new ListAppender<>();
        pvEvents.start();
        pvLogger.addAppender(pvEvents);
        try {
            new GrpcWorkerStatusRunner(
                    "test-model", host, RoleType.PDFUSION, new WorkerStatus(), engineHealthReporter,
                    engineGrpcService, 20, cacheAwareService).run();

            assertTrue(pvEvents.list.stream()
                    .map(ILoggingEvent::getFormattedMessage)
                    .anyMatch(message -> message.contains("\"event\":\"prefill_worker_status\"")
                            && message.contains("\"requestId\":\"" + requestId + "\"")
                            && message.contains("\"role\":\"" + RoleType.PDFUSION.getCode() + "\"")));
            assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream().anyMatch(invocation -> {
                Object[] arguments = invocation.getArguments();
                return invocation.getMethod().getName().equals("reportPrefillWorkerStatusTask")
                        && arguments.length == 5
                        && RoleType.PDFUSION.getCode().equals(arguments[2]);
            }));

            Mockito.reset(engineHealthReporter);
            new GrpcWorkerStatusRunner(
                    "test-model", host, RoleType.DECODE, new WorkerStatus(), engineHealthReporter,
                    engineGrpcService, 20, cacheAwareService).run();

            assertEquals(1, pvEvents.list.stream()
                    .map(ILoggingEvent::getFormattedMessage)
                    .filter(message -> message.contains("\"event\":\"prefill_worker_status\""))
                    .count());
            assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream()
                    .noneMatch(invocation -> invocation.getMethod().getName().equals("reportPrefillWorkerStatusTask")));
        } finally {
            pvLogger.detachAppender(pvEvents);
            pvEvents.stop();
        }
    }

    @Test
    void shouldReportWaitingConfirmationLatency_whenTaskFirstAppearsInWaitingQueue() {
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        workerStatus.putLocalTask(requestId, localTask);
        localTask.setLastActiveTimeUs(System.nanoTime() / 1000 - 5_000);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        assertTrue(Mockito.mockingDetails(engineHealthReporter).getInvocations().stream().anyMatch(invocation -> {
            Object[] arguments = invocation.getArguments();
            return invocation.getMethod().getName().equals("reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency")
                    && arguments.length == 5
                    && "test-model".equals(arguments[0])
                    && "127.0.0.1@0".equals(arguments[1])
                    && RoleType.PREFILL.getCode().equals(arguments[2])
                    && "test-group".equals(arguments[3])
                    && (long) arguments[4] >= 5;
        }));
    }

    @Test
    void shouldReportWaitingToRunningLatency_whenTaskMovesFromWaitingToRunning() {
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        workerStatus.putLocalTask(requestId, localTask);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB waitingStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(waitingStatusPB);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .build();
        EngineRpcService.WorkerStatusPB runningStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(runningTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(runningStatusPB);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        long waitingToRunningReportCount = Mockito.mockingDetails(engineHealthReporter)
                .getInvocations().stream()
                .filter(invocation -> {
                    Object[] arguments = invocation.getArguments();
                    return invocation.getMethod().getName().equals("reportFlexlbObservedWaitingToRunningLatency")
                            && arguments.length == 5
                            && "test-model".equals(arguments[0])
                            && "127.0.0.1@0".equals(arguments[1])
                            && RoleType.PREFILL.getCode().equals(arguments[2])
                            && "test-group".equals(arguments[3])
                            && (long) arguments[4] >= 0;
                })
                .count();
        assertEquals(1, waitingToRunningReportCount);
    }

    @Test
    void shouldRefreshPendingQueueSnapshot_whenStatusVersionIsUnchanged() {
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        localTask.setInputLength(64_000);
        localTask.setPredictedPrefixLength(48_000);
        workerStatus.putLocalTask(requestId, localTask);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputLength(64_000)
                .setIsWaiting(true)
                .build();
        EngineRpcService.WorkerStatusPB waitingStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();
        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setInputLength(64_000)
                .setCompletedPrefillTokens(16_384)
                .setRemainingPrefillTokens(16_000)
                .setLastCompletedPrefillStepId(1)
                .build();
        EngineRpcService.WorkerStatusPB runningStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(runningTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(waitingStatusPB, runningStatusPB);

        GrpcWorkerStatusRunner runner = new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService);
        runner.run();
        assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(16_000, workerStatus.getInTransitAndWaitingUncachedTokens());

        runner.run();

        assertEquals(0, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(0, workerStatus.getInTransitAndWaitingUncachedTokens());
        assertEquals(16_000, workerStatus.getRunningRemainingPrefillTokens());
        assertEquals(16_384,
                workerStatus.getLocalTaskMap().get(requestId).getCompletedPrefillTokens());
        assertEquals(1,
                workerStatus.getLocalTaskMap().get(requestId).getLastCompletedPrefillStepId());
    }

    @Test
    void shouldReportEngineObservedWaitingToRunningLatency_fromEngineTimestamps() {
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        workerStatus.putLocalTask(requestId, localTask);

        EngineRpcService.TaskInfoPB runningTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setWaitingEnteredTimeMs(1_000L)
                .setRunningEnteredTimeMs(1_250L)
                .build();
        EngineRpcService.WorkerStatusPB runningStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(runningTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(runningStatusPB);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        long engineObservedReportCount = Mockito.mockingDetails(engineHealthReporter)
                .getInvocations().stream()
                .filter(invocation -> {
                    Object[] arguments = invocation.getArguments();
                    return invocation.getMethod().getName().equals("reportEngineObservedWaitingToRunningLatency")
                            && arguments.length == 5
                            && "test-model".equals(arguments[0])
                            && "127.0.0.1@0".equals(arguments[1])
                            && RoleType.PREFILL.getCode().equals(arguments[2])
                            && "test-group".equals(arguments[3])
                            && (long) arguments[4] == 250L;
                })
                .count();
        assertEquals(1, engineObservedReportCount);
    }

    @Test
    void shouldReportEngineObservedReceivedToWaitingLatency_fromEngineTimestamps() {
        String requestId = "request-1";
        WorkerHost host = new WorkerHost(
                "127.0.0.1", 8080, 8081, 8085, 18002, "test-site", "test-group", "deployment-a");
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");

        TaskInfo localTask = new TaskInfo();
        localTask.setRequestId(requestId);
        workerStatus.putLocalTask(requestId, localTask);

        EngineRpcService.TaskInfoPB waitingTask = EngineRpcService.TaskInfoPB.newBuilder()
                .setRequestId(requestId)
                .setIsWaiting(true)
                .setRequestReceivedTimeMs(1_000L)
                .setWaitingEnteredTimeMs(1_080L)
                .build();
        EngineRpcService.WorkerStatusPB waitingStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(RoleType.PREFILL.getCode())
                .setStatusVersion(100)
                .setAlive(true)
                .addRunningTaskInfo(waitingTask)
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(waitingStatusPB);

        new GrpcWorkerStatusRunner(
                "test-model", host, RoleType.PREFILL, workerStatus, engineHealthReporter,
                engineGrpcService, 20, cacheAwareService).run();

        long receivedToWaitingReportCount = Mockito.mockingDetails(engineHealthReporter)
                .getInvocations().stream()
                .filter(invocation -> {
                    Object[] arguments = invocation.getArguments();
                    return invocation.getMethod().getName().equals("reportEngineObservedReceivedToWaitingLatency")
                            && arguments.length == 5
                            && "test-model".equals(arguments[0])
                            && "127.0.0.1@0".equals(arguments[1])
                            && RoleType.PREFILL.getCode().equals(arguments[2])
                            && "test-group".equals(arguments[3])
                            && (long) arguments[4] == 80L;
                })
                .count();
        assertEquals(1, receivedToWaitingReportCount);
    }
}
