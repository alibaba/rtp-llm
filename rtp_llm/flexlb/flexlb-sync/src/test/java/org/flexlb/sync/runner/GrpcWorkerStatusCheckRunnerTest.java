package org.flexlb.sync.runner;

import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.PrefillBatch;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
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
                group, workerStatus, engineHealthReporter, engineGrpcService,
                new InflightBatchRegistry(), 20);
        runner.run();

        // Assert
        verify(engineGrpcService).getWorkerStatus("127.0.0.1", 8081, -1L, 20L, RoleType.PREFILL);
    }

    @Test
    void worker_finished_task_drops_active_inflight_entry() {
        // ACTIVE entry must drop when worker reports the requestId in
        // finishedTaskList — this is the cleanup-on-completion contract that
        // bounds InflightBatchRegistry by the actual request lifetime.
        InflightBatchRegistry registry = new InflightBatchRegistry();
        ServerStatus prefill = serverStatus("10.0.0.1", 8080, 9080);
        ServerStatus decode = serverStatus("10.0.0.2", 8081, 9081);
        PrefillBatch batch = new PrefillBatch(prefill, List.of(
                pending(1L, prefill, decode),
                pending(2L, prefill, decode)), 2);
        registry.register(7L, batch);
        assertTrue(registry.markActive(1L));
        assertTrue(registry.markActive(2L));

        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setStatusVersion(1)
                .setDpSize(1)
                .setAlive(true)
                .addRunningTaskInfo(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(1L)
                        .setIsWaiting(false)
                        .build())
                .addFinishedTaskList(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(1L)
                        .setIsWaiting(false)
                        .build())
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        new GrpcWorkerStatusRunner("m", "127.0.0.1:8080", "site", RoleType.PREFILL, "g",
                workerStatus, engineHealthReporter, engineGrpcService, registry, 20).run();

        assertNull(registry.lookupByRequest(1L), "finished requestId must be released");
        assertNotNull(registry.lookupByRequest(2L), "still-running requestId must stay");
    }

    @Test
    void worker_finished_task_does_not_touch_pending_ack_entry() {
        // Race guard: if worker reports finished for a requestId that is still
        // PENDING_ACK on master (handleAck hasn't run yet), removing it would
        // make the upcoming markActive look like a tombstone. Skip and let the
        // next reporter cycle catch it once state is ACTIVE.
        InflightBatchRegistry registry = new InflightBatchRegistry();
        ServerStatus prefill = serverStatus("10.0.0.1", 8080, 9080);
        ServerStatus decode = serverStatus("10.0.0.2", 8081, 9081);
        PrefillBatch batch = new PrefillBatch(prefill, List.of(
                pending(1L, prefill, decode)), 1);
        registry.register(7L, batch);
        assertEquals(InflightBatchRegistry.RequestState.PENDING_ACK, registry.getState(1L));

        EngineRpcService.WorkerStatusPB workerStatusPB = EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole("PREFILL")
                .setStatusVersion(1)
                .setDpSize(1)
                .setAlive(true)
                .addFinishedTaskList(EngineRpcService.TaskInfoPB.newBuilder()
                        .setRequestId(1L)
                        .setIsWaiting(false)
                        .build())
                .build();
        when(engineGrpcService.getWorkerStatus(anyString(), anyInt(), anyLong(), anyLong(),
                org.mockito.ArgumentMatchers.any(RoleType.class))).thenReturn(workerStatusPB);

        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp("127.0.0.1");
        workerStatus.setPort(8080);
        new GrpcWorkerStatusRunner("m", "127.0.0.1:8080", "site", RoleType.PREFILL, "g",
                workerStatus, engineHealthReporter, engineGrpcService, registry, 20).run();

        assertNotNull(registry.lookupByRequest(1L),
                "PENDING_ACK entry must NOT be released by reporter — let handleAck transition it first");
        assertEquals(InflightBatchRegistry.RequestState.PENDING_ACK, registry.getState(1L));
    }

    private static ServerStatus serverStatus(String ip, int httpPort, int grpcPort) {
        ServerStatus s = new ServerStatus();
        s.setServerIp(ip);
        s.setHttpPort(httpPort);
        s.setGrpcPort(grpcPort);
        s.setGroup("g");
        return s;
    }

    private static PendingRequest pending(long requestId, ServerStatus prefill, ServerStatus decode) {
        BalanceContext ctx = new BalanceContext();
        Request req = new Request();
        req.setRequestId(requestId);
        ctx.setRequest(req);
        return new PendingRequest(ctx, prefill, decode, new CompletableFuture<>(), 0L);
    }
}