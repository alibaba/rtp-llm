package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * A-class ghost convergence: post-ACK inflight entries whose request vanished
 * from the decode engine without a finished record (lost finished delta) must
 * be settled synchronously on the WorkerStatus sync path, while fenced
 * (D/E-class) and pre-ACK entries stay untouched. Also smokes the
 * window-aggregated release counter (audit + vanish settles merged into one
 * series). The 30s audit fallback shares the same release path.
 */
class PostAckInvisibleSettleTest {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private EngineCancelChannel cancelChannel;
    private WorkerStatus decodeWs;
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        cancelChannel = mock(EngineCancelChannel.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.tombstoned()));

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.completedFuture(ackFor(request));
                });
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, null, null, cancelChannel);
        replacePrefillEndpoint();
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void decodeVanish_settlesPostAckGhostImmediately() throws Exception {
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        // Dispatch claims decode ownership, which requires a live reservation.
        decode.reserve(310, 128, 136, 50, 0);

        CompletableFuture<Response> future = scheduler.submit(context(310));
        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, scheduler.getInflightSize(), "post-ACK entry is legal tracking");
        long batchId = sentBatches.getLast().getBatchId();

        // Decode engine confirms the request (RUNNING) — endpoint calibrate
        // first, then scheduler (DECODE_OWNED), mirroring the runner order.
        WorkerStatusResponse running = decodeStatus(
                Map.of("310", task(310, TaskPhase.RUNNING)), null);
        decode.onWorkerStatusUpdate(decodeWs, running);
        scheduler.onWorkerStatusUpdate(running, decode);
        assertTrue(decode.isEngineConfirmed(310));
        assertEquals(1, scheduler.getInflightSize());

        // Prefill side of the terminal already settled (invisible).
        assertTrue(prefill.tracksRequest(310));
        prefill.repackBatch(batchId, Set.of(310L));
        assertFalse(prefill.tracksRequest(310));

        // Lost finished delta: the request vanishes from the report entirely.
        WorkerStatusResponse empty = decodeStatus(null, null);
        decode.onWorkerStatusUpdate(decodeWs, empty);
        scheduler.onWorkerStatusUpdate(empty, decode);

        assertEquals(0, scheduler.getInflightSize(),
                "A-class ghost must settle on the same sync tick, not on the 30s audit");
        assertFalse(decode.isEngineConfirmed(310));

        // Window-aggregated counter (audit + vanish settles merged): flushed
        // once, then reset.
        scheduler.reportBatchMetrics();
        verify(reporter).reportSchedulerInflightAuditRelease(1L);
        scheduler.reportBatchMetrics();
        verify(reporter, times(1)).reportSchedulerInflightAuditRelease(anyLong());
    }

    @Test
    void decodeFinishedInReport_takesNormalTerminalPath_notVanishSettle() throws Exception {
        replacePrefillEndpoint();
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        decode.reserve(311, 128, 136, 50, 0);

        CompletableFuture<Response> future = scheduler.submit(context(311));
        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
        long batchId = sentBatches.getLast().getBatchId();

        WorkerStatusResponse running = decodeStatus(
                Map.of("311", task(311, TaskPhase.RUNNING)), null);
        decode.onWorkerStatusUpdate(decodeWs, running);
        scheduler.onWorkerStatusUpdate(running, decode);

        // The request leaves the running view WITH its finished record — the
        // ordinary decode terminal, not a vanish.
        TaskInfo finished = task(311, TaskPhase.RUNNING);
        finished.setBatchId(batchId);
        finished.setErrorCode(0L);
        WorkerStatusResponse finishedReport = decodeStatus(null, Map.of("311", finished));
        decode.onWorkerStatusUpdate(decodeWs, finishedReport);
        scheduler.onWorkerStatusUpdate(finishedReport, decode);

        assertEquals(0, scheduler.getInflightSize());
        scheduler.reportBatchMetrics();
        verify(reporter, never()).reportSchedulerInflightAuditRelease(anyLong());
    }

    @Test
    void reconciliationFencedEntry_isNeverTouchedByVanishSettle() throws Exception {
        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            cancelCalls.incrementAndGet();
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.accepted());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        decode.reserve(320, 128, 136, 50, 0);

        BatchItem item = new BatchItem(context(320), new CompletableFuture<>(),
                successRoute(320),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 320),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 320),
                prefill, decode, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        // The cancel fence (dispatch reconciliation) is set before the first
        // Cancel is issued, so one observed Cancel proves the fence is up.
        awaitCondition(() -> cancelCalls.get() >= 1);

        // Endpoint-level vanish signal for the fenced request id: track on the
        // decode registry, then drop it from the report entirely. The running
        // observation deliberately bypasses the scheduler so the legacy
        // Decode-ownership shortcut cannot clear the fence underneath us.
        decode.onWorkerStatusUpdate(decodeWs,
                decodeStatus(Map.of("320", task(320, TaskPhase.RUNNING)), null));
        WorkerStatusResponse empty = decodeStatus(null, null);
        decode.onWorkerStatusUpdate(decodeWs, empty);
        scheduler.onWorkerStatusUpdate(empty, decode);

        assertEquals(1, scheduler.getInflightSize(),
                "a reconciliation-fenced entry must never be force-settled");
        assertFalse(item.future().isDone());

        scheduler.reportBatchMetrics();
        verify(reporter, never()).reportSchedulerInflightAuditRelease(anyLong());
    }

    @Test
    void preAckEntry_isNotSettledByVanishSignal() throws Exception {
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);

        // Registered but never dispatched: the future is still pending.
        BatchItem item = new BatchItem(context(321), new CompletableFuture<>(),
                successRoute(321),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 321),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 321),
                prefill, decode, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));

        // Decode ownership observed (DECODE_OWNED), then the request vanishes.
        WorkerStatusResponse running = decodeStatus(
                Map.of("321", task(321, TaskPhase.RUNNING)), null);
        decode.onWorkerStatusUpdate(decodeWs, running);
        scheduler.onWorkerStatusUpdate(running, decode);
        WorkerStatusResponse empty = decodeStatus(null, null);
        decode.onWorkerStatusUpdate(decodeWs, empty);
        scheduler.onWorkerStatusUpdate(empty, decode);

        assertEquals(1, scheduler.getInflightSize(),
                "pre-ACK entries (future pending) keep their ordinary lifecycle");
        assertFalse(item.future().isDone());
    }

    @Test
    void requeuedToReceived_isNotTreatedAsVanished() {
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);

        decode.onWorkerStatusUpdate(decodeWs,
                decodeStatus(Map.of("330", task(330, TaskPhase.RUNNING)), null));
        assertTrue(decode.isEngineConfirmed(330));

        // Requeue: still present in the report, only demoted to RECEIVED.
        decode.onWorkerStatusUpdate(decodeWs,
                decodeStatus(Map.of("330", task(330, TaskPhase.RECEIVED)), null));
        assertFalse(decode.isEngineConfirmed(330));
        assertTrue(decode.drainVanishedEngineConfirmed().isEmpty(),
                "a requeued-but-reported request must not raise a vanish signal");
    }

    // ==================== helpers ====================

    private static WorkerStatusResponse decodeStatus(Map<String, TaskInfo> running,
                                                     Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        return response;
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        return task;
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
            response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                    .setRequestId(input.getRequestId())
                    .build());
        }
        return response.build();
    }

    private static List<EngineRpcService.GenerateInputPB> batchInputs(
            EngineRpcService.EnqueueBatchRequestPB request) {
        return request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(EngineRpcService.EnqueueBatchExternalInputPB::getInput)
                .toList();
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static byte[] generateInputBytes(long requestId) {
        return EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build()
                .toByteArray();
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort,
                                       int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }

    private PrefillEndpoint replacePrefillEndpoint() {
        WorkerStatus ws = new WorkerStatus();
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        ws.setAlive(true);
        return (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, "10.0.0.1:8080", ws);
    }

    private DecodeEndpoint ensureDecodeEndpoint(String ip, int httpPort, int grpcPort) {
        WorkerStatus ws = new WorkerStatus();
        ws.setIp(ip);
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        ws.setAlive(true);
        decodeWs = ws;
        return (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, ip + ":" + httpPort, ws);
    }

    private static void awaitCondition(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (!condition.getAsBoolean() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true before timeout");
    }
}
