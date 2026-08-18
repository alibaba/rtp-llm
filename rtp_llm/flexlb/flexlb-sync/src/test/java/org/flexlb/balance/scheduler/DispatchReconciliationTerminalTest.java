package org.flexlb.balance.scheduler;

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
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Dispatch reconciliation terminal-settlement coverage: fix A (target
 * deregistration, 15s grace) and fix B (consecutive-failure backstop, 36
 * failures at the ~3.2s backoff ceiling) are hardcoded bounds too slow to
 * drive in a unit test, so these cases pin the sub-bound behavior: a
 * registered target and a NOT_FOUND below the cap must keep the retry
 * chain alive, and terminal entries must ignore late Cancel completions.
 */
class DispatchReconciliationTerminalTest {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private EngineCancelChannel cancelChannel;
    private WorkerStatus prefillWorkerStatus;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        cancelChannel = mock(EngineCancelChannel.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        // Every enqueue loses its ACK, forcing dispatch reconciliation.
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(new TimeoutException("lost ack")));
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, null, null, cancelChannel);

        prefillWorkerStatus = new WorkerStatus();
        prefillWorkerStatus.setIp("10.0.0.1");
        prefillWorkerStatus.setPort(8080);
        prefillWorkerStatus.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, "10.0.0.1:8080", prefillWorkerStatus);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    // ==================== Fix A: target deregistered ====================

    @Test
    void targetStillRegistered_fixADoesNotFire() throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.failed()));

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(402, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));

        Thread.sleep(400);
        assertFalse(item.future().isDone(),
                "a registered target must keep the legacy retry loop alive");
        assertEquals(1, scheduler.getInflightSize());
    }

    // ==================== Fix B: consecutive failure cap (hardcoded 36) ====================

    @Test
    void notFound_countsTowardCapButRetainsEntryBeforeCap() throws Exception {
        // Default gates (36 failures / 15s grace): a bare NOT_FOUND installs
        // no absent fence on the engine, so the entry must survive early
        // retries exactly like the legacy semantics.
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.notFound()));

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(405, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));

        Thread.sleep(300);
        assertFalse(item.future().isDone());
        assertEquals(1, scheduler.getInflightSize(),
                "NOT_FOUND below the cap must not settle the entry");
    }

    // ==================== Regression: normal exits unaffected ====================

    @Test
    void normalTombstonePath_unaffected() throws Exception {
        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            cancelCalls.incrementAndGet();
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.tombstoned());
        });

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(406, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        awaitInflightDrained();
        assertEquals(1, cancelCalls.get(), "TOMBSTONED must settle on the first try");
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void lateFailedOutcomeAfterSettlementIsNoOp() throws Exception {
        // Race: a Cancel is in flight when typed CANCELED(8429) settles the
        // entry. The late failed() completion must be absorbed by the
        // liveness guard and must not touch the failure counter or re-settle.
        AtomicInteger cancelCalls = new AtomicInteger();
        CompletableFuture<EngineCancelChannel.CancelOutcome> pending =
                new CompletableFuture<>();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            cancelCalls.incrementAndGet();
            return pending;
        });
        List<EngineRpcService.EnqueueBatchRequestPB> sent = new java.util.ArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(407, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sent.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sent.getLast().getBatchId();

        scheduler.onWorkerStatusUpdate(prefillFinished(
                407, batchId, 8429, PriorityPreemptionProgress.CANCELED));
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        awaitInflightDrained();

        int callsAtTerminal = cancelCalls.get();
        pending.complete(EngineCancelChannel.CancelOutcome.failed());
        Thread.sleep(250);
        assertEquals(callsAtTerminal, cancelCalls.get(),
                "a late failed outcome after settlement must be a no-op");
        assertEquals(0, scheduler.getInflightSize());
    }

    // ==================== Helpers (mirroring FlexlbBatchSchedulerTest) ====================

    private void awaitInflightDrained() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 1_000;
        while (scheduler.getInflightSize() != 0
                && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        assertEquals(0, scheduler.getInflightSize());
    }

    private BatchItem reconciliationItem(long requestId, PrefillEndpoint endpoint) {
        return new BatchItem(context(requestId), new CompletableFuture<>(),
                successRoute(requestId),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId),
                endpoint, null, System.currentTimeMillis());
    }

    private static WorkerStatusResponse prefillFinished(
            long requestId,
            long batchId,
            long errorCode,
            PriorityPreemptionProgress progress) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(requestId);
        finished.setBatchId(batchId);
        finished.setErrorCode(errorCode);
        finished.setPriorityPreemptionProgress(progress);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.PREFILL);
        status.setFinishedTaskInfo(Map.of(Long.toString(requestId), finished));
        return status;
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
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
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
}
