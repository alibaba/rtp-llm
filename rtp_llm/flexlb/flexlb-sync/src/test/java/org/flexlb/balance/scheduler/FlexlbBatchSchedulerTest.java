package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Base64;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbBatchSchedulerTest {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private EngineWorkerStatus engineWorkerStatus;
    private FlexlbBatchScheduler scheduler;
    private FlexlbConfig config;
    private final List<EngineRpcService.BatchEnqueueRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchFillThreshold(1.0);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.BatchEnqueueRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return ackFor(request);
                });
        when(grpcClient.cancel(anyString(), anyInt(), anyLong(), anyLong()))
                .thenReturn(EngineRpcService.EmptyPB.getDefaultInstance());

        scheduler = new FlexlbBatchScheduler(configService, router, grpcClient, engineWorkerStatus);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void submit_flushes_grouped_requests_with_force_batch_payload() throws Exception {
        CompletableFuture<Response> first = scheduler.submit(context(1));
        assertFalse(first.isDone());

        CompletableFuture<Response> second = scheduler.submit(context(2));

        Response firstResponse = first.get(2, TimeUnit.SECONDS);
        Response secondResponse = second.get(2, TimeUnit.SECONDS);
        assertTrue(firstResponse.isSuccess());
        assertTrue(secondResponse.isSuccess());
        assertTrue(firstResponse.isEnqueuedByMaster());
        assertTrue(secondResponse.isEnqueuedByMaster());

        assertEquals(1, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB batch = sentBatches.getFirst();
        assertEquals(2, batch.getInputsCount());
        assertEquals(2, batch.getInputs(0).getBatchGroupSize());
        assertEquals(batch.getBatchId(), batch.getInputs(0).getBatchGroupId().getValue());
        assertEquals(batch.getBatchId(), batch.getInputs(1).getBatchGroupId().getValue());
        assertEquals(1, batch.getInputs(0).getGenerateConfig().getForceBatch().getValue());
        assertEquals(77, batch.getInputs(0).getGenerateConfig().getBatchGroupTimeout().getValue());
        assertEquals(2, batch.getInputs(0).getGenerateConfig().getRoleAddrsCount());
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.PREFILL,
                batch.getInputs(0).getGenerateConfig().getRoleAddrs(0).getRole());
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.DECODE,
                batch.getInputs(0).getGenerateConfig().getRoleAddrs(1).getRole());
    }

    @Test
    void cancel_removes_request_before_batch_enqueue() throws Exception {
        CompletableFuture<Response> future = scheduler.submit(context(11));

        scheduler.cancel(11L);

        assertTrue(future.isCompletedExceptionally());
        assertThrows(CancellationException.class, () -> future.get(1, TimeUnit.SECONDS));
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void cancel_inflight_before_ack_completes_cancelled_and_sends_engine_cancel() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        CountDownLatch batchStarted = new CountDownLatch(1);
        CountDownLatch cancelSeen = new CountDownLatch(1);

        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.BatchEnqueueRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    batchStarted.countDown();
                    assertTrue(cancelSeen.await(2, TimeUnit.SECONDS));
                    return ackFor(request);
                });
        when(grpcClient.cancel(anyString(), anyInt(), anyLong(), anyLong()))
                .thenAnswer(inv -> {
                    cancelSeen.countDown();
                    return EngineRpcService.EmptyPB.getDefaultInstance();
                });

        CompletableFuture<Response> future = scheduler.submit(context(12));

        assertTrue(batchStarted.await(2, TimeUnit.SECONDS));
        scheduler.cancel(12L);

        assertThrows(CancellationException.class, () -> future.get(2, TimeUnit.SECONDS));
        verify(grpcClient, atLeastOnce()).cancel(anyString(), anyInt(), anyLong(), anyLong());
    }

    @Test
    void route_failure_completes_without_batch_enqueue() throws Exception {
        Response failure = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
        when(router.route(any(BalanceContext.class))).thenReturn(failure);

        Response response = scheduler.submit(context(21)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void submit_rejects_when_global_inflight_limit_reached() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchMaxInflight(1);

        CountDownLatch batchBlocked = new CountDownLatch(1);
        CountDownLatch releaseBlock = new CountDownLatch(1);
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    batchBlocked.countDown();
                    assertTrue(releaseBlock.await(5, TimeUnit.SECONDS));
                    EngineRpcService.BatchEnqueueRequestPB request = inv.getArgument(2);
                    return ackFor(request);
                });

        scheduler.submit(context(41));
        assertTrue(batchBlocked.await(2, TimeUnit.SECONDS));

        Response rejected = scheduler.submit(context(42)).get(1, TimeUnit.SECONDS);
        assertFalse(rejected.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), rejected.getCode());

        releaseBlock.countDown();
    }

    @Test
    void batcher_rejects_when_queue_full() throws Exception {
        config.setFlexlbBatchQueueMaxSize(1);
        config.setFlexlbBatchFillThreshold(1.0);

        CompletableFuture<Response> first = scheduler.submit(context(51));
        assertFalse(first.isDone());

        Response rejected = scheduler.submit(context(52)).get(1, TimeUnit.SECONDS);
        assertFalse(rejected.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), rejected.getCode());
    }

    @Test
    void processQueue_drops_expired_request() throws Exception {
        config.setCostSloMs(0L);
        config.setCostSloRiskMarginMs(0L);
        config.setFlexlbBatchFillThreshold(2.0);
        config.setFlexlbBatchSizeMax(10000);

        CompletableFuture<Response> future = scheduler.submit(context(100));

        assertThrows(Exception.class, () -> future.get(2, TimeUnit.SECONDS));
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void processQueue_dispatches_urgent_request_alone() throws Exception {
        config.setCostSloMs(5000L);
        config.setCostSloRiskMarginMs(10000L);
        config.setFlexlbBatchSizeMax(8);

        CompletableFuture<Response> f1 = scheduler.submit(context(201));
        assertFalse(f1.isDone());
        CompletableFuture<Response> f2 = scheduler.submit(context(202));

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(2, sentBatches.size());
        assertEquals(1, sentBatches.get(0).getInputsCount());
        assertEquals(1, sentBatches.get(1).getInputsCount());
    }

    @Test
    void processQueue_binary_search_limits_batch_by_token_budget() throws Exception {
        config.setCostSloMs(7500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFillThreshold(2.0);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(301, 2000));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(302, 2000));
        CompletableFuture<Response> f3 = scheduler.submit(contextWithSeqLen(303, 2000));
        Thread.sleep(50);

        config.setFlexlbBatchFillThreshold(0.3);

        f1.get(3, TimeUnit.SECONDS);
        f2.get(3, TimeUnit.SECONDS);
        f3.get(3, TimeUnit.SECONDS);

        assertTrue(sentBatches.stream().anyMatch(b -> b.getInputsCount() == 2),
                "Budget allows ~5450 tokens: 2 items of 2000 = 4000 fits, 3 items = 6000 exceeds");
        int totalInputs = sentBatches.stream()
                .mapToInt(EngineRpcService.BatchEnqueueRequestPB::getInputsCount).sum();
        assertEquals(3, totalInputs);
    }

    @Test
    void processQueue_waits_when_fill_below_threshold() throws Exception {
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchSizeMax(1000);
        config.setFlexlbBatchFillThreshold(0.9);

        CompletableFuture<Response> future = scheduler.submit(context(401));

        Thread.sleep(100);
        assertFalse(future.isDone(), "Should wait when fill ratio is below threshold");

        config.setFlexlbBatchFillThreshold(0.001);
        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void processQueue_dispatches_when_batch_size_reached() throws Exception {
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchSizeMax(4);
        config.setFlexlbBatchFillThreshold(0.99);

        CompletableFuture<Response> f1 = scheduler.submit(context(501));
        CompletableFuture<Response> f2 = scheduler.submit(context(502));
        CompletableFuture<Response> f3 = scheduler.submit(context(503));
        CompletableFuture<Response> f4 = scheduler.submit(context(504));

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f3.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f4.get(2, TimeUnit.SECONDS).isSuccess());

        assertTrue(sentBatches.stream().anyMatch(b -> b.getInputsCount() == 4),
                "Should dispatch when picked.size() reaches batchSizeMax despite low fill ratio");
    }

    @Test
    void resolveSloMs_uses_buckets_when_configured() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setCostSloMs(500L);
        cfg.setCostSloBuckets("4096:2000,32768:10000,131072:30000,524288:60000");

        assertEquals(2000L, cfg.resolveSloMs(100));
        assertEquals(2000L, cfg.resolveSloMs(4096));
        assertEquals(10000L, cfg.resolveSloMs(4097));
        assertEquals(10000L, cfg.resolveSloMs(32768));
        assertEquals(30000L, cfg.resolveSloMs(32769));
        assertEquals(30000L, cfg.resolveSloMs(131072));
        assertEquals(60000L, cfg.resolveSloMs(131073));
        assertEquals(60000L, cfg.resolveSloMs(1000000));
    }

    @Test
    void resolveSloMs_falls_back_to_costSloMs_when_no_buckets() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setCostSloMs(500L);
        cfg.setCostSloBuckets("");

        assertEquals(500L, cfg.resolveSloMs(100));
        assertEquals(500L, cfg.resolveSloMs(100000));
    }

    @Test
    void resolveSloMs_handles_unsorted_bucket_input() {
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setCostSloBuckets("131072:30000,4096:2000,32768:10000");

        assertEquals(2000L, cfg.resolveSloMs(1000));
        assertEquals(10000L, cfg.resolveSloMs(5000));
        assertEquals(30000L, cfg.resolveSloMs(50000));
    }

    @Test
    void dynamic_slo_prevents_drop_for_requests_exceeding_fixed_slo() throws Exception {
        // With default costSloMs=500 and alpha1=1.0, a 600-token request has
        // predMs=600 > sloMs=500 → budget=0 → immediate drop.
        // With buckets "1000:5000,...", sloMs=5000 → budget=4400 → enough to batch.
        config.setCostSloBuckets("1000:5000,100000:50000");
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchFillThreshold(1.0);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(601, 600));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(602, 600));

        assertTrue(f1.get(3, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(3, TimeUnit.SECONDS).isSuccess());

        assertEquals(1, sentBatches.size());
        assertEquals(2, sentBatches.getFirst().getInputsCount());
    }

    @Test
    void mismatched_generate_input_request_id_fails_before_batch_enqueue() {
        config.setFlexlbBatchSizeMax(1);

        CompletableFuture<Response> future = scheduler.submit(context(31, 999));

        assertThrows(Exception.class, () -> future.get(2, TimeUnit.SECONDS));
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    private static EngineRpcService.BatchEnqueueResponsePB ackFor(EngineRpcService.BatchEnqueueRequestPB request) {
        EngineRpcService.BatchEnqueueResponsePB.Builder response =
                EngineRpcService.BatchEnqueueResponsePB.newBuilder().setBatchId(request.getBatchId());
        for (EngineRpcService.GenerateInputPB input : request.getInputsList()) {
            response.addAcks(EngineRpcService.BatchEnqueueAckPB.newBuilder()
                    .setRequestId(input.getRequestId())
                    .build());
        }
        return response.build();
    }

    private static BalanceContext context(long requestId) {
        return context(requestId, requestId);
    }

    private static BalanceContext context(long requestId, long generateInputRequestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setGenerateInputPbB64(generateInput(generateInputRequestId));

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }

    private static BalanceContext contextWithSeqLen(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setGenerateInputPbB64(generateInput(requestId));

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }

    private static String generateInput(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .setBatchGroupTimeout(com.google.protobuf.Int32Value.of(77))
                        .build())
                .build();
        return Base64.getEncoder().encodeToString(input.toByteArray());
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 9080, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 9081, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }
}
