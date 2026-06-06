package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Base64;
import java.util.List;
import java.util.Map;
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
    private final List<String> sentEndpoints = new CopyOnWriteArrayList<>();

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
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
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
        List<EngineRpcService.GenerateInputPB> inputs = batchInputs(batch);
        assertEquals(1, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(2, batch.getDpSlots(0).getRequestsCount());
        assertEquals(2, inputs.size());
        assertEquals(2, inputs.get(0).getBatchGroupSize());
        assertEquals(batch.getBatchId(), inputs.get(0).getBatchGroupId().getValue());
        assertEquals(batch.getBatchId(), inputs.get(1).getBatchGroupId().getValue());
        assertEquals(1, inputs.get(0).getGenerateConfig().getForceBatch().getValue());
        assertEquals(77, inputs.get(0).getGenerateConfig().getBatchGroupTimeout().getValue());
        assertEquals(2, inputs.get(0).getGenerateConfig().getRoleAddrsCount());
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.PREFILL,
                inputs.get(0).getGenerateConfig().getRoleAddrs(0).getRole());
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.DECODE,
                inputs.get(0).getGenerateConfig().getRoleAddrs(1).getRole());
    }

    @Test
    void submit_groups_batch_payload_by_dp_rank() throws Exception {
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            return successRouteWithPrefillDp(requestId, requestId == 71L ? 0 : 1);
        });

        CompletableFuture<Response> first = scheduler.submit(context(71));
        CompletableFuture<Response> second = scheduler.submit(context(72));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(1, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB batch = sentBatches.getFirst();
        assertEquals(2, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(1, batch.getDpSlots(1).getDpRank());
        assertEquals(1, batch.getDpSlots(0).getRequestsCount());
        assertEquals(1, batch.getDpSlots(1).getRequestsCount());
    }

    @Test
    void batch_enqueue_error_list_fails_only_rejected_request() throws Exception {
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.BatchEnqueueRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    List<EngineRpcService.GenerateInputPB> inputs = batchInputs(request);
                    return EngineRpcService.BatchEnqueueResponsePB.newBuilder()
                            .setBatchId(request.getBatchId())
                            .addSuccesses(EngineRpcService.BatchEnqueueSuccessPB.newBuilder()
                                    .setRequestId(inputs.get(0).getRequestId())
                                    .build())
                            .addErrors(EngineRpcService.BatchEnqueueErrorPB.newBuilder()
                                    .setRequestId(inputs.get(1).getRequestId())
                                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                            .setErrorCode(13)
                                            .setErrorMessage("decode alloc failed")
                                            .build())
                                    .build())
                            .build();
                });

        CompletableFuture<Response> first = scheduler.submit(context(81));
        CompletableFuture<Response> second = scheduler.submit(context(82));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertThrows(Exception.class, () -> second.get(2, TimeUnit.SECONDS));
    }

    @Test
    void batch_enqueue_missing_success_fails_missing_request() throws Exception {
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.BatchEnqueueRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.BatchEnqueueRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    List<EngineRpcService.GenerateInputPB> inputs = batchInputs(request);
                    return EngineRpcService.BatchEnqueueResponsePB.newBuilder()
                            .setBatchId(request.getBatchId())
                            .addSuccesses(EngineRpcService.BatchEnqueueSuccessPB.newBuilder()
                                    .setRequestId(inputs.get(0).getRequestId())
                                    .build())
                            .build();
                });

        CompletableFuture<Response> first = scheduler.submit(context(83));
        CompletableFuture<Response> second = scheduler.submit(context(84));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        Exception thrown = assertThrows(Exception.class, () -> second.get(2, TimeUnit.SECONDS));
        assertTrue(thrown.getCause().getMessage().contains("BatchEnqueue missing ack for request 84"));
    }

    @Test
    void dispatch_uses_dp0_entrypoint_when_status_available() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> successRouteWithPrefillDp(91, 1));

        WorkerStatus dp0 = new WorkerStatus();
        dp0.setIp("10.0.0.9");
        dp0.setPort(8090);
        dp0.setDpRank(0);
        dp0.getStatusVersion().set(1);
        when(engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, "g1"))
                .thenReturn(Map.of("10.0.0.9:8090", dp0));

        Response response = scheduler.submit(context(91)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        assertEquals("10.0.0.9:8091", sentEndpoints.getFirst());
        assertEquals(1, sentBatches.getFirst().getDpSlots(0).getDpRank());
    }

    @Test
    void dispatch_falls_back_to_selected_prefill_when_dp0_status_not_synced() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> successRouteWithPrefillDp(92, 1));

        WorkerStatus unsyncedDp0 = new WorkerStatus();
        unsyncedDp0.setIp("10.0.0.9");
        unsyncedDp0.setPort(8090);
        unsyncedDp0.setDpRank(0);
        when(engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, "g1"))
                .thenReturn(Map.of("10.0.0.9:8090", unsyncedDp0));

        Response response = scheduler.submit(context(92)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        assertEquals("10.0.0.1:9080", sentEndpoints.getFirst());
        assertEquals(1, sentBatches.getFirst().getDpSlots(0).getDpRank());
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
        assertEquals(1, batchInputs(sentBatches.get(0)).size());
        assertEquals(1, batchInputs(sentBatches.get(1)).size());
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

        assertTrue(sentBatches.stream().anyMatch(b -> batchInputs(b).size() == 2),
                "Budget allows ~5450 tokens: 2 items of 2000 = 4000 fits, 3 items = 6000 exceeds");
        int totalInputs = sentBatches.stream()
                .mapToInt(b -> batchInputs(b).size()).sum();
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

        assertTrue(sentBatches.stream().anyMatch(b -> batchInputs(b).size() == 4),
                "Should dispatch when picked.size() reaches batchSizeMax despite low fill ratio");
    }

    @Test
    void processQueue_park_converges_to_urgent_dispatch() throws Exception {
        // budget = sloMs(300) - predMs(128) = 172ms, margin = 100ms
        // fillThreshold=2.0 → fillRatio can never reach it (max 1.0)
        // batchSizeMax=1000 → single request can't trigger size condition
        // So request parks, budget shrinks each 1ms iteration, after ~72ms budget < margin → urgent dispatch
        config.setCostSloMs(300L);
        config.setCostSloRiskMarginMs(100L);
        config.setFlexlbBatchFillThreshold(2.0);
        config.setFlexlbBatchSizeMax(1000);

        CompletableFuture<Response> future = scheduler.submit(context(901));

        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size());
        assertEquals(1, batchInputs(sentBatches.getFirst()).size());
    }

    @Test
    void processQueue_fillRatio_triggers_dispatch() throws Exception {
        // budget = sloMs(500) - predMs(128) = 372ms, margin = 50ms
        // binary search: lo=128, hi=500, converges to ~322 → batchMaxTokens ≈ 322
        // fillRatio = 128/322 ≈ 0.40 >= threshold(0.3) → dispatches immediately via fillRatio
        // batchSizeMax=1000 ensures size condition is NOT the trigger
        config.setCostSloMs(500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(500);
        config.setFlexlbBatchFillThreshold(0.3);
        config.setFlexlbBatchSizeMax(1000);

        CompletableFuture<Response> future = scheduler.submit(context(1001));

        assertTrue(future.get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size());
        assertEquals(1, batchInputs(sentBatches.getFirst()).size());
    }

    @Test
    void processQueue_binary_search_lo_equals_hi() throws Exception {
        // headTokens(500) > maxCapacity(100) → lo > hi, binary search loop never executes
        // batchMaxTokens = max(headTokens, lo) = 500
        // greedy: 500 + 50 = 550 > 500 → second request doesn't fit
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(100);
        config.setFlexlbBatchFillThreshold(2.0);
        config.setFlexlbBatchSizeMax(100);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(1101, 500));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(1102, 50));
        Thread.sleep(50);
        config.setFlexlbBatchFillThreshold(0.01);

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(2, sentBatches.size());
        assertEquals(1, batchInputs(sentBatches.get(0)).size(),
                "Head alone — headTokens exceeds maxCapacity so no room for others");
        assertEquals(1, batchInputs(sentBatches.get(1)).size());
    }

    @Test
    void processQueue_maxScan_limits_greedy_fill() throws Exception {
        // scanAhead=1: greedy only checks 1 candidate beyond head
        // 3 requests queued, but only head + 1 scanned → first batch has 2 inputs, not 3
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchScanAhead(1);
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchFillThreshold(2.0);

        CompletableFuture<Response> f1 = scheduler.submit(context(1201));
        CompletableFuture<Response> f2 = scheduler.submit(context(1202));
        CompletableFuture<Response> f3 = scheduler.submit(context(1203));
        Thread.sleep(50);
        config.setFlexlbBatchFillThreshold(0.001);

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f3.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(2, sentBatches.size(), "maxScan=1 splits 3 requests into 2 batches");
        assertEquals(2, batchInputs(sentBatches.get(0)).size(),
                "First batch: head + 1 scanned candidate");
        assertEquals(1, batchInputs(sentBatches.get(1)).size());
    }

    @Test
    void processQueue_greedy_skips_oversized_and_picks_smaller() throws Exception {
        // Head=150tok, candidates: 120tok (too large: 150+120=270>200) and 50tok (fits: 150+50=200≤200)
        // Greedy skips 120-token and picks 50-token regardless of PQ iteration order
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(200);
        config.setFlexlbBatchScanAhead(10);
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchFillThreshold(2.0);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(1301, 150));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(1302, 120));
        CompletableFuture<Response> f3 = scheduler.submit(contextWithSeqLen(1303, 50));
        Thread.sleep(50);
        config.setFlexlbBatchFillThreshold(0.01);

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f3.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(2, sentBatches.size());
        EngineRpcService.BatchEnqueueRequestPB firstBatch = sentBatches.get(0);
        assertEquals(2, batchInputs(firstBatch).size(), "Head(150) + small(50) fit; large(120) skipped");

        long[] ids = batchInputs(firstBatch).stream()
                .mapToLong(EngineRpcService.GenerateInputPB::getRequestId).sorted().toArray();
        assertEquals(1301, ids[0], "150-token head in first batch");
        assertEquals(1303, ids[1], "50-token picked over 120-token");

        assertEquals(1, batchInputs(sentBatches.get(1)).size());
        assertEquals(1302, batchInputs(sentBatches.get(1)).get(0).getRequestId(),
                "Skipped 120-token dispatches alone in second batch");
    }

    @Test
    void processQueue_bsIter_exhaustion_uses_conservative_bound() throws Exception {
        // bsIter=1 with huge maxCapacity: binary search does only 1 step
        // mid ≈ 50050, estimateMs(50050) >> budget(350) → hi drops, lo stays at headTokens(100)
        // batchMaxTokens = 100, so second 100-token request doesn't fit (100+100=200>100)
        config.setCostSloMs(500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchSearchIter(1);
        config.setFlexlbBatchMaxCapacity(100000);
        config.setFlexlbBatchFillThreshold(2.0);
        config.setFlexlbBatchSizeMax(100);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(1401, 100));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(1402, 100));
        Thread.sleep(50);
        config.setFlexlbBatchFillThreshold(0.5);

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(2, sentBatches.size(),
                "bsIter=1 yields conservative batchMaxTokens=headTokens, preventing batching");
        assertEquals(1, batchInputs(sentBatches.get(0)).size());
        assertEquals(1, batchInputs(sentBatches.get(1)).size());
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
        assertEquals(2, batchInputs(sentBatches.getFirst()).size());
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
        for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
            response.addSuccesses(EngineRpcService.BatchEnqueueSuccessPB.newBuilder()
                    .setRequestId(input.getRequestId())
                    .build());
        }
        return response.build();
    }

    private static List<EngineRpcService.GenerateInputPB> batchInputs(
            EngineRpcService.BatchEnqueueRequestPB request) {
        return request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(EngineRpcService.BatchEnqueueExternalInputPB::getInput)
                .toList();
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
        return successRouteWithPrefillDp(requestId, 0);
    }

    private static Response successRouteWithPrefillDp(long requestId, long dpRank) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 9080, requestId, dpRank),
                server(RoleType.DECODE, "10.0.0.2", 8081, 9081, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        return server(role, ip, httpPort, grpcPort, requestId, 0);
    }

    private static ServerStatus server(RoleType role,
                                       String ip,
                                       int httpPort,
                                       int grpcPort,
                                       long requestId,
                                       long dpRank) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(dpRank);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }
}
