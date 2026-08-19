package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FlexlbBatchSchedulerTest {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private EngineCancelChannel cancelChannel;
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();
    private final List<String> sentEndpoints = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        cancelChannel = mock(EngineCancelChannel.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.tombstoned()));

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            reserveDefaultDecode(ctx);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.completedFuture(ackFor(request));
                });
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, null, null, cancelChannel);

        // Create endpoint and batcher for the worker that successRoute() returns
        String ipPort = "10.0.0.1:8080";
        WorkerStatus ws = new WorkerStatus();
        ws.setRole(RoleType.PREFILL);
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setRole(RoleType.PREFILL);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);
        ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void submit_flushes_grouped_requests_with_force_batch_payload() throws Exception {
        CompletableFuture<Response> first = scheduler.submit(contextWithLegacyBatchFields(1));
        assertFalse(first.isDone());

        CompletableFuture<Response> second = scheduler.submit(contextWithLegacyBatchFields(2));

        Response firstResponse = first.get(2, TimeUnit.SECONDS);
        Response secondResponse = second.get(2, TimeUnit.SECONDS);
        assertTrue(firstResponse.isSuccess());
        assertTrue(secondResponse.isSuccess());
        assertTrue(firstResponse.isEnqueuedByMaster());
        assertTrue(secondResponse.isEnqueuedByMaster());

        assertEquals(1, sentBatches.size());
        EngineRpcService.EnqueueBatchRequestPB batch = sentBatches.getFirst();
        List<EngineRpcService.GenerateInputPB> inputs = batchInputs(batch);
        assertEquals(1, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(2, batch.getDpSlots(0).getRequestsCount());
        assertEquals(2, inputs.size());
        assertEquals(0, inputs.get(0).getGroupSize());
        assertEquals(0, inputs.get(1).getGroupSize());
        assertFalse(inputs.get(0).hasGroupId());
        assertFalse(inputs.get(1).hasGroupId());
        assertEquals(1, legacyForceBatchValue(inputs.get(0).getGenerateConfig()));
        assertEquals(77, inputs.get(0).getGenerateConfig().getGroupTimeout().getValue());
        assertEquals(2, inputs.get(0).getGenerateConfig().getRoleAddrsCount());
        assertEquals("PREFILL",
                inputs.get(0).getGenerateConfig().getRoleAddrs(0).getRoleStr());
        assertEquals("DECODE",
                inputs.get(0).getGenerateConfig().getRoleAddrs(1).getRoleStr());
    }

    @Test
    void submit_groups_batch_payload_by_dp_rank() throws Exception {
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            reserveDefaultDecode(ctx);
            return successRouteWithPrefillDp(requestId, requestId == 71L ? 0 : 1);
        });

        CompletableFuture<Response> first = scheduler.submit(context(71));
        CompletableFuture<Response> second = scheduler.submit(context(72));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());

        assertEquals(1, sentBatches.size());
        EngineRpcService.EnqueueBatchRequestPB batch = sentBatches.getFirst();
        assertEquals(2, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(1, batch.getDpSlots(1).getDpRank());
        assertEquals(1, batch.getDpSlots(0).getRequestsCount());
        assertEquals(1, batch.getDpSlots(1).getRequestsCount());
    }

    @Test
    void decodeDispatchLimit_sameDpBatchSendsOnlyFreeSlot_thenDispatchesEachItemOnce()
            throws Exception {
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchSizeMax(20);
        config.setFlexlbBatchFixedWaitMs(60_000);
        config.setDecodeConcurrencyLimit(5);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);

        for (long requestId = 9_000; requestId < 9_004; requestId++) {
            decode.reserve(requestId, 128, 136, 30, 0);
        }
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            decode.reserve(requestId, 128, 136, 50, 0);
            decode.markQueuedPhase(requestId);
            return successRoute(requestId);
        });

        List<Long> requestIds = java.util.stream.LongStream.range(1_000, 1_020)
                .boxed().toList();
        List<CompletableFuture<Response>> futures = requestIds.stream()
                .map(requestId -> scheduler.submit(context(requestId)))
                .toList();

        awaitCondition(() -> sentBatches.size() == 1
                && prefill.getBatcher().dispatchPendingSize() == 0
                && prefill.getBatcher().queueSize() == 19);
        List<Long> firstSent = batchInputs(sentBatches.getFirst()).stream()
                .map(EngineRpcService.GenerateInputPB::getRequestId).toList();
        assertEquals(1, firstSent.size());
        assertEquals(5, decode.getEngineLoad());
        assertEquals(requestIds.stream().filter(id -> !firstSent.contains(id)).toList(),
                prefill.getBatcher().queueManager().snapshot().items().stream()
                        .map(item -> item.requestId()).toList(),
                "capacity-blocked members retain their original strict queue order");

        // Free exactly one slot. The next head may dispatch once, while all
        // other members remain charged and queued at the limit.
        decode.release(firstSent.getFirst());
        config.setFlexlbBatchSizeMax(19);
        awaitCondition(() -> sentBatches.size() == 2
                && prefill.getBatcher().dispatchPendingSize() == 0
                && prefill.getBatcher().queueSize() == 18
                && futures.stream().filter(CompletableFuture::isDone).count() >= 2);
        List<Long> allSent = sentBatches.stream()
                .flatMap(batch -> batchInputs(batch).stream())
                .map(EngineRpcService.GenerateInputPB::getRequestId)
                .toList();
        assertEquals(2, allSent.size());
        assertEquals(2, Set.copyOf(allSent).size(),
                "a CLAIMED member must never be restored and dispatched twice");
    }

    @Test
    void decodeDispatchClaimException_completesPendingAndTerminatesMember() throws Exception {
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchSizeMax(1);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint realDecode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        DecodeEndpoint throwingDecode = org.mockito.Mockito.spy(realDecode);
        long requestId = 2_100;
        realDecode.reserve(requestId, 128, 136, 50, 0);
        realDecode.markQueuedPhase(requestId);
        when(throwingDecode.tryClaimEngineDispatch(eq(requestId), anyLong()))
                .thenThrow(new IllegalStateException("claim failed"));

        BatchItem item = new BatchItem(context(requestId), new CompletableFuture<>(),
                successRoute(requestId),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId),
                prefill, throwingDecode, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        prefill.getBatcher().offer(item);

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        awaitCondition(() -> prefill.getBatcher().dispatchPendingSize() == 0
                && prefill.getBatcher().queueSize() == 0);
        assertEquals(0, scheduler.getInflightSize());
        assertFalse(realDecode.reservedView().containsKey(requestId));
        assertTrue(sentBatches.isEmpty());
    }

    @Test
    void decodeDispatchLimit_fullDpDoesNotBlockAnotherDpInSameBatch() throws Exception {
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchFixedWaitMs(60_000);
        config.setDecodeConcurrencyLimit(5);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint full = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        DecodeEndpoint available = ensureDecodeEndpoint("10.0.0.3", 8081, 8082);
        for (long requestId = 9_100; requestId < 9_105; requestId++) {
            full.reserve(requestId, 128, 136, 30, 0);
        }
        for (long requestId = 9_200; requestId < 9_204; requestId++) {
            available.reserve(requestId, 128, 136, 30, 0);
        }
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            DecodeEndpoint target = requestId == 2_001 ? full : available;
            target.reserve(requestId, 128, 136, 50, 0);
            target.markQueuedPhase(requestId);
            return successRouteWithDecode(requestId,
                    requestId == 2_001 ? "10.0.0.2" : "10.0.0.3");
        });

        CompletableFuture<Response> blocked = scheduler.submit(context(2_001));
        CompletableFuture<Response> allowed = scheduler.submit(context(2_002));

        assertTrue(allowed.get(2, TimeUnit.SECONDS).isSuccess());
        awaitCondition(() -> prefill.getBatcher().dispatchPendingSize() == 0
                && prefill.getBatcher().queueSize() == 1);
        assertFalse(blocked.isDone());
        assertEquals(List.of(2_002L), batchInputs(sentBatches.getFirst()).stream()
                .map(EngineRpcService.GenerateInputPB::getRequestId).toList());
        assertEquals(List.of(2_001L), prefill.getBatcher().queueManager().snapshot().items()
                .stream().map(item -> item.requestId()).toList());
    }

    @Test
    void batch_enqueue_error_list_fails_only_rejected_request() throws Exception {
        // Use request IDs to match, not input positions
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);

                    EngineRpcService.EnqueueBatchResponsePB.Builder response =
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());

                    for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
                        long reqId = input.getRequestId();
                        if (reqId == 81) {
                            response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                    .setRequestId(reqId).build());
                        } else {
                            response.addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                                    .setRequestId(reqId)
                                    .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                            .setErrorCode(13)
                                            .setErrorMessage("decode alloc failed")
                                            .build())
                                    .build());
                        }
                    }
                    return CompletableFuture.completedFuture(response.build());
                });

        CompletableFuture<Response> first = scheduler.submit(context(81));
        CompletableFuture<Response> second = scheduler.submit(context(82));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertFalse(second.get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void batch_enqueue_missing_success_reconciles_missing_request() throws Exception {
        // Only return success for request 83, missing ack for 84
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);

                    EngineRpcService.EnqueueBatchResponsePB.Builder response =
                            EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());

                    for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
                        if (input.getRequestId() == 83) {
                            response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                    .setRequestId(83).build());
                        }
                    }
                    return CompletableFuture.completedFuture(response.build());
                });

        CompletableFuture<Response> first = scheduler.submit(context(83));
        CompletableFuture<Response> second = scheduler.submit(context(84));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        Response secondResp = second.get(2, TimeUnit.SECONDS);
        assertFalse(secondResp.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), secondResp.getCode());
    }

    @Test
    void worker_completion_before_enqueue_ack_still_completes_schedule_future() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> ackFuture = new CompletableFuture<>();
        CountDownLatch enqueueStarted = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    enqueueStarted.countDown();
                    return ackFuture;
                });

        CompletableFuture<Response> scheduleFuture = scheduler.submit(context(85));
        assertTrue(enqueueStarted.await(2, TimeUnit.SECONDS));
        long batchId = sentBatches.getFirst().getBatchId();

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(85L);
        finished.setBatchId(batchId);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of("85", finished));
        applyWorkerStatus(status);

        // Decode completion is terminal: the schedule future completes right away
        // without waiting for the EnqueueBatch ack.
        Response response = scheduleFuture.get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(RequestLifecycleState.COMPLETED,
                scheduler.getRequestState(85L, batchId).state());

        // The late ack is ignored gracefully and does not disturb the terminal state.
        ackFuture.complete(ackFor(sentBatches.getFirst()));
        assertEquals(RequestLifecycleState.COMPLETED,
                scheduler.getRequestState(85L, batchId).state());
    }

    @Test
    void route_failure_completes_without_batch_enqueue() throws Exception {
        Response failure = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
        when(router.route(any(BalanceContext.class))).thenReturn(failure);

        Response response = scheduler.submit(context(21)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void submit_rejects_when_global_inflight_limit_reached() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchMaxInflight(1);

        CountDownLatch batchBlocked = new CountDownLatch(1);
        CountDownLatch releaseBlock = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    batchBlocked.countDown();
                    assertTrue(releaseBlock.await(5, TimeUnit.SECONDS));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    return CompletableFuture.completedFuture(ackFor(request));
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

        CompletableFuture<Response> first = scheduler.submit(context(51));
        assertFalse(first.isDone());

        // Second submit should fail because queue is full (maxSize=1)
        CompletableFuture<Response> second = scheduler.submit(context(52));
        Response response = second.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
    }

    // ============ onOfferFailure error-code mapping (task10 P1-1) ============

    @Test
    void offer_failure_maps_token_capacity_exceeded_to_dedicated_error_code() throws Exception {
        BatchItem item = offerFailureItem(61);

        scheduler.onOfferFailure(item, new BatchTokenCapacityExceededException(
                "seq_len exceeds batch token capacity"));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_TOKEN_CAPACITY_EXCEEDED.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("batch token capacity"));
    }

    @Test
    void offer_failure_keeps_generic_dispatch_error_for_other_causes() throws Exception {
        BatchItem item = offerFailureItem(62);

        scheduler.onOfferFailure(item, new IllegalStateException("queue stopped"));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), response.getCode());
    }

    private BatchItem offerFailureItem(long requestId) {
        return batchItemWithDefaultEndpoints(
                requestId, endpointRegistry.getPrefill("10.0.0.1:8080"));
    }

    @Test
    void processQueue_park_converges_to_urgent_dispatch() throws Exception {
        // budget = sloMs(300) - predMs(128) = 172ms, margin = 100ms
        // fillThreshold=2.0 → fillRatio can never reach it (max 1.0)
        // batchSizeMax=1000 → single request can't trigger size condition
        // So request parks, budget shrinks each 1ms iteration, after ~72ms budget < margin → urgent dispatch
        config.setCostSloMs(300L);
        config.setCostSloRiskMarginMs(100L);
        config.setFlexlbBatchSizeMax(1000);

        CompletableFuture<Response> future = scheduler.submit(context(901));

        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size());
        assertEquals(1, batchInputs(sentBatches.getFirst()).size());
    }

    @Test
    void processQueue_fillRatio_triggers_dispatch() throws Exception {
        // budget = sloMs(500) - predMs(128) = 372ms, margin = 50ms
        // fillRatio = 128/322 ≈ 0.40 >= threshold(0.3) → dispatches immediately via fillRatio
        // batchSizeMax=1000 ensures size condition is NOT the trigger
        config.setCostSloMs(500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(500);
        config.setFlexlbBatchSizeMax(1000);

        CompletableFuture<Response> future = scheduler.submit(context(1001));

        assertTrue(future.get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size());
        assertEquals(1, batchInputs(sentBatches.getFirst()).size());
    }

    @Test
    void processQueue_dispatches_requests_within_budget() throws Exception {
        // With slo_budget batcher (default), two 100-token requests each have
        // budget ≈ 350ms (slo=500, margin=50, pred≈100). Both fit within the
        // incremental budget and are dispatched together in a single batch.
        // flexlbBatchScanAhead (default 64) determines how many candidates are
        // scanned per iteration.
        config.setCostSloMs(500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(100000);
        config.setFlexlbBatchSizeMax(100);

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(1401, 100));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(1402, 100));

        assertTrue(f1.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(2, TimeUnit.SECONDS).isSuccess());

        // Both requests fit within the incremental budget → 1 combined batch
        assertEquals(1, sentBatches.size(),
                "slo_budget dispatches both requests together when they fit within budget");
        assertEquals(2, batchInputs(sentBatches.get(0)).size());
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

        CompletableFuture<Response> f1 = scheduler.submit(contextWithSeqLen(601, 600));
        CompletableFuture<Response> f2 = scheduler.submit(contextWithSeqLen(602, 600));

        assertTrue(f1.get(3, TimeUnit.SECONDS).isSuccess());
        assertTrue(f2.get(3, TimeUnit.SECONDS).isSuccess());

        assertEquals(1, sentBatches.size());
        assertEquals(2, batchInputs(sentBatches.getFirst()).size());
    }

    @Test
    void mismatched_generate_input_request_id_fails_before_batch_enqueue() throws Exception {
        config.setFlexlbBatchSizeMax(1);

        CompletableFuture<Response> future = scheduler.submit(context(31, 999));

        Response response = future.get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    // ==================== BatchIdGenerator Snowflake uniqueness ====================

    @Test
    void batchIdGeneratorProducesUniqueIds() {
        BatchIdGenerator gen = new BatchIdGenerator("10.0.0.1", 7001);
        Set<Long> ids = new HashSet<>();
        for (int i = 0; i < 4000; i++) {
            long id = gen.nextBatchId();
            assertTrue(id > 0, "batch_id must be positive (not -1 default)");
            ids.add(id);
        }
        assertEquals(4000, ids.size());
    }

    @Test
    void batchIdGeneratorDifferentiatesMasters() {
        // Two masters with different IP:port should produce non-overlapping IDs
        BatchIdGenerator gen1 = new BatchIdGenerator("10.0.0.1", 7001);
        BatchIdGenerator gen2 = new BatchIdGenerator("10.0.0.2", 7001);

        // Even if called at the same millisecond, master_id bits differ
        Set<Long> ids1 = new HashSet<>();
        Set<Long> ids2 = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            ids1.add(gen1.nextBatchId());
            ids2.add(gen2.nextBatchId());
        }
        // No overlap between two different masters
        ids1.retainAll(ids2);
        assertTrue(ids1.isEmpty(), "Different masters must not produce overlapping batch IDs");
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(EngineRpcService.EnqueueBatchRequestPB request) {
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
        return context(requestId, requestId);
    }

    private static BalanceContext context(long requestId, long generateInputRequestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(generateInputRequestId));
        return ctx;
    }

    private static BalanceContext contextWithSeqLen(long requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static BalanceContext contextWithLegacyBatchFields(long requestId) {
        BalanceContext ctx = context(requestId);
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId, true));
        return ctx;
    }

    // ==================== P0-1: onTimeout terminal handling (PR-D) ====================

    @Test
    void onTimeout_beforeDispatchClaim_settlesPriorityAdmissionAsResourceExhausted() {
        // A timeout is locally terminal only before startDispatch assigns a
        // batch id. The engine provably cannot have observed this item yet.
        BatchItem item = offerFailureItem(301);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onTimeout(item, new TimeoutException("test EnqueueBatch deadline"));

        Response response = item.future().getNow(null);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());

        // Idempotent: a second timeout is also a no-op
        scheduler.onTimeout(item, new TimeoutException("second"));
        Response stillUnchanged = item.future().getNow(null);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), stillUnchanged.getCode());
    }

    @Test
    void admissionDeadline_betweenStartDispatchAndCommit_fencesLateEnqueue_thenSettles8431()
            throws Exception {
        config.setAutoTpmEnabled(true);

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = batchItemWithDefaultEndpoints(303, endpoint);
        assertTrue(scheduler.registerInflight(item));

        CountDownLatch dispatchClaimed = new CountDownLatch(1);
        CountDownLatch allowCommit = new CountDownLatch(1);
        PrefillTimePredictor predictor = endpoint.getPredictor();
        PrefillEndpoint blockingEndpoint = org.mockito.Mockito.spy(endpoint);
        when(blockingEndpoint.getPredictor()).thenAnswer(inv -> {
            dispatchClaimed.countDown();
            assertTrue(allowCommit.await(5, TimeUnit.SECONDS));
            return predictor;
        });
        BatchItem blockingItem = new BatchItem(item.ctx(), item.future(), item.routeResponse(),
                item.prefill(), item.decode(), blockingEndpoint, item.decodeEp(), item.enqueuedAtMs());
        scheduler.unregisterInflight(item);
        assertTrue(scheduler.registerInflight(blockingItem));

        CompletableFuture<Void> flush = CompletableFuture.runAsync(() ->
                scheduler.onBatchReady(List.of(blockingItem), new DispatchMeta("race", 0)));
        assertTrue(dispatchClaimed.await(2, TimeUnit.SECONDS));

        scheduler.onAdmissionDeadline(blockingItem.requestId(), blockingItem.future());
        Response fenced = blockingItem.future().get(2, TimeUnit.SECONDS);
        assertFalse(fenced.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), fenced.getCode());

        allowCommit.countDown();
        flush.get(2, TimeUnit.SECONDS);
        assertEquals(0, blockingEndpoint.getInflightBatchCount(),
                "TOMBSTONED proves no late Engine ownership and releases the ledger");

        config.setFlexlbBatchSloMaxInflightBatches(1);
        config.setFlexlbBatchFixedMaxInflightBatches(1);
        config.setFlexlbBatchSizeMax(1);
        assertTrue(scheduler.submit(context(304)).get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size(),
                "after authoritative settlement, maxInflight=1 admits the next batch");

        verify(cancelChannel).cancel(any(), eq(303L), anyLong());
    }

    @Test
    void cleanupInflight_betweenStartDispatchAndCommit_reconcilesBeforeFinalSend()
            throws Exception {
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(314L, endpoint);
        assertTrue(scheduler.registerInflight(item));

        CountDownLatch dispatchClaimed = new CountDownLatch(1);
        CountDownLatch allowPrediction = new CountDownLatch(1);
        PrefillTimePredictor predictor = endpoint.getPredictor();
        PrefillEndpoint blockingEndpoint = org.mockito.Mockito.spy(endpoint);
        when(blockingEndpoint.getPredictor()).thenAnswer(inv -> {
            dispatchClaimed.countDown();
            assertTrue(allowPrediction.await(5, TimeUnit.SECONDS));
            return predictor;
        });
        BatchItem blockingItem = new BatchItem(item.ctx(), item.future(), item.routeResponse(),
                item.prefill(), item.decode(), blockingEndpoint, item.decodeEp(),
                item.enqueuedAtMs());
        scheduler.unregisterInflight(item);
        assertTrue(scheduler.registerInflight(blockingItem));

        CompletableFuture<Void> flush = CompletableFuture.runAsync(() ->
                scheduler.onBatchReady(List.of(blockingItem), new DispatchMeta("ttl_race", 0)));
        try {
            assertTrue(dispatchClaimed.await(2, TimeUnit.SECONDS));

            // Avoid a wall-clock sleep: after startDispatch owns a positive
            // batch id, a negative TTL makes this entry deterministically due.
            config.setFlexlbInflightTtlMs(-1);
            scheduler.cleanupInflight();

            verify(cancelChannel).cancel(any(), eq(314L), anyLong());
            Response response = blockingItem.future().get(1, TimeUnit.SECONDS);
            assertFalse(response.isSuccess());
            assertEquals(0, scheduler.getInflightSize());
            assertFalse(blockingItem.decodeEp().reservedView()
                    .containsKey(blockingItem.requestId()));
        } finally {
            allowPrediction.countDown();
        }

        flush.get(2, TimeUnit.SECONDS);
        assertTrue(sentBatches.isEmpty(),
                "the final ownership filter must drop a TTL-reconciled request");
    }

    @Test
    void dispatchUncertain_legacyNotFoundRetainsFutureAndBothLedgers() throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.notFound()));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = batchItemWithDefaultEndpoints(305, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sentBatches.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sentBatches.getLast().getBatchId();

        Thread.sleep(50);

        assertFalse(item.future().isDone());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, endpoint.getInflightBatchCount());
        config.setFlexlbInflightTtlMs(0);
        scheduler.cleanupInflight();
        assertEquals(1, scheduler.getInflightSize(), "TTL cannot break uncertain ownership");
    }

    @Test
    void cleanupInflight_afterDispatchClaimReconcilesBeforeReleasingAccounting()
            throws Exception {
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> enqueueAck =
                new CompletableFuture<>();
        CompletableFuture<EngineCancelChannel.CancelOutcome> cancelAck =
                new CompletableFuture<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sentBatches.add(inv.getArgument(2));
                    return enqueueAck;
                });
        when(cancelChannel.cancel(any(), eq(313L), anyLong())).thenReturn(cancelAck);

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(313L, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        awaitCondition(() -> !sentBatches.isEmpty());

        config.setFlexlbInflightTtlMs(0);
        Thread.sleep(2);
        scheduler.cleanupInflight();

        verify(cancelChannel).cancel(any(), eq(313L), anyLong());
        assertFalse(item.future().isDone(),
                "TTL cannot release a request after startDispatch");
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertTrue(item.decodeEp().reservedView().containsKey(item.requestId()));

        cancelAck.complete(EngineCancelChannel.CancelOutcome.tombstoned());
        assertFalse(item.future().get(1, TimeUnit.SECONDS).isSuccess());
        awaitCondition(() -> scheduler.getInflightSize() == 0);
        assertEquals(0, endpoint.getInflightBatchCount());
        assertFalse(item.decodeEp().reservedView().containsKey(item.requestId()));
    }

    @Test
    void dispatchUncertain_acceptedCancelWaitsForTypedPrefillFinished() throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted()));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = batchItemWithDefaultEndpoints(306, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sentBatches.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sentBatches.getLast().getBatchId();

        Thread.sleep(20);
        assertFalse(item.future().isDone());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(306L);
        finished.setBatchId(batchId);
        finished.setErrorCode(8429L);
        finished.setPriorityPreemptionProgress(PriorityPreemptionProgress.CANCELED);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.PREFILL);
        status.setFinishedTaskInfo(Map.of("306", finished));
        applyWorkerStatus(status);

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void dispatchUncertainAfterAcknowledgedFutureDoesNotStartCancelReconciliation()
            throws Exception {
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = batchItemWithDefaultEndpoints(307, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        long batchId = sentBatches.getLast().getBatchId();
        RequestLifecycleSnapshot acknowledged = scheduler.getRequestState(307L, batchId);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, acknowledged.state());

        scheduler.onDispatchUncertain(item, batchId, new RuntimeException("late callback"));

        verify(cancelChannel, never()).cancel(any(), eq(307L), anyLong());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(307L, batchId).state());
    }

    @Test
    void dispatchUncertain_acceptedRetriesUntilTombstoned() throws Exception {
        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv ->
                CompletableFuture.completedFuture(cancelCalls.getAndIncrement() == 0
                        ? EngineCancelChannel.CancelOutcome.accepted()
                        : EngineCancelChannel.CancelOutcome.tombstoned()));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(307, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(2, cancelCalls.get());
        long settlementDeadline = System.currentTimeMillis() + 1_000;
        while (scheduler.getInflightSize() != 0
                && System.currentTimeMillis() < settlementDeadline) {
            Thread.sleep(1);
        }
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, endpoint.getInflightBatchCount());

        Thread.sleep(250);
        assertEquals(2, cancelCalls.get(), "terminal entry must stop delayed retries");
    }

    @Test
    void dispatchUncertain_synchronousCancelThrowIsRetried() throws Exception {
        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            if (cancelCalls.getAndIncrement() == 0) {
                throw new IllegalStateException("sync transport failure");
            }
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.tombstoned());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(new TimeoutException("lost ack")));

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(308, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(2, cancelCalls.get());
        long settlementDeadline = System.currentTimeMillis() + 1_000;
        while (scheduler.getInflightSize() != 0
                && System.currentTimeMillis() < settlementDeadline) {
            Thread.sleep(1);
        }
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void dispatchUncertain_configFailureIsRetriedWithoutStrandingInflight() throws Exception {
        AtomicBoolean failNextConfigRead = new AtomicBoolean();
        when(configService.loadBalanceConfig()).thenAnswer(inv -> {
            if (failNextConfigRead.compareAndSet(true, false)) {
                throw new IllegalStateException("transient config failure");
            }
            return config;
        });
        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            cancelCalls.incrementAndGet();
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.tombstoned());
        });
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> ack =
                new CompletableFuture<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sentBatches.add(inv.getArgument(2));
                    return ack;
                });

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(311, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        awaitCondition(() -> !sentBatches.isEmpty());

        failNextConfigRead.set(true);
        ack.completeExceptionally(new TimeoutException("lost ack"));

        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        awaitCondition(() -> scheduler.getInflightSize() == 0);
        assertEquals(1, cancelCalls.get(), "the retry must reach the Cancel transport");
        assertEquals(0, endpoint.getInflightBatchCount());
        assertFalse(item.decodeEp().reservedView().containsKey(item.requestId()));
    }

    @Test
    void dispatchUncertain_retiredGenerationRetainsAccountingUntilExactDecodeOwnership()
            throws Exception {
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> ack =
                new CompletableFuture<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sentBatches.add(inv.getArgument(2));
                    return ack;
                });

        PrefillEndpoint original = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(310, original);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        awaitCondition(() -> !sentBatches.isEmpty());

        original.close();
        PrefillEndpoint replacement = replacePrefillEndpoint();
        assertFalse(replacement == original);
        ack.completeExceptionally(new TimeoutException("lost ack after restart"));

        Thread.sleep(20);
        assertFalse(item.future().isDone(),
                "Master retirement is not an Engine terminal proof");
        assertEquals(1, scheduler.getInflightSize());
        assertTrue(item.decodeEp().reservedView().containsKey(item.requestId()));
        verify(cancelChannel, never()).cancel(any(), eq(310L), anyLong());

        long batchId = sentBatches.getLast().getBatchId();
        applyWorkerStatus(decodeRunning(310L, batchId));

        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());
        assertTrue(item.decodeEp().isConfirmedTracked(item.requestId()));
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(item.requestId(), batchId).state());

        applyWorkerStatus(decodeFinished(310L, batchId, 0));
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void dispatchUncertain_retiredGenerationAcceptsExactDecodeTerminal() throws Exception {
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> ack =
                new CompletableFuture<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sentBatches.add(inv.getArgument(2));
                    return ack;
                });

        PrefillEndpoint original = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(312, original);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        awaitCondition(() -> !sentBatches.isEmpty());
        long batchId = sentBatches.getLast().getBatchId();

        original.close();
        replacePrefillEndpoint();
        ack.completeExceptionally(new TimeoutException("lost ack after restart"));
        Thread.sleep(20);
        assertFalse(item.future().isDone());

        applyWorkerStatus(decodeFinished(312L, batchId, 9001));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.WORKER_EXECUTION_FAILED.getErrorCode(),
                response.getCode());
        assertEquals(0, scheduler.getInflightSize());
        assertFalse(item.decodeEp().reservedView().containsKey(item.requestId()));
        verify(cancelChannel, never()).cancel(any(), eq(312L), anyLong());
    }

    @Test
    void dispatchUncertain_onlyMatchingTypedCanceled8429IsTerminal() throws Exception {
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

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = reconciliationItem(309, endpoint);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onBatchReady(List.of(item), new DispatchMeta("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sentBatches.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sentBatches.getLast().getBatchId();

        applyWorkerStatus(prefillFinished(
                309, batchId, 0, PriorityPreemptionProgress.NONE));
        applyWorkerStatus(prefillFinished(
                309, batchId, 500, PriorityPreemptionProgress.NONE));
        applyWorkerStatus(prefillFinished(
                309, batchId + 1, 8429, PriorityPreemptionProgress.CANCELED));
        assertFalse(item.future().isDone());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, endpoint.getInflightBatchCount());

        applyWorkerStatus(prefillFinished(
                309, batchId, 8429, PriorityPreemptionProgress.CANCELED));
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, endpoint.getInflightBatchCount());

        int callsAtTerminal = cancelCalls.get();
        Thread.sleep(250);
        assertEquals(callsAtTerminal, cancelCalls.get(),
                "a retry scheduled before typed CANCELED must become a no-op");
    }

    // ==================== P0-3: close/handover race (PR-D) ====================

    @Test
    void concurrentTimeout_exactlyOneTerminalVerb() throws Exception {
        // Two threads race to time out the same inflight entry. The
        // synchronized(entry) + RequestLifecycle.isTerminal() guard ensures
        // exactly one terminal verb settles the future (CAS-like idempotency).
        BatchItem item = offerFailureItem(302);
        assertTrue(scheduler.registerInflight(item));

        CompletableFuture<Void> t1 = CompletableFuture.runAsync(() ->
                scheduler.onTimeout(item, new TimeoutException("first")));
        CompletableFuture<Void> t2 = CompletableFuture.runAsync(() ->
                scheduler.onTimeout(item, new TimeoutException("second")));
        CompletableFuture.allOf(t1, t2).get(3, TimeUnit.SECONDS);

        assertTrue(item.future().isDone());
        Response response = item.future().getNow(null);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
    }

    private BatchItem reconciliationItem(long requestId, PrefillEndpoint endpoint) {
        return batchItemWithDefaultEndpoints(requestId, endpoint);
    }

    private BatchItem batchItemWithDefaultEndpoints(
            long requestId, PrefillEndpoint prefillEndpoint) {
        BalanceContext ctx = context(requestId);
        Response route = successRoute(requestId);
        DecodeEndpoint decodeEndpoint = reserveDefaultDecode(ctx);
        return new BatchItem(ctx, new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
                prefillEndpoint, decodeEndpoint, System.currentTimeMillis());
    }

    private void applyWorkerStatus(WorkerStatusResponse response) {
        WorkerStatus source;
        if (response.getRole() == RoleType.PREFILL) {
            PrefillEndpoint endpoint =
                    endpointRegistry.getPrefill("10.0.0.1:8080");
            source = endpoint.getStatus();
            endpoint.applyWorkerStatusResponse(source, response);
        } else if (response.getRole() == RoleType.DECODE) {
            DecodeEndpoint endpoint =
                    endpointRegistry.getDecode("10.0.0.2:8081");
            source = endpoint.getStatus();
            endpoint.applyWorkerStatusResponse(source, response);
        } else {
            throw new IllegalArgumentException("unsupported test role " + response.getRole());
        }
        scheduler.recordRequestActivity(source, response);
        scheduler.updateRequestLifecycleFromWorkerStatus(source, response);
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

    private static WorkerStatusResponse decodeRunning(long requestId, long batchId) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setBatchId(batchId);
        running.setPhase(TaskPhase.RUNNING);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setRunningTaskInfo(Map.of(Long.toString(requestId), running));
        return status;
    }

    private static WorkerStatusResponse decodeFinished(
            long requestId, long batchId, long errorCode) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(requestId);
        finished.setBatchId(batchId);
        finished.setErrorCode(errorCode);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of(Long.toString(requestId), finished));
        return status;
    }

    private static byte[] generateInputBytes(long requestId) {
        return generateInputBytes(requestId, false);
    }

    private static byte[] generateInputBytes(long requestId, boolean includeLegacyBatchFields) {
        EngineRpcService.GenerateConfigPB.Builder config = EngineRpcService.GenerateConfigPB.newBuilder()
                .setMaxNewTokens(8);
        if (includeLegacyBatchFields) {
            com.google.protobuf.UnknownFieldSet.Field forceBatch =
                    com.google.protobuf.UnknownFieldSet.Field.newBuilder()
                            .addLengthDelimited(com.google.protobuf.Int32Value.of(1).toByteString())
                            .build();
            config.setUnknownFields(com.google.protobuf.UnknownFieldSet.newBuilder()
                    .addField(55, forceBatch)
                    .build());
            config.setGroupTimeout(com.google.protobuf.Int32Value.of(77));
        }
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(config.build())
                .build();
        return input.toByteArray();
    }

    private static int legacyForceBatchValue(EngineRpcService.GenerateConfigPB config) throws Exception {
        return com.google.protobuf.Int32Value.parseFrom(
                config.getUnknownFields().getField(55).getLengthDelimitedList().get(0)).getValue();
    }

    private static Response successRoute(long requestId) {
        return successRouteWithPrefillDp(requestId, 0);
    }

    private static Response successRouteWithDecode(long requestId, String decodeIp) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, decodeIp, 8081, 8082, requestId)));
        return response;
    }

    private static Response successRouteWithPrefillDp(long requestId, long dpRank) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId, dpRank),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
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
        ws.setRole(RoleType.DECODE);
        ws.setIp(ip);
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        ws.setAlive(true);
        return (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, ip + ":" + httpPort, ws);
    }

    private DecodeEndpoint reserveDefaultDecode(BalanceContext ctx) {
        DecodeEndpoint endpoint = endpointRegistry.getDecode("10.0.0.2:8081");
        Request request = ctx.getRequest();
        endpoint.reserve(request.getRequestId(), request.getSeqLen(),
                request.getSeqLen() + request.getMaxNewTokens());
        return endpoint;
    }

    private static void awaitCondition(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (!condition.getAsBoolean() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true before timeout");
    }
}
