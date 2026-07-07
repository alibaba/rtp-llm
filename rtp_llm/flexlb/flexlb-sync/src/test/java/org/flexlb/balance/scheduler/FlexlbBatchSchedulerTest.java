package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    private BatchSchedulerReporter reporter;
    private FlexlbBatchScheduler scheduler;
    private FlexlbConfig config;
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();
    private final List<String> sentEndpoints = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        reporter = mock(BatchSchedulerReporter.class);

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
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sentEndpoints.add(inv.getArgument(0) + ":" + inv.getArgument(1));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return ackFor(request);
                });
        when(grpcClient.cancel(anyString(), anyInt(), anyLong(), anyLong()))
                .thenReturn(EngineRpcService.EmptyPB.getDefaultInstance());

        EndpointRegistry endpointRegistry = new EndpointRegistry(configService, null, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService);
        scheduler = new FlexlbBatchScheduler(configService, router, grpcClient, engineWorkerStatus,
                endpointRegistry, dispatcher, reporter);

        // Create endpoint and batcher for the worker that successRoute() returns
        String ipPort = "10.0.0.1:8080";
        WorkerStatus ws = new WorkerStatus();
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(9080);
        PrefillEndpoint endpoint = new PrefillEndpoint(ws, config, scheduler, reporter);
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(9080);
        prefill.setRole(RoleType.PREFILL);
        endpointRegistry.putPrefill(ipPort, endpoint);
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
        EngineRpcService.EnqueueBatchRequestPB batch = sentBatches.getFirst();
        List<EngineRpcService.GenerateInputPB> inputs = batchInputs(batch);
        assertEquals(1, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(2, batch.getDpSlots(0).getRequestsCount());
        assertEquals(2, inputs.size());
        assertEquals(2, inputs.get(0).getGroupSize());
        assertEquals(batch.getBatchId(), inputs.get(0).getGroupId().getValue());
        assertEquals(batch.getBatchId(), inputs.get(1).getGroupId().getValue());
        assertEquals(77, inputs.get(0).getGenerateConfig().getGroupTimeout().getValue());
        assertEquals(2, inputs.get(0).getGenerateConfig().getRoleAddrsCount());
        assertEquals(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                inputs.get(0).getGenerateConfig().getRoleAddrs(0).getRoleType());
        assertEquals(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                inputs.get(0).getGenerateConfig().getRoleAddrs(1).getRoleType());
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
        EngineRpcService.EnqueueBatchRequestPB batch = sentBatches.getFirst();
        assertEquals(2, batch.getDpSlotsCount());
        assertEquals(0, batch.getDpSlots(0).getDpRank());
        assertEquals(1, batch.getDpSlots(1).getDpRank());
        assertEquals(1, batch.getDpSlots(0).getRequestsCount());
        assertEquals(1, batch.getDpSlots(1).getRequestsCount());
    }

    @Test
    void batch_enqueue_error_list_fails_only_rejected_request() throws Exception {
        // Use request IDs to match, not input positions
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
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
                    return response.build();
                });

        CompletableFuture<Response> first = scheduler.submit(context(81));
        CompletableFuture<Response> second = scheduler.submit(context(82));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertFalse(second.get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void batch_enqueue_missing_success_fails_missing_request() throws Exception {
        // Only return success for request 83, missing ack for 84
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
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
                    return response.build();
                });

        CompletableFuture<Response> first = scheduler.submit(context(83));
        CompletableFuture<Response> second = scheduler.submit(context(84));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        Response secondResp = second.get(2, TimeUnit.SECONDS);
        assertFalse(secondResp.isSuccess());
        assertTrue(secondResp.getErrorMessage().contains("EnqueueBatch missing ack for request 84"));
    }

    @Test
    void dispatch_falls_back_to_selected_prefill_when_dp0_status_not_synced() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> successRouteWithPrefillDp(92, 1));

        WorkerStatus unsyncedDp0 = new WorkerStatus();
        unsyncedDp0.setIp("10.0.0.9");
        unsyncedDp0.setPort(8090);
        unsyncedDp0.setGrpcPort(9090);
        unsyncedDp0.setDpRank(0);
        PrefillEndpoint unsyncedEp = new PrefillEndpoint(unsyncedDp0, config, scheduler, reporter);
        when(engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, "g1"))
                .thenReturn(Map.of("10.0.0.9:8090", unsyncedEp));

        Response response = scheduler.submit(context(92)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        assertEquals("10.0.0.1:9080", sentEndpoints.getFirst());
        assertEquals(1, sentBatches.getFirst().getDpSlots(0).getDpRank());
    }

    @Test
    void cancel_removes_request_before_batch_enqueue() throws Exception {
        CompletableFuture<Response> future = scheduler.submit(context(11));

        scheduler.cancel(11L);

        assertTrue(future.isDone());
        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void cancel_inflight_before_ack_completes_cancelled_and_sends_engine_cancel() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        CountDownLatch batchStarted = new CountDownLatch(1);
        CountDownLatch cancelSeen = new CountDownLatch(1);

        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
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

        Response response = future.get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
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
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    batchBlocked.countDown();
                    assertTrue(releaseBlock.await(5, TimeUnit.SECONDS));
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
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

        // Second submit should fail because queue is full (maxSize=1)
        CompletableFuture<Response> second = scheduler.submit(context(52));
        Response response = second.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
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
    void processQueue_bsIter_exhaustion_uses_conservative_bound() throws Exception {
        // With slo_budget batcher (default), two 100-token requests each have
        // budget ≈ 350ms (slo=500, margin=50, pred≈100). Both fit within the
        // incremental budget and are dispatched together in a single batch.
        // flexlbBatchSearchIter is NOT used by slo_budget; flexlbBatchScanAhead
        // (default 64) determines how many candidates are scanned per iteration.
        config.setCostSloMs(500L);
        config.setCostSloRiskMarginMs(50L);
        config.setFlexlbBatchMaxCapacity(100000);
        config.setFlexlbBatchFillThreshold(0.5);
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
        config.setFlexlbBatchFillThreshold(1.0);

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
        verify(grpcClient, never()).batchEnqueue(anyString(), anyInt(), any(), anyLong());
    }

    // ==================== cancel / onRequestsFinished → Decode endpoint release ====================

    @Test
    void cancel_releases_decode_endpoint_resource() {
        // Register a DecodeEndpoint at the address the router returns for DECODE
        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        decodeStatus.setGrpcPort(9081);
        DecodeEndpoint decodeEp = scheduler.endpointRegistry.ensureDecodeEndpoint(
                "10.0.0.2:8081", decodeStatus);

        // Simulate strategy having reserved resources on the decode endpoint
        decodeEp.calibrate(null, null, 10000);
        decodeEp.reserve(17L, 500);
        assertEquals(1, decodeEp.getInflightCount());

        // Submit a request — router returns decode at 10.0.0.2:8081
        CompletableFuture<Response> future = scheduler.submit(context(17));

        // Cancel before batch is dispatched
        scheduler.cancel(17L);

        // Decode endpoint resource should be released
        assertEquals(0, decodeEp.getInflightCount(),
                "Cancel should propagate to DecodeEndpoint.release()");
        assertTrue(future.isDone());
        assertFalse(future.getNow(null).isSuccess());
    }


    @Test
    void cancel_with_decode_endpoint_not_registered_is_noop() throws Exception {
        // No DecodeEndpoint registered at 10.0.0.2:8081 — cancel should not throw
        CompletableFuture<Response> future = scheduler.submit(context(20));

        scheduler.cancel(20L);

        assertTrue(future.isDone());
        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        // No exception thrown — passes
    }

    // ==================== BatchIdGenerator Snowflake uniqueness ====================

    @Test
    void batchIdGeneratorProducesUniqueIds() {
        BatchIdGenerator gen = new BatchIdGenerator("10.0.0.1", 7001);
        Set<Long> ids = new HashSet<>();
        // 4000 is safely below the 12-bit sequence limit (4096 per ms per master),
        // matching the Python request_id design guarantee.
        for (int i = 0; i < 4000; i++) {
            long id = gen.nextBatchId();
            assertTrue(id > 0, "batch_id must be positive (not -1 default)");
            ids.add(id);
        }
        assertEquals(4000, ids.size(), "All batch IDs must be unique within a single ms window");
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

    @Test
    void batchIdGeneratorSurvivesRestart() {
        // A new generator instance (simulating restart) produces IDs
        // with higher timestamps, never colliding with old ones
        BatchIdGenerator gen1 = new BatchIdGenerator("10.0.0.1", 7001);
        long id1 = gen1.nextBatchId();

        // Simulate restart — new instance, same IP:port
        BatchIdGenerator gen2 = new BatchIdGenerator("10.0.0.1", 7001);
        long id2 = gen2.nextBatchId();

        // Timestamp portion should be >= id1's timestamp (time only moves forward)
        long ts1 = id1 >>> 24;
        long ts2 = id2 >>> 24;
        assertTrue(ts2 >= ts1, "Restarted generator must not produce IDs with older timestamps");
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

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .setGroupTimeout(com.google.protobuf.Int32Value.of(77))
                        .build())
                .build();
        return input.toByteArray();
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
