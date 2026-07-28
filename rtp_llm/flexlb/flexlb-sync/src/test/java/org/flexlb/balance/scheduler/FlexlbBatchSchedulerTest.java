package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
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
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
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
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();
    private final List<String> sentEndpoints = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
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
                endpointRegistry, dispatcher, reporter, null);

        // Create endpoint and batcher for the worker that successRoute() returns
        String ipPort = "10.0.0.1:8080";
        WorkerStatus ws = new WorkerStatus();
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setRole(RoleType.PREFILL);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);
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
        assertEquals(0, inputs.get(0).getGroupSize());
        assertEquals(0, inputs.get(0).getGroupId().getValue());
        assertEquals(0, inputs.get(1).getGroupId().getValue());
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
    void predictorFailureAfterClaimFailsRequestsAndReleasesAllInflight() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        DecodeEndpoint decode = registerDecodeEndpointWithReservation();
        PrefillEndpoint prefill = installPrefillSpy();
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.predictBatchMs(any())).thenThrow(new IllegalStateException("predict failed"));
        doReturn(predictor).when(prefill).getPredictor();

        Response response = scheduler.submit(context(801L)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(801L, 0).state());
        assertAllInflightReleased(prefill, decode);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void commitFailureAfterClaimFailsRequestsAndReleasesAllInflight() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        DecodeEndpoint decode = registerDecodeEndpointWithReservation();
        PrefillEndpoint prefill = installPrefillSpy();
        doThrow(new IllegalStateException("commit failed"))
                .when(prefill).commitBatch(anyLong(), anyLong(), any());

        Response response = scheduler.submit(context(802L)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(802L, 0).state());
        assertAllInflightReleased(prefill, decode);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void reporterFailureIsBestEffortAndDoesNotBlockDispatch() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(reporter).reportBatchWaitTimeMs(
                        anyString(), anyString(), anyLong());
        doThrow(new IllegalStateException("ack metrics unavailable"))
                .when(reporter).reportDispatchAckTimeMs(
                        anyString(), anyString(), anyLong());

        CompletableFuture<Response> future = scheduler.submit(context(803L));

        assertTrue(future.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size());
    }

    @Test
    void repackFailureCannotBlockTimeoutTerminalization() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        DecodeEndpoint decode = registerDecodeEndpointWithReservation();
        PrefillEndpoint prefill = installPrefillSpy();
        doThrow(new IllegalStateException("repack failed"))
                .when(prefill).repackBatch(anyLong(), any());
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> timeout = new CompletableFuture<>();
        timeout.completeExceptionally(
                io.grpc.Status.DEADLINE_EXCEEDED.withDescription("enqueue timeout")
                        .asRuntimeException());
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(timeout);

        Response response = scheduler.submit(context(805L)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(805L, 0).state());
        assertAllInflightReleased(prefill, decode);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void shutdownTerminalizesRequestStillBlockedInRouting() throws Exception {
        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        decodeStatus.setGrpcPort(8082);
        DecodeEndpoint decode = (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, "10.0.0.2:8081", decodeStatus);
        CountDownLatch routeStarted = new CountDownLatch(1);
        CountDownLatch releaseRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext routed = invocation.getArgument(0);
            decode.reserve(routed.getRequestId(), routed.getRequest().getSeqLen(), routed);
            routeStarted.countDown();
            assertTrue(releaseRoute.await(8, TimeUnit.SECONDS));
            return successRoute(routed.getRequestId());
        });
        ExecutorService submitter = Executors.newSingleThreadExecutor();

        try {
            Future<CompletableFuture<Response>> submitted =
                    submitter.submit(() -> scheduler.submit(context(806L)));
            assertTrue(routeStarted.await(2, TimeUnit.SECONDS));

            scheduler.shutdown();

            assertEquals(0, scheduler.getInflightSize());
            assertEquals(0, decode.getInflightCount());
            assertEquals(RequestLifecycleState.FAILED,
                    scheduler.getRequestState(806L, 0).state());
            releaseRoute.countDown();
            assertFalse(submitted.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS).isSuccess());
        } finally {
            releaseRoute.countDown();
            submitter.shutdownNow();
        }
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
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        assertEquals(1, prefill.getInflightBatchCount(),
                "partial ACK must keep the accepted part of the committed batch");
        assertEquals(1, prefill.realPendingCount(),
                "rejected request must be removed from Prefill batch accounting");
        assertEquals(1, scheduler.getInflightSize());

        TaskInfo finished = new TaskInfo();
        finished.setRequestId(81L);
        finished.setBatchId(sentBatches.getFirst().getBatchId());
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of("81", finished));
        prefill.onWorkerStatusUpdate(prefill.getStatus(), status);
        scheduler.onWorkerStatusUpdate(status);
        assertEquals(0, prefill.getInflightBatchCount());
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void batch_enqueue_missing_success_fails_missing_request() throws Exception {
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
        assertTrue(secondResp.getErrorMessage().contains("EnqueueBatch missing ack for request 84"));
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
        scheduler.onWorkerStatusUpdate(status);

        assertFalse(scheduleFuture.isDone());
        ackFuture.complete(ackFor(sentBatches.getFirst()));

        Response response = scheduleFuture.get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
        assertEquals(RequestLifecycleState.COMPLETED,
                scheduler.getRequestState(85L, batchId).state());
    }

    @Test
    void worker_completion_before_enqueue_failure_still_completes_schedule_future() throws Exception {
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

        CompletableFuture<Response> scheduleFuture = scheduler.submit(context(87));
        assertTrue(enqueueStarted.await(2, TimeUnit.SECONDS));
        long batchId = sentBatches.getFirst().getBatchId();
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(87L);
        finished.setBatchId(batchId);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of("87", finished));
        scheduler.onWorkerStatusUpdate(status);

        ackFuture.completeExceptionally(new IllegalStateException("enqueue transport failed"));

        Response response = scheduleFuture.get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(response.isEnqueuedByMaster());
    }

    @Test
    void workerStatusWithoutPositiveBatchIdCannotFinishQueuedReplacement() {
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchFixedWaitMs(60_000L);
        CompletableFuture<Response> scheduleFuture = scheduler.submit(context(88));
        TaskInfo staleFinished = new TaskInfo();
        staleFinished.setRequestId(88L);
        staleFinished.setBatchId(0L);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of("88", staleFinished));

        scheduler.onWorkerStatusUpdate(status);

        assertFalse(scheduleFuture.isDone());
        assertEquals(1, scheduler.getInflightSize());
    }

    @Test
    void worker_error_before_enqueue_ack_completes_schedule_future() throws Exception {
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

        CompletableFuture<Response> scheduleFuture = scheduler.submit(context(86));
        assertTrue(enqueueStarted.await(2, TimeUnit.SECONDS));
        long batchId = sentBatches.getFirst().getBatchId();

        TaskInfo failed = new TaskInfo();
        failed.setRequestId(86L);
        failed.setBatchId(batchId);
        failed.setErrorCode(1);
        failed.setErrorMessage("worker rejected request");
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(RoleType.DECODE);
        status.setFinishedTaskInfo(Map.of("86", failed));
        scheduler.onWorkerStatusUpdate(status);

        Response response = scheduleFuture.get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(86L, batchId).state());

        ackFuture.complete(ackFor(sentBatches.getFirst()));
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                scheduleFuture.getNow(null).getCode());
    }

    @Test
    void duplicateSubmitIsRejectedBeforeSecondRouterSideEffect() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        CountDownLatch firstRouteEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstRoute = new CountDownLatch(1);
        AtomicInteger routeCalls = new AtomicInteger();
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext context = invocation.getArgument(0);
            if (routeCalls.incrementAndGet() == 1) {
                firstRouteEntered.countDown();
                assertTrue(releaseFirstRoute.await(3, TimeUnit.SECONDS));
            }
            return successRoute(context.getRequestId());
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<CompletableFuture<Response>> firstSubmit =
                    executor.submit(() -> scheduler.submit(context(13L)));
            assertTrue(firstRouteEntered.await(2, TimeUnit.SECONDS));

            Response duplicate = scheduler.submit(context(13L))
                    .get(1, TimeUnit.SECONDS);
            assertFalse(duplicate.isSuccess());
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                    duplicate.getCode());
            assertEquals(1, routeCalls.get(),
                    "duplicate admission must not enter Router");

            releaseFirstRoute.countDown();
            assertTrue(firstSubmit.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS).isSuccess());
        } finally {
            releaseFirstRoute.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }

    @Test
    void routingTtlCleanupReleasesAdmissionBeforeRouteReturns() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchMaxInflight(1);
        config.setFlexlbInflightTtlMs(20L);

        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        DecodeEndpoint decode = (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, "10.0.0.2:8081", decodeStatus);
        CountDownLatch routeReserved = new CountDownLatch(1);
        CountDownLatch releaseRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext context = invocation.getArgument(0);
            if (context.getRequestId() == 16L) {
                decode.reserve(context.getRequestId(), 128, context);
                routeReserved.countDown();
                assertTrue(releaseRoute.await(3, TimeUnit.SECONDS));
            }
            return successRoute(context.getRequestId());
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<CompletableFuture<Response>> submitted =
                    executor.submit(() -> scheduler.submit(context(16L)));
            assertTrue(routeReserved.await(2, TimeUnit.SECONDS));
            Thread.sleep(50L);

            scheduler.cleanupInflight();

            assertEquals(0, scheduler.getInflightSize(),
                    "routing TTL must release the global admission slot");
            RequestLifecycleSnapshot timedOut = scheduler.getRequestState(16L, 0);
            assertEquals(RequestLifecycleState.TIMED_OUT, timedOut.state());
            assertTrue(scheduler.submit(context(18L))
                    .get(2, TimeUnit.SECONDS).isSuccess());

            releaseRoute.countDown();
            Response response = submitted.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS);
            assertFalse(response.isSuccess());
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                    response.getCode());
            assertEquals(0, decode.getInflightCount());
        } finally {
            releaseRoute.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }

    @Test
    void lateRouteCannotReleaseReplacementLeaseWithSameRequestId() throws Exception {
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbInflightTtlMs(20L);

        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        DecodeEndpoint decode = (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, "10.0.0.2:8081", decodeStatus);
        BalanceContext staleContext = context(19L);
        BalanceContext replacementContext = context(19L);
        CountDownLatch staleRouteReserved = new CountDownLatch(1);
        CountDownLatch releaseStaleRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext routed = invocation.getArgument(0);
            decode.reserve(routed.getRequestId(), 128, routed);
            if (routed == staleContext) {
                staleRouteReserved.countDown();
                assertTrue(releaseStaleRoute.await(3, TimeUnit.SECONDS));
            }
            return successRoute(routed.getRequestId());
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<CompletableFuture<Response>> staleSubmit =
                    executor.submit(() -> scheduler.submit(staleContext));
            assertTrue(staleRouteReserved.await(2, TimeUnit.SECONDS));
            Thread.sleep(50L);
            scheduler.cleanupInflight();
            assertEquals(1, decode.evictExpiredRequests(20L));

            Thread.sleep(50L);
            scheduler.cleanupInflight();
            scheduler.submit(replacementContext);
            assertEquals(1, decode.getInflightCount());

            releaseStaleRoute.countDown();
            Response staleResponse = staleSubmit.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS);
            assertFalse(staleResponse.isSuccess());
            assertEquals(1, decode.getInflightCount(),
                    "late route must not acquire and release the replacement lease");

        } finally {
            releaseStaleRoute.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
    }

    @Test
    void lateRouteCannotOverwriteReplacementCompletionTombstone() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbInflightTtlMs(20L);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> replacementAck =
                new CompletableFuture<>();
        CountDownLatch replacementEnqueueStarted = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(invocation -> {
                    EngineRpcService.EnqueueBatchRequestPB request = invocation.getArgument(2);
                    sentBatches.add(request);
                    replacementEnqueueStarted.countDown();
                    return replacementAck;
                });

        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        DecodeEndpoint decode = (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, "10.0.0.2:8081", decodeStatus);
        BalanceContext staleContext = context(89L);
        BalanceContext replacementContext = context(89L);
        CountDownLatch staleRouteReserved = new CountDownLatch(1);
        CountDownLatch releaseStaleRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext routed = invocation.getArgument(0);
            decode.reserve(routed.getRequestId(), 128, routed);
            if (routed == staleContext) {
                staleRouteReserved.countDown();
                assertTrue(releaseStaleRoute.await(3, TimeUnit.SECONDS));
            }
            return successRoute(routed.getRequestId());
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<CompletableFuture<Response>> staleSubmit =
                    executor.submit(() -> scheduler.submit(staleContext));
            assertTrue(staleRouteReserved.await(2, TimeUnit.SECONDS));
            Thread.sleep(50L);
            scheduler.cleanupInflight();
            assertEquals(1, decode.evictExpiredRequests(20L));
            Thread.sleep(50L);
            scheduler.cleanupInflight();

            CompletableFuture<Response> replacementFuture = scheduler.submit(replacementContext);
            assertTrue(replacementEnqueueStarted.await(2, TimeUnit.SECONDS));
            long replacementBatchId = sentBatches.getFirst().getBatchId();
            TaskInfo completed = new TaskInfo();
            completed.setRequestId(89L);
            completed.setBatchId(replacementBatchId);
            WorkerStatusResponse status = new WorkerStatusResponse();
            status.setRole(RoleType.DECODE);
            status.setFinishedTaskInfo(Map.of("89", completed));
            scheduler.onWorkerStatusUpdate(status);
            assertFalse(replacementFuture.isDone());

            releaseStaleRoute.countDown();
            assertFalse(staleSubmit.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS).isSuccess());
            replacementAck.complete(ackFor(sentBatches.getFirst()));

            assertTrue(replacementFuture.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(RequestLifecycleState.COMPLETED,
                    scheduler.getRequestState(89L, replacementBatchId).state());
        } finally {
            releaseStaleRoute.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
        }
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

    private static BatcherContext batcherContext(WorkerBatcher batcher) throws Exception {
        java.lang.reflect.Field field = WorkerBatcher.class.getDeclaredField("ctx");
        field.setAccessible(true);
        return (BatcherContext) field.get(batcher);
    }

    private PrefillEndpoint installPrefillSpy() {
        String key = "10.0.0.1:8080";
        PrefillEndpoint endpoint = spy(endpointRegistry.getPrefill(key));
        endpointRegistry.getPrefillEndpoints().put(key, endpoint);
        return endpoint;
    }

    private DecodeEndpoint registerDecodeEndpointWithReservation() {
        WorkerStatus decodeStatus = new WorkerStatus();
        decodeStatus.setIp("10.0.0.2");
        decodeStatus.setPort(8081);
        decodeStatus.setGrpcPort(8082);
        DecodeEndpoint decode = (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, "10.0.0.2:8081", decodeStatus);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext routed = invocation.getArgument(0);
            decode.reserve(routed.getRequestId(), routed.getRequest().getSeqLen(), routed);
            return successRoute(routed.getRequestId());
        });
        return decode;
    }

    private void assertAllInflightReleased(PrefillEndpoint prefill, DecodeEndpoint decode) {
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefill.getBatcher().queueSize());
        assertEquals(0, prefill.getInflightBatchCount());
        assertEquals(0, decode.getInflightCount());
    }

    private void awaitRequestState(long requestId,
                                   long batchId,
                                   RequestLifecycleState expected) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        RequestLifecycleSnapshot current;
        do {
            current = scheduler.getRequestState(requestId, batchId);
            if (current != null && current.state() == expected) {
                return;
            }
            Thread.sleep(5L);
        } while (System.nanoTime() < deadline);
        assertEquals(expected, current == null ? null : current.state());
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
}
