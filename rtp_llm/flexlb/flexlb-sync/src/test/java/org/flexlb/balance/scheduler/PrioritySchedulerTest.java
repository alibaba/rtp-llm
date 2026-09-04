package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.ShortestTTFTStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
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
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrioritySchedulerTest {

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private PriorityScheduler scheduler;
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
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10_000);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.tombstoned()));

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
        scheduler = new PriorityScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, null, null, cancelChannel,
                new PriorityScheduler.EngineFencePolicy(2, 100, 100, 2));

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
    void queueModeMatrixUsesOneSchedulingTimeoutCodeAcrossDispatchers() {
        long expiredAtMs = System.currentTimeMillis() - 1;

        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                submitExpired(90_001L, expiredAtMs).getCode());

        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                submitExpired(90_002L, expiredAtMs).getCode());

        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                submitExpired(90_003L, expiredAtMs).getCode());

        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                submitExpired(90_004L, expiredAtMs).getCode());
    }

    @Test
    void canonicalDiagnosticsReadWorkerQueueAndSchedulerLifecycle() throws Exception {
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);

        CompletableFuture<Response> pending = scheduler.submit(context(90_010L));
        awaitCondition(() -> scheduler.getQueuedRequestCount() == 1);

        assertFalse(pending.isDone());
        assertEquals(1, scheduler.getQueuedRequestCount());
        List<RequestLifecycleSnapshot> snapshot = scheduler.snapshotActiveRequests();
        assertEquals(1, snapshot.size());
        assertEquals(90_010L, snapshot.getFirst().requestId());
        assertEquals(RequestLifecycleState.QUEUED, snapshot.getFirst().state());
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
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(20);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(5L);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);

        for (long requestId = 9_000; requestId < 9_004; requestId++) {
            decode.reserve(requestId, 128, 136, 30);
        }
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            decode.reserve(requestId, 128, 136, 50);
            decode.markQueuedPhase(requestId);
            return successRoute(requestId);
        });

        List<Long> requestIds = java.util.stream.LongStream.range(1_000, 1_020)
                .boxed().toList();
        List<CompletableFuture<Response>> futures = requestIds.stream()
                .map(requestId -> scheduler.submit(context(requestId)))
                .toList();

        awaitCondition(() -> sentBatches.size() == 1
                && prefill.getBatcher().pendingDeliveryCount() == 0
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
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(19);
        awaitCondition(() -> sentBatches.size() == 2
                && prefill.getBatcher().pendingDeliveryCount() == 0
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
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint realDecode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        DecodeEndpoint throwingDecode = org.mockito.Mockito.spy(realDecode);
        long requestId = 2_100;
        realDecode.reserve(requestId, 128, 136, 50);
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
        awaitCondition(() -> prefill.getBatcher().pendingDeliveryCount() == 0
                && prefill.getBatcher().queueSize() == 0);
        assertEquals(0, scheduler.getInflightSize());
        assertFalse(realDecode.reservedView().containsKey(requestId));
        assertTrue(sentBatches.isEmpty());
    }

    @Test
    void decodeDispatchLimit_fullDpDoesNotBlockAnotherDpInSameBatch() throws Exception {
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(60_000);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(5L);
        PrefillEndpoint prefill = replacePrefillEndpoint();
        DecodeEndpoint full = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        DecodeEndpoint available = ensureDecodeEndpoint("10.0.0.3", 8081, 8082);
        for (long requestId = 9_100; requestId < 9_105; requestId++) {
            full.reserve(requestId, 128, 136, 30);
        }
        for (long requestId = 9_200; requestId < 9_204; requestId++) {
            available.reserve(requestId, 128, 136, 30);
        }
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            long requestId = ctx.getRequestId();
            DecodeEndpoint target = requestId == 2_001 ? full : available;
            target.reserve(requestId, 128, 136, 50);
            target.markQueuedPhase(requestId);
            return successRouteWithDecode(requestId,
                    requestId == 2_001 ? "10.0.0.2" : "10.0.0.3");
        });

        CompletableFuture<Response> blocked = scheduler.submit(context(2_001));
        CompletableFuture<Response> allowed = scheduler.submit(context(2_002));

        assertTrue(allowed.get(2, TimeUnit.SECONDS).isSuccess());
        awaitCondition(() -> prefill.getBatcher().pendingDeliveryCount() == 0
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
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
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
    void queueSchedulerReservationSpillsConcurrentEstablishedSession() throws Exception {
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(64);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10_000);
        RoutingConfig.SessionAffinityConfig affinity = new RoutingConfig.SessionAffinityConfig();
        affinity.setTtlMs(1_800_000L);
        affinity.setMaxExtraTtftMs(6_000L);
        config.getRouter().getRoles().getPrefill().setSessionAffinity(affinity);
        useFixedCandidatePool(config, 1);

        PrefillEndpoint placement = ensureSessionPrefill("10.0.0.1");
        ensureSessionPrefill("10.0.0.3");
        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.findMatchingEngines(any(), any(), any()))
                .thenReturn(new HashMap<>());
        ResourceMeasureFactory resourceMeasureFactory = mock(ResourceMeasureFactory.class);
        PrefillResourceMeasure resourceMeasure = mock(PrefillResourceMeasure.class);
        when(resourceMeasureFactory.getMeasure(any())).thenReturn(resourceMeasure);
        when(resourceMeasure.isResourceAvailable(any(PrefillEndpoint.class))).thenReturn(true);
        SessionPlacementStore placementStore = new SessionPlacementStore();
        placementStore.record("test-model", "session-queue", "10.0.0.1:8080");
        EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);
        ShortestTTFTStrategy strategy = new ShortestTTFTStrategy(
                new EngineWorkerStatus(endpointRegistry),
                cacheAwareService,
                resourceMeasureFactory,
                healthReporter,
                placementStore);
        long baselinePressureBatchId = 49_999L;
        placement.commitBatch(
                baselinePressureBatchId,
                5_000L,
                List.of(routeDecisionItem(baselinePressureBatchId, placement)));
        BalanceContext baselineContext = context(49_998L);
        baselineContext.setConfig(config);
        assertEquals("10.0.0.3",
                strategy.select(baselineContext, RoleType.PREFILL, null).getServerIp());
        clearInvocations(healthReporter);
        List<String> selected = new CopyOnWriteArrayList<>();
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext routed = invocation.getArgument(0);
            ServerStatus prefill = strategy.select(routed, RoleType.PREFILL, null);
            selected.add(prefill.getServerIp());
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(prefill));
            return response;
        });

        List<CompletableFuture<Response>> pending = new CopyOnWriteArrayList<>();
        ExecutorService submitters = Executors.newFixedThreadPool(6);
        CountDownLatch ready = new CountDownLatch(6);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> submissions = new ArrayList<>();
        try {
            pending.add(scheduler.submit(establishedSessionContext(50_000L)));
            awaitCondition(() -> placement.getBatcher().queueSize() == 1);
            assertEquals("10.0.0.1", selected.getFirst());
            verify(healthReporter).reportSessionAffinityDecision(
                    RoleType.PREFILL, "SESSION_AFFINITY");

            pending.add(scheduler.submit(establishedSessionContext(50_001L)));
            assertEquals("10.0.0.3", selected.get(1));
            verify(healthReporter).reportSessionAffinityDecision(
                    RoleType.PREFILL, "OVER_CAP");

            int concurrentStart = selected.size();
            for (int i = 0; i < 6; i++) {
                long requestId = 50_002L + i;
                submissions.add(submitters.submit(() -> {
                    ready.countDown();
                    assertTrue(start.await(2, TimeUnit.SECONDS));
                    pending.add(scheduler.submit(establishedSessionContext(requestId)));
                    return null;
                }));
            }
            assertTrue(ready.await(2, TimeUnit.SECONDS));
            start.countDown();
            for (Future<?> submission : submissions) {
                submission.get(2, TimeUnit.SECONDS);
            }

            assertEquals(8, selected.size());
            assertTrue(selected.subList(concurrentStart, selected.size()).contains("10.0.0.3"),
                    "scheduler-owned queue pressure must spill the hot session");
        } finally {
            placement.releaseBatch(baselinePressureBatchId);
            pending.forEach(future -> future.cancel(true));
            submitters.shutdownNow();
            assertTrue(submitters.awaitTermination(2, TimeUnit.SECONDS));
        }
    }

    @Test
    void routeDecisionMode_honorsRequestCap_withoutBatchEnqueue_andReleasesOnTerminal()
            throws Exception {
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);

        WorkerStatus prefillStatus = workerStatus("10.0.0.9", 8090, 8091);
        PrefillEndpoint prefill = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, "10.0.0.9:8090", prefillStatus);
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.8", 8180, 8181);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            long requestId = ((BalanceContext) invocation.getArgument(0)).getRequestId();
            decode.reserve(requestId, 128, 136, 50);
            decode.markQueuedPhase(requestId);
            return successRoute(
                    requestId,
                    server(RoleType.PREFILL, "10.0.0.9", 8090, 8091, requestId),
                    server(RoleType.DECODE, "10.0.0.8", 8180, 8181, requestId));
        });

        CompletableFuture<Response> first = scheduler.submit(routeDecisionContext(4_001));
        Response firstDecision = first.get(2, TimeUnit.SECONDS);
        assertTrue(firstDecision.isSuccess());
        assertFalse(firstDecision.isEnqueuedByMaster());
        assertEquals(0, prefill.getInflightBatchCount());
        assertEquals(1, prefill.getInflightRouteRequestCount());
        assertEquals(1, prefill.getInflightRequestCount());
        assertEquals(DeliveryClaimKind.ROUTE_DECISION,
                scheduler.getRequestState(4_001, 0).deliveryClaimKind());

        CompletableFuture<Response> second = scheduler.submit(routeDecisionContext(4_002));
        awaitCondition(() -> prefill.getBatcher().queueSize() == 1);
        assertFalse(second.isDone(), "the per-worker request cap must hold the next decision");
        assertEquals(1, prefill.getInflightRouteRequestCount());
        assertTrue(sentBatches.isEmpty());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());

        WorkerStatusResponse prefillFinished = finishedStatus(RoleType.PREFILL, 4_001, -1, 0);
        prefill.onWorkerStatusUpdate(prefillStatus, prefillFinished);
        scheduler.onWorkerStatusUpdate(prefillFinished);

        Response secondDecision = second.get(2, TimeUnit.SECONDS);
        assertTrue(secondDecision.isSuccess());
        assertFalse(secondDecision.isEnqueuedByMaster());
        assertEquals(0, prefill.getInflightBatchCount());
        assertEquals(1, prefill.getInflightRouteRequestCount());
        assertTrue(sentBatches.isEmpty());

        scheduler.onWorkerStatusUpdate(finishedStatus(RoleType.DECODE, 4_001, -1, 0));
        scheduler.onWorkerStatusUpdate(finishedStatus(RoleType.DECODE, 4_002, -1, 0));
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, prefill.getInflightRouteRequestCount());
        assertEquals(0, prefill.getInflightRequestCount());
        assertFalse(decode.reservedView().containsKey(4_001L));
        assertFalse(decode.reservedView().containsKey(4_002L));
    }

    @Test
    void routePublicationFenceRejectsPreemptionUntilAck_andUsesRequestIdForTerminal()
            throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt()))
                .thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(prefill.getIp()).thenReturn("10.0.0.1");

        CountDownLatch secondPredictionEntered = new CountDownLatch(1);
        CountDownLatch releaseSecondPrediction = new CountDownLatch(1);
        AtomicInteger predictions = new AtomicInteger();
        when(predictor.estimateMs(anyLong(), anyLong())).thenAnswer(invocation -> {
            if (predictions.incrementAndGet() == 2) {
                secondPredictionEntered.countDown();
                assertTrue(releaseSecondPrediction.await(2, TimeUnit.SECONDS));
            }
            return 1L;
        });

        BatchItem first = routeDecisionItem(4_101L, prefill);
        BatchItem second = routeDecisionItem(4_102L, prefill);
        assertTrue(scheduler.registerInflight(first));
        assertTrue(scheduler.registerInflight(second));

        CompletableFuture<Void> publication = CompletableFuture.runAsync(() ->
                scheduler.onDecisionGroupReady(List.of(first, second),
                        new DecisionGroupMetadata("route_fence_test", 0)));
        assertTrue(secondPredictionEntered.await(1, TimeUnit.SECONDS));
        assertEquals(RequestLifecycleState.DISPATCHING,
                scheduler.getRequestState(first.requestId(), 0).state());
        assertFalse(scheduler.claimForPreemption(
                        first.requestId(), 91L, "must not preempt publication"),
                "a DISPATCHING route is not yet visible to the frontend");

        releaseSecondPrediction.countDown();
        publication.get(1, TimeUnit.SECONDS);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(first.requestId(), 0).state());
        assertTrue(scheduler.claimForPreemption(
                        first.requestId(), 92L, "acknowledged route is eligible"),
                "the normal priority policy resumes after route publication");
        assertTrue(scheduler.releasePreemptionClaim(first.requestId(), 92L));

        // Frontends may assign a positive synthetic batch id when they send
        // the delivered request. Route accounting is request-scoped, so the
        // worker terminal must not be rejected as a batch-generation mismatch.
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.PREFILL, first.requestId(), 77_777L, 9_001L));
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(first.requestId(), 0).state());
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, second.requestId(), 88_888L, 0));
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void postDeliveryFenceRetainsLedgersForNonTerminalCancelOutcomes() throws Exception {
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        Map<Long, EngineCancelChannel.CancelOutcome> firstOutcomes = Map.of(
                4_201L, EngineCancelChannel.CancelOutcome.notFound(),
                4_202L, EngineCancelChannel.CancelOutcome.failed(),
                4_203L, EngineCancelChannel.CancelOutcome.accepted());
        Map<Long, AtomicInteger> calls = new java.util.concurrent.ConcurrentHashMap<>();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(invocation -> {
            long requestId = invocation.getArgument(1);
            int call = calls.computeIfAbsent(requestId, ignored -> new AtomicInteger())
                    .incrementAndGet();
            if (requestId == 4_204L && call == 1) {
                throw new IllegalStateException("synchronous transport failure");
            }
            return CompletableFuture.completedFuture(call == 1
                    ? firstOutcomes.get(requestId)
                    : EngineCancelChannel.CancelOutcome.tombstoned());
        });

        List<BatchItem> items = java.util.stream.LongStream.rangeClosed(4_201L, 4_204L)
                .mapToObj(requestId -> {
                    decode.reserve(requestId, 128, 136, 50);
                    decode.markQueuedPhase(requestId);
                    BatchItem item = routeDecisionItem(requestId, prefill, decode);
                    assertTrue(scheduler.registerInflight(item));
                    return item;
        }).toList();
        scheduler.onDecisionGroupReady(items, new DecisionGroupMetadata("post_delivery_fence", 0));
        awaitCondition(() -> items.stream().allMatch(item -> item.future().isDone()));

        for (BatchItem item : items) {
            assertEquals(
                    InflightRegistrar.PostDeliveryFenceResult.STARTED,
                    scheduler.fenceAfterDeliveryTimeout(item, "test_soft_timeout"));
            assertTrue(decode.reservedView().containsKey(item.requestId()),
                    "non-terminal Cancel outcome must retain Decode accounting");
            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(item.requestId(), 0).state());
        }

        awaitCondition(() -> items.stream().allMatch(item ->
                scheduler.getRequestState(item.requestId(), 0).state()
                        == RequestLifecycleState.TIMED_OUT));
        for (BatchItem item : items) {
            assertFalse(decode.reservedView().containsKey(item.requestId()));
            assertEquals(2, calls.get(item.requestId()).get(),
                    "one retained outcome must be followed by one authoritative TOMBSTONE");
        }
    }

    @Test
    void postDeliveryFenceUsesInternalCancelTimeoutWithoutConfigRead() throws Exception {
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        long requestId = 4_205L;
        decode.reserve(requestId, 128, 136, 50);
        decode.markQueuedPhase(requestId);
        BatchItem item = routeDecisionItem(requestId, prefill, decode);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("config_failure_fence", 0));

        clearInvocations(configService);
        when(cancelChannel.cancel(any(), eq(requestId), anyLong())).thenReturn(
                CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.tombstoned()));

        assertEquals(
                InflightRegistrar.PostDeliveryFenceResult.STARTED,
                scheduler.fenceAfterDeliveryTimeout(item, "test_config_failure"));

        awaitCondition(() -> scheduler.getRequestState(requestId, 0).state()
                == RequestLifecycleState.TIMED_OUT);
        assertFalse(decode.reservedView().containsKey(requestId));
        verify(cancelChannel, times(1)).cancel(any(), eq(requestId), eq(50L));
        verify(configService, never()).loadBalanceConfig();
    }

    @Test
    void blockingFrontendContinuationCannotHoldEntryLockOrBlockSiblingRoutePublication()
            throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt()))
                .thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(prefill.getIp()).thenReturn("10.0.0.1");
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(1L);

        BatchItem first = routeDecisionItem(4_211L, prefill);
        BatchItem second = routeDecisionItem(4_212L, prefill);
        assertTrue(scheduler.registerInflight(first));
        assertTrue(scheduler.registerInflight(second));
        CountDownLatch firstContinuationEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstContinuation = new CountDownLatch(1);
        first.future().thenRun(() -> {
            firstContinuationEntered.countDown();
            try {
                assertTrue(releaseFirstContinuation.await(2, TimeUnit.SECONDS));
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new AssertionError(interrupted);
            }
        });

        scheduler.onDecisionGroupReady(List.of(first, second),
                new DecisionGroupMetadata("unlocked_publication", 0));
        assertTrue(firstContinuationEntered.await(1, TimeUnit.SECONDS));
        assertTrue(second.future().get(1, TimeUnit.SECONDS).isSuccess(),
                "a blocked first continuation must not serialize its sibling publication");
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(first.requestId(), 0).state());
        assertTrue(scheduler.claimForPreemption(
                        first.requestId(), 9_211L, "reentrant lock probe"),
                "the completion continuation must not retain the request-entry monitor");
        assertTrue(scheduler.releasePreemptionClaim(first.requestId(), 9_211L));

        releaseFirstContinuation.countDown();
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, first.requestId(), 0, 0));
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, second.requestId(), 0, 0));
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void completionExecutorBoundsRealRoutePublications_andBackpressuresOnCaller()
            throws Exception {
        recreateScheduler(
                new PriorityScheduler.EngineFencePolicy(2, 10_000, 10_000, 2),
                new PriorityScheduler.CompletionExecutorPolicy(1, 1));

        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt())).thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(prefill.getIp()).thenReturn("10.0.0.1");
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(1L);

        BatchItem blocking = routeDecisionItem(4_221L, prefill);
        BatchItem queued = routeDecisionItem(4_222L, prefill);
        BatchItem callerRuns = routeDecisionItem(4_223L, prefill);
        BatchItem shutdownRace = routeDecisionItem(4_224L, prefill);
        assertTrue(scheduler.registerInflight(blocking));
        assertTrue(scheduler.registerInflight(queued));
        assertTrue(scheduler.registerInflight(callerRuns));
        assertTrue(scheduler.registerInflight(shutdownRace));

        CountDownLatch blockingContinuationEntered = new CountDownLatch(1);
        CountDownLatch releaseBlockingContinuation = new CountDownLatch(1);
        AtomicReference<Thread> completionWorker = new AtomicReference<>();
        AtomicReference<Thread> backpressureCaller = new AtomicReference<>();
        AtomicReference<Thread> shutdownCaller = new AtomicReference<>();
        blocking.future().thenRun(() -> {
            completionWorker.set(Thread.currentThread());
            blockingContinuationEntered.countDown();
            try {
                releaseBlockingContinuation.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        });
        callerRuns.future().thenRun(() -> backpressureCaller.set(Thread.currentThread()));
        shutdownRace.future().thenRun(() -> shutdownCaller.set(Thread.currentThread()));

        try {
            scheduler.onDecisionGroupReady(List.of(blocking),
                    new DecisionGroupMetadata("bounded_completion_worker", 0));
            assertTrue(blockingContinuationEntered.await(1, TimeUnit.SECONDS));
            scheduler.onDecisionGroupReady(List.of(queued),
                    new DecisionGroupMetadata("bounded_completion_queue", 0));
            awaitCondition(() -> scheduler.completionExecutorSnapshot().queueSize() == 1);

            Thread submittingThread = Thread.currentThread();
            scheduler.onDecisionGroupReady(List.of(callerRuns),
                    new DecisionGroupMetadata("bounded_completion_backpressure", 0));
            assertTrue(callerRuns.future().get(1, TimeUnit.SECONDS).isSuccess());
            assertSame(submittingThread, backpressureCaller.get(),
                    "a full completion queue must apply caller-runs backpressure");

            PriorityScheduler.CompletionExecutorSnapshot saturated =
                    scheduler.completionExecutorSnapshot();
            assertEquals(1, saturated.workerLimit());
            assertEquals(1, saturated.queueCapacity());
            assertEquals(1, saturated.largestPoolSize());
            assertTrue(saturated.queueSize() <= saturated.queueCapacity());
            assertTrue(completionWorker.get().isDaemon());
            assertFalse(completionWorker.get().isVirtual());
            assertTrue(completionWorker.get().getName()
                    .startsWith("priority-scheduler-completion-"));

            scheduler.shutdown();
            assertTrue(scheduler.completionExecutorSnapshot().shutdown());
            scheduler.onDeliveryFailure(shutdownRace,
                    new IllegalStateException("shutdown terminal race"));
            assertFalse(shutdownRace.future().get(1, TimeUnit.SECONDS).isSuccess());
            assertSame(submittingThread, shutdownCaller.get(),
                    "an in-flight terminal publication racing shutdown must not be dropped");
        } finally {
            releaseBlockingContinuation.countDown();
        }

        assertTrue(queued.future().get(1, TimeUnit.SECONDS).isSuccess());
        awaitCondition(() -> scheduler.completionExecutorSnapshot().completedTaskCount() == 2);
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, blocking.requestId(), 0, 0));
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, queued.requestId(), 0, 0));
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, callerRuns.requestId(), 0, 0));
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, shutdownRace.requestId(), 0, 0));
        assertEquals(0, scheduler.getInflightSize());
        assertTrue(scheduler.awaitCompletionExecutorTermination(1, TimeUnit.SECONDS));
        assertTrue(scheduler.completionExecutorSnapshot().shutdown());
    }

    @Test
    void synchronousBatchRejectionDrainsCallbacksOutsideDeliveryFence() throws Exception {
        scheduler.shutdown();
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher rejectingDispatcher = (items, prefill, batchId, predictedMs,
                                                reason, callback) -> {
            for (BatchItem item : items) {
                callback.onFailure(item, new RejectedExecutionException("dispatch saturated"));
            }
        };
        scheduler = new PriorityScheduler(
                configService, router, endpointRegistry, rejectingDispatcher, reporter,
                null, null, cancelChannel,
                new PriorityScheduler.EngineFencePolicy(2, 10_000, 10_000, 2),
                RouteDecisionDelivery.INSTANCE,
                new PriorityScheduler.CompletionExecutorPolicy(1, 1));

        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt())).thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(prefill.getIp()).thenReturn("10.0.0.1");
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(1L);
        when(predictor.predictBatchMs(any())).thenReturn(1.0);

        BatchItem blocking = routeDecisionItem(4_231L, prefill);
        BatchItem queued = routeDecisionItem(4_232L, prefill);
        assertTrue(scheduler.registerInflight(blocking));
        assertTrue(scheduler.registerInflight(queued));
        CountDownLatch completionWorkerBlocked = new CountDownLatch(1);
        CountDownLatch releaseCompletionWorker = new CountDownLatch(1);
        blocking.future().thenRun(() -> {
            completionWorkerBlocked.countDown();
            awaitLatch(releaseCompletionWorker);
        });
        scheduler.onDecisionGroupReady(List.of(blocking),
                new DecisionGroupMetadata("block_completion_worker", 0));
        assertTrue(completionWorkerBlocked.await(1, TimeUnit.SECONDS));
        scheduler.onDecisionGroupReady(List.of(queued),
                new DecisionGroupMetadata("fill_completion_queue", 0));
        assertEquals(1, scheduler.completionExecutorSnapshot().queueSize());

        long failedRequestId = 4_233L;
        Response route = successRoute(failedRequestId);
        BatchItem rejected = new BatchItem(
                context(failedRequestId), new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                prefill, null, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(rejected));
        CountDownLatch rejectionContinuationEntered = new CountDownLatch(1);
        CountDownLatch releaseRejectionContinuation = new CountDownLatch(1);
        rejected.future().thenRun(() -> {
            rejectionContinuationEntered.countDown();
            awaitLatch(releaseRejectionContinuation);
        });

        CompletableFuture<Void> rejectedDelivery = CompletableFuture.runAsync(() ->
                scheduler.onDecisionGroupReady(List.of(rejected),
                        new DecisionGroupMetadata("synchronous_rejection", 0)));
        try {
            assertTrue(rejectionContinuationEntered.await(1, TimeUnit.SECONDS));

            // WorkerStatus reduction uses the same global delivery fence. It
            // must remain available while caller-runs backpressure blocks the
            // synchronously rejected batch's response continuation.
            CompletableFuture<Void> statusReduction = CompletableFuture.runAsync(() ->
                    scheduler.onWorkerStatusUpdate(
                            finishedStatus(RoleType.DECODE, blocking.requestId(), 0, 0)));
            statusReduction.get(1, TimeUnit.SECONDS);
        } finally {
            releaseRejectionContinuation.countDown();
            releaseCompletionWorker.countDown();
        }
        rejectedDelivery.get(1, TimeUnit.SECONDS);
        assertFalse(rejected.future().get(1, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void shutdownClosesDeliveryGateBeforeEndpointsAndWaitsForAcceptedGroup()
            throws Exception {
        scheduler.shutdown();

        CountDownLatch deliveryEntered = new CountDownLatch(1);
        CountDownLatch releaseDelivery = new CountDownLatch(1);
        CountDownLatch endpointClosed = new CountDownLatch(1);
        CountDownLatch shutdownStarted = new CountDownLatch(1);
        AtomicInteger deliveryCalls = new AtomicInteger();
        DecisionDelivery<List<BatchItem>> blockingDelivery = (items, callback) -> {
            deliveryCalls.incrementAndGet();
            deliveryEntered.countDown();
            awaitLatch(releaseDelivery);
            items.forEach(callback::onDelivered);
        };

        endpointRegistry = mock(EndpointRegistry.class);
        doAnswer(invocation -> {
            endpointClosed.countDown();
            return null;
        }).when(endpointRegistry).close();
        scheduler = new PriorityScheduler(
                configService, router, endpointRegistry, mock(BatchDispatcher.class), reporter,
                null, null, cancelChannel,
                new PriorityScheduler.EngineFencePolicy(2, 10_000, 10_000, 2),
                blockingDelivery,
                new PriorityScheduler.CompletionExecutorPolicy(1, 4));

        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt())).thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(1L);

        BatchItem accepted = routeDecisionItem(4_234L, prefill);
        BatchItem rejected = routeDecisionItem(4_235L, prefill);
        assertTrue(scheduler.registerInflight(accepted));
        assertTrue(scheduler.registerInflight(rejected));

        CompletableFuture<Void> acceptedDelivery = CompletableFuture.runAsync(() ->
                scheduler.onDecisionGroupReady(List.of(accepted),
                        new DecisionGroupMetadata("accepted_before_shutdown", 0)));
        assertTrue(deliveryEntered.await(1, TimeUnit.SECONDS));
        assertEquals(1, scheduler.activeDeliveryPermitCount());

        CompletableFuture<Void> shutdown = CompletableFuture.runAsync(() -> {
            shutdownStarted.countDown();
            scheduler.shutdown();
        });
        assertTrue(shutdownStarted.await(1, TimeUnit.SECONDS));
        awaitDeliveryLifecycleClosed(scheduler);
        assertFalse(shutdown.isDone());
        assertFalse(endpointClosed.await(100, TimeUnit.MILLISECONDS),
                "endpoint close must wait for a delivery which crossed the gate");

        scheduler.onDecisionGroupReady(List.of(rejected),
                new DecisionGroupMetadata("rejected_after_shutdown", 0));
        Response rejection = rejected.future().get(1, TimeUnit.SECONDS);
        assertFalse(rejection.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                rejection.getCode());
        assertEquals(1, deliveryCalls.get(),
                "a staged group must not enter delivery after the gate closes");
        assertEquals(InflightRegistrar.PostDeliveryFenceResult.ALREADY_TERMINAL,
                scheduler.fenceAfterDeliveryTimeout(accepted,
                        "must not install a fence after the gate closes"));
        verify(cancelChannel, never()).cancel(any(), eq(accepted.requestId()), anyLong());

        releaseDelivery.countDown();
        acceptedDelivery.get(1, TimeUnit.SECONDS);
        shutdown.get(1, TimeUnit.SECONDS);
        assertTrue(endpointClosed.await(1, TimeUnit.SECONDS));
        assertTrue(accepted.future().get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, scheduler.activeDeliveryPermitCount());
    }

    @Test
    void shutdownCancelsEngineFenceRetry_andRejectsNewWork() throws Exception {
        recreateScheduler(
                new PriorityScheduler.EngineFencePolicy(2, 10_000, 10_000, 2),
                new PriorityScheduler.CompletionExecutorPolicy(1, 4));

        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(prefill.getPredictor()).thenReturn(predictor);
        when(prefill.tryCommitRequest(anyLong(), anyLong(), anyInt())).thenReturn(true);
        when(prefill.releaseRequest(anyLong())).thenReturn(true);
        when(prefill.getIp()).thenReturn("10.0.0.1");
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(1L);

        AtomicInteger cancelCalls = new AtomicInteger();
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(invocation -> {
            cancelCalls.incrementAndGet();
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.failed());
        });
        BatchItem item = routeDecisionItem(4_224L, prefill);
        assertTrue(scheduler.registerInflight(item));
        scheduler.onDecisionGroupReady(List.of(item),
                new DecisionGroupMetadata("shutdown_retry", 0));
        assertTrue(item.future().get(1, TimeUnit.SECONDS).isSuccess());

        assertEquals(InflightRegistrar.PostDeliveryFenceResult.STARTED,
                scheduler.fenceAfterDeliveryTimeout(item, "shutdown_retry"));
        assertEquals(1, cancelCalls.get());
        assertEquals(1, scheduler.engineFenceRetryQueueSize());

        scheduler.shutdown();
        assertEquals(0, scheduler.engineFenceRetryQueueSize());
        assertFalse(scheduler.registerInflight(routeDecisionItem(4_225L, prefill)));
        Response rejected = scheduler.submit(context(4_226L)).get(1, TimeUnit.SECONDS);
        assertFalse(rejected.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), rejected.getCode());
        verify(router, never()).route(any(BalanceContext.class));
        assertEquals(1, cancelCalls.get(),
                "shutdown must cancel, rather than merely invalidate, the delayed retry");
        scheduler.onWorkerStatusUpdate(
                finishedStatus(RoleType.DECODE, item.requestId(), 0, 0));
        assertEquals(0, scheduler.getInflightSize());
    }

    @Test
    void completedAdmissionFuturesAreRemovedFromSharedDeadlineTimer() throws Exception {
        assertTrue(scheduler.removesCanceledRequestExpirations());
        int baseline = scheduler.requestExpirationQueueSize();
        int requestCount = 128;
        long now = System.currentTimeMillis();
        List<CompletableFuture<Response>> futures = new java.util.ArrayList<>(requestCount);
        for (int i = 0; i < requestCount; i++) {
            BalanceContext context = context(4_250L + i);
            context.setSchedulingMetadata(SchedulingMetadata.explicit(
                    50, now + TimeUnit.MINUTES.toMillis(1)));
            CompletableFuture<Response> future = new CompletableFuture<>();
            futures.add(future);
            scheduler.attachRequestExpiration(context, future);
        }
        awaitCondition(() -> scheduler.requestExpirationQueueSize()
                == baseline + requestCount);

        Response success = new Response();
        success.setSuccess(true);
        futures.forEach(future -> future.complete(success));
        awaitCondition(() -> scheduler.requestExpirationQueueSize() == baseline);
        assertEquals(baseline, scheduler.requestExpirationQueueSize(),
                "completed requests must not be retained until their original deadline");
    }

    @Test
    void quarantinedFencesStopDelayedRetries_andCleanupProbesRoundRobin() throws Exception {
        config.queueScheduler().getLifecycle()
                .setStaleInflightTimeoutMs(TimeUnit.MINUTES.toMillis(10));
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        DecodeEndpoint decode = ensureDecodeEndpoint("10.0.0.2", 8081, 8082);
        List<Long> requestIds = java.util.stream.LongStream.rangeClosed(4_301L, 4_305L)
                .boxed().toList();
        Map<Long, AtomicInteger> calls = new java.util.concurrent.ConcurrentHashMap<>();
        java.util.concurrent.atomic.AtomicBoolean settleProbes =
                new java.util.concurrent.atomic.AtomicBoolean(false);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(invocation -> {
            long requestId = invocation.getArgument(1);
            int call = calls.computeIfAbsent(requestId, ignored -> new AtomicInteger())
                    .incrementAndGet();
            return CompletableFuture.completedFuture(settleProbes.get() && call >= 4
                    ? EngineCancelChannel.CancelOutcome.tombstoned()
                    : EngineCancelChannel.CancelOutcome.failed());
        });

        List<BatchItem> items = requestIds.stream().map(requestId -> {
            decode.reserve(requestId, 128, 136, 50);
            decode.markQueuedPhase(requestId);
            BatchItem item = routeDecisionItem(requestId, prefill, decode);
            assertTrue(scheduler.registerInflight(item));
            return item;
        }).toList();
        scheduler.onDecisionGroupReady(items, new DecisionGroupMetadata("quarantine_fairness", 0));
        for (BatchItem item : items) {
            assertEquals(InflightRegistrar.PostDeliveryFenceResult.STARTED,
                    scheduler.fenceAfterDeliveryTimeout(item, "quarantine_test"));
        }

        awaitCondition(() -> requestIds.stream().allMatch(requestId ->
                calls.getOrDefault(requestId, new AtomicInteger()).get() == 2));
        Thread.sleep(250);
        assertTrue(requestIds.stream().allMatch(requestId -> calls.get(requestId).get() == 2),
                "quarantine must own no permanent delayed retry task");
        assertEquals(items.size(), scheduler.getInflightSize());
        assertTrue(requestIds.stream().allMatch(decode.reservedView()::containsKey));
        assertEquals(items.size(), prefill.getInflightRouteRequestCount());

        // Endpoint TTL is intentionally much shorter than the quarantined
        // fence age. The scheduler-owned resource handle must keep both
        // request ledgers charged until an authoritative Engine proof.
        assertEquals(0, prefill.evictExpiredRequests(1));
        assertEquals(0, decode.evictExpiredRequests(1));
        assertEquals(items.size(), prefill.getInflightRouteRequestCount());
        assertTrue(requestIds.stream().allMatch(decode.reservedView()::containsKey));

        // Probe cap is two in this test. Three cleanup sweeps provide six
        // slots for five live fences; FIFO rotation must visit every fence at
        // least once instead of repeatedly selecting the same CHM prefix.
        scheduler.cleanupInflight();
        scheduler.cleanupInflight();
        scheduler.cleanupInflight();
        assertTrue(requestIds.stream().allMatch(requestId -> calls.get(requestId).get() >= 3),
                "every quarantined generation must receive a fair low-frequency probe");
        assertEquals(items.size(), scheduler.getInflightSize());
        assertTrue(requestIds.stream().allMatch(decode.reservedView()::containsKey),
                "non-terminal probes must retain every ledger");

        settleProbes.set(true);
        scheduler.cleanupInflight();
        scheduler.cleanupInflight();
        scheduler.cleanupInflight();
        assertEquals(0, scheduler.getInflightSize());
        assertTrue(requestIds.stream().noneMatch(decode.reservedView()::containsKey));
        assertEquals(0, prefill.getInflightRouteRequestCount(),
                "TOMBSTONED must release protected Prefill accounting exactly once");
        assertTrue(requestIds.stream().allMatch(requestId ->
                scheduler.getRequestState(requestId, 0).state()
                        == RequestLifecycleState.TIMED_OUT));

        assertTrue(scheduler.quarantinedProbeQueueSize() > 0,
                "terminal probes are generation-checked and may leave lazy stale refs");
        scheduler.cleanupInflight();
        assertEquals(0, scheduler.quarantinedProbeQueueSize(),
                "the next empty quarantine sweep must discard every stale generation ref");

        long shutdownRequestId = 4_306L;
        decode.reserve(shutdownRequestId, 128, 136, 50);
        decode.markQueuedPhase(shutdownRequestId);
        BatchItem shutdownItem = routeDecisionItem(shutdownRequestId, prefill, decode);
        assertTrue(scheduler.registerInflight(shutdownItem));
        scheduler.onDecisionGroupReady(List.of(shutdownItem),
                new DecisionGroupMetadata("quarantine_shutdown", 0));
        assertEquals(InflightRegistrar.PostDeliveryFenceResult.STARTED,
                scheduler.fenceAfterDeliveryTimeout(shutdownItem, "quarantine_shutdown"));
        awaitCondition(() -> calls.getOrDefault(
                shutdownRequestId, new AtomicInteger()).get() == 2);
        awaitCondition(() -> scheduler.quarantinedProbeQueueSize() > 0);
        scheduler.shutdown();
        assertEquals(0, scheduler.quarantinedProbeQueueSize(),
                "shutdown must not retain exact request/fence generations");
    }

    @Test
    void submit_rejects_when_global_inflight_limit_reached() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(1);

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
    void globalOutstandingCapacityIsAtomicAndReusableAfterConcurrentTerminals()
            throws Exception {
        int limit = 4;
        int attempts = 64;
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(limit);

        AtomicInteger routed = new AtomicInteger();
        AtomicInteger activeRoutes = new AtomicInteger();
        AtomicInteger peakRoutes = new AtomicInteger();
        CountDownLatch acceptedRoutesEntered = new CountDownLatch(limit);
        CountDownLatch releaseAcceptedRoutes = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            routed.incrementAndGet();
            int active = activeRoutes.incrementAndGet();
            peakRoutes.accumulateAndGet(active, Math::max);
            acceptedRoutesEntered.countDown();
            try {
                assertTrue(releaseAcceptedRoutes.await(5, TimeUnit.SECONDS));
            } finally {
                activeRoutes.decrementAndGet();
            }
            return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        });

        ExecutorService submitters = Executors.newFixedThreadPool(attempts);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch rejectedReturned = new CountDownLatch(attempts - limit);
        CountDownLatch allReturned = new CountDownLatch(attempts);
        List<CompletableFuture<Response>> futures =
                Collections.synchronizedList(new ArrayList<>());
        try {
            for (int i = 0; i < attempts; i++) {
                long requestId = 10_000L + i;
                submitters.execute(() -> {
                    try {
                        assertTrue(start.await(5, TimeUnit.SECONDS));
                        CompletableFuture<Response> future = scheduler.submit(context(requestId));
                        futures.add(future);
                        if (future.isDone()) {
                            rejectedReturned.countDown();
                        }
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                    } finally {
                        allReturned.countDown();
                    }
                });
            }
            start.countDown();
            assertTrue(acceptedRoutesEntered.await(5, TimeUnit.SECONDS));
            assertTrue(rejectedReturned.await(5, TimeUnit.SECONDS),
                    "every request beyond the hard bound must reject while N routes are blocked");

            assertEquals(limit, scheduler.outstandingRequestCount());
            assertEquals(limit, routed.get());
            assertEquals(limit, peakRoutes.get(),
                    "the capacity CAS must prevent concurrent check-then-act penetration");

            releaseAcceptedRoutes.countDown();
            assertTrue(allReturned.await(5, TimeUnit.SECONDS));
            assertEquals(attempts, futures.size());
            int queueFull = 0;
            int routedFailure = 0;
            for (CompletableFuture<Response> future : futures) {
                Response response = future.get(2, TimeUnit.SECONDS);
                if (response.getCode() == StrategyErrorType.QUEUE_FULL.getErrorCode()) {
                    queueFull++;
                } else if (response.getCode()
                        == StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()) {
                    routedFailure++;
                }
            }
            assertEquals(attempts - limit, queueFull);
            assertEquals(limit, routedFailure);
            awaitCondition(() -> scheduler.outstandingRequestCount() == 0);

            Response afterRelease = scheduler.submit(context(20_000L))
                    .get(2, TimeUnit.SECONDS);
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                    afterRelease.getCode());
            assertEquals(limit + 1, routed.get(),
                    "terminal requests must return their permits for later submissions");
            assertEquals(0, scheduler.outstandingRequestCount());
        } finally {
            releaseAcceptedRoutes.countDown();
            submitters.shutdownNow();
        }
    }

    @Test
    void priorityAdmissionsBeforeInflightRegistrationConsumeAtomicCapacity()
            throws Exception {
        int limit = 3;
        int attempts = 48;
        SchedulingTestConfig.usePriorityQueue(config);
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(limit);

        PriorityAdmissionScheduler priorityAdmission = mock(PriorityAdmissionScheduler.class);
        List<CompletableFuture<Response>> admitted = new CopyOnWriteArrayList<>();
        doAnswer(invocation -> {
            admitted.add(invocation.getArgument(1));
            return null;
        }).when(priorityAdmission).schedule(
                any(BalanceContext.class), any(CompletableFuture.class), any(InflightRegistrar.class));

        scheduler.shutdown();
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
                mock(BatchDispatcher.class), reporter, priorityAdmission, null, cancelChannel);

        ExecutorService submitters = Executors.newFixedThreadPool(attempts);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch allReturned = new CountDownLatch(attempts);
        List<CompletableFuture<Response>> futures =
                Collections.synchronizedList(new ArrayList<>());
        try {
            for (int i = 0; i < attempts; i++) {
                long requestId = 50_000L + i;
                submitters.execute(() -> {
                    try {
                        assertTrue(start.await(5, TimeUnit.SECONDS));
                        futures.add(scheduler.submit(context(requestId)));
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                    } finally {
                        allReturned.countDown();
                    }
                });
            }
            start.countDown();
            assertTrue(allReturned.await(5, TimeUnit.SECONDS));

            assertEquals(limit, admitted.size());
            assertEquals(0, scheduler.getInflightSize(),
                    "pre-registration admissions expose the old inflight.size race");
            assertEquals(limit, scheduler.outstandingRequestCount());
            assertEquals(attempts - limit, futures.stream()
                    .filter(CompletableFuture::isDone)
                    .map(CompletableFuture::join)
                    .filter(response -> response.getCode()
                            == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode())
                    .count());

            admitted.forEach(future -> future.complete(
                    Response.error(StrategyErrorType.NO_AVAILABLE_WORKER)));
            awaitCondition(() -> scheduler.outstandingRequestCount() == 0);

            CompletableFuture<Response> afterRelease = scheduler.submit(context(60_000L));
            assertEquals(limit + 1, admitted.size());
            admitted.getLast().complete(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
            assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                    afterRelease.get(1, TimeUnit.SECONDS).getCode());
            assertEquals(0, scheduler.outstandingRequestCount());
        } finally {
            submitters.shutdownNow();
        }
    }

    @Test
    void shutdownClosesOutstandingAdmissionAndReleasesPendingPermitExactlyOnce()
            throws Exception {
        SchedulingTestConfig.usePriorityQueue(config);
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(1);
        PriorityAdmissionScheduler priorityAdmission = mock(PriorityAdmissionScheduler.class);

        scheduler.shutdown();
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
                mock(BatchDispatcher.class), reporter, priorityAdmission, null, cancelChannel);

        CompletableFuture<Response> pending = scheduler.submit(context(70_000L));
        assertFalse(pending.isDone());
        assertEquals(1, scheduler.outstandingRequestCount());

        scheduler.shutdown();
        Response response = pending.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                response.getCode());
        assertEquals(0, scheduler.outstandingRequestCount());

        scheduler.shutdown();
        pending.cancel(false);
        assertEquals(0, scheduler.outstandingRequestCount(),
                "repeated shutdown/future terminal must not underflow capacity");
    }

    @Test
    void duplicateAndRepeatedTerminalReducersReleaseOnePermitExactlyOnce()
            throws Exception {
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(1);
        CountDownLatch routeEntered = new CountDownLatch(1);
        CountDownLatch releaseRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            routeEntered.countDown();
            assertTrue(releaseRoute.await(5, TimeUnit.SECONDS));
            BalanceContext routedContext = invocation.getArgument(0);
            return successRoute(routedContext.getRequestId());
        });

        long requestId = 30_000L;
        CompletableFuture<CompletableFuture<Response>> submitted =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context(requestId)));
        assertTrue(routeEntered.await(2, TimeUnit.SECONDS));
        assertEquals(1, scheduler.outstandingRequestCount());

        Response duplicate = scheduler.submit(context(requestId)).get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), duplicate.getCode());
        assertEquals(1, scheduler.outstandingRequestCount(),
                "duplicate rejection must not release the live generation's permit");

        releaseRoute.countDown();
        CompletableFuture<Response> original = submitted.get(2, TimeUnit.SECONDS);
        assertEquals(1, scheduler.getInflightSize());
        scheduler.cancelRequest(requestId, 0, CancelReason.CLIENT_CANCELLED);
        Response cancelled = original.get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), cancelled.getCode());
        awaitCondition(() -> scheduler.outstandingRequestCount() == 0);

        scheduler.cancelRequest(requestId, 0, CancelReason.CLIENT_CANCELLED);
        scheduler.onRequestExpired(requestId, original);
        Response duplicateAfterTerminal = scheduler.submit(context(requestId))
                .get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicateAfterTerminal.getCode());
        assertEquals(0, scheduler.outstandingRequestCount(),
                "duplicate and repeated terminal reducers must not double-release");

        when(router.route(any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
        Response nextGeneration = scheduler.submit(context(requestId + 1))
                .get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(),
                nextGeneration.getCode());
        assertEquals(0, scheduler.outstandingRequestCount());
    }

    @Test
    void capacityRejectionPreservesFifoNonBatchAndPriorityStatusContracts()
            throws Exception {
        config.queueScheduler().getCapacity().setMaxOutstandingRequestsGlobal(1);
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);

        CountDownLatch fifoRouteEntered = new CountDownLatch(1);
        CountDownLatch releaseFifoRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            fifoRouteEntered.countDown();
            assertTrue(releaseFifoRoute.await(5, TimeUnit.SECONDS));
            return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        });
        CompletableFuture<CompletableFuture<Response>> fifoSubmit =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context(40_000L)));
        assertTrue(fifoRouteEntered.await(2, TimeUnit.SECONDS));
        Response fifoRejected = scheduler.submit(context(40_001L))
                .get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), fifoRejected.getCode(),
                "FIFO + NON_BATCH preserves the established queue-timeout contract");
        releaseFifoRoute.countDown();
        fifoSubmit.get(2, TimeUnit.SECONDS).get(2, TimeUnit.SECONDS);
        awaitCondition(() -> scheduler.outstandingRequestCount() == 0);

        SchedulingTestConfig.usePriorityQueue(config);
        CountDownLatch priorityRouteEntered = new CountDownLatch(1);
        CountDownLatch releasePriorityRoute = new CountDownLatch(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            priorityRouteEntered.countDown();
            assertTrue(releasePriorityRoute.await(5, TimeUnit.SECONDS));
            return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        });
        CompletableFuture<CompletableFuture<Response>> prioritySubmit =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context(40_002L)));
        assertTrue(priorityRouteEntered.await(2, TimeUnit.SECONDS));
        Response priorityRejected = scheduler.submit(context(40_003L))
                .get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                priorityRejected.getCode());
        assertEquals(org.flexlb.dao.loadbalance.AdmissionRejectReason.RESOURCE_EXHAUSTED,
                priorityRejected.getAdmissionRejectReason());
        releasePriorityRoute.countDown();
        prioritySubmit.get(2, TimeUnit.SECONDS).get(2, TimeUnit.SECONDS);
        awaitCondition(() -> scheduler.outstandingRequestCount() == 0);
    }

    @Test
    void batcher_rejects_when_queue_full() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config)
                .setMaxWaitingRequestsPerPrefillWorker(1);

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
        assertTrue(response.getErrorMessage().contains(
                "Worker scheduling queue rejected request"));
    }

    @Test
    void staged_delivery_failure_uses_delivery_reducer_and_neutral_message() throws Exception {
        BatchItem item = offerFailureItem(63);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onDeliveryFailure(item,
                new IllegalStateException("callback lost delivery ownership"));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), response.getCode());
        assertTrue(response.getErrorMessage().contains("Decision delivery failed"));
        assertFalse(response.getErrorMessage().contains("queue rejected"));
        assertEquals(0, scheduler.getInflightSize());
    }

    private BatchItem offerFailureItem(long requestId) {
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId), new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill("10.0.0.1:8080"), null,
                System.currentTimeMillis());
    }

    @Test
    void mismatched_generate_input_request_id_fails_before_batch_enqueue() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);

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

    private Response submitExpired(long requestId, long expiredAtMs) {
        BalanceContext context = context(requestId);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, expiredAtMs));
        return scheduler.submit(context).join();
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

    private BalanceContext establishedSessionContext(long requestId) {
        BalanceContext context = contextWithSeqLen(requestId, 1_000L);
        context.setConfig(config);
        context.getRequest().setSessionSchemaVersion(1);
        context.getRequest().setInferenceSessionId("session-queue");
        context.getRequest().setInferenceSessionState(Request.SessionState.ESTABLISHED);
        return context;
    }

    private static BalanceContext contextWithLegacyBatchFields(long requestId) {
        BalanceContext ctx = context(requestId);
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId, true));
        return ctx;
    }

    // ==================== P0-1: onTimeout terminal handling (PR-D) ====================

    @Test
    void onTimeout_beforeDeliveryClaim_settlesPriorityAdmissionAsResourceExhausted()
            throws Exception {
        // A timeout is locally terminal only before a batch delivery assigns a
        // batch id. The engine provably cannot have observed this item yet.
        BatchItem item = offerFailureItem(301);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onTimeout(item, new TimeoutException("test EnqueueBatch deadline"));

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());

        // Idempotent: a second timeout is also a no-op
        scheduler.onTimeout(item, new TimeoutException("second"));
        Response stillUnchanged = item.future().get(1, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), stillUnchanged.getCode());
    }

    @Test
    void requestExpirationBeforeBatchCommitAbortsWithoutEngineFence()
            throws Exception {
        SchedulingTestConfig.usePriorityQueue(config);

        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = new BatchItem(context(303), new CompletableFuture<>(), successRoute(303),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 303),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 303),
                endpoint, null, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));

        CountDownLatch deliveryClaimed = new CountDownLatch(1);
        CountDownLatch allowCommit = new CountDownLatch(1);
        PrefillTimePredictor predictor = endpoint.getPredictor();
        PrefillEndpoint blockingEndpoint = org.mockito.Mockito.spy(endpoint);
        when(blockingEndpoint.getPredictor()).thenAnswer(inv -> {
            deliveryClaimed.countDown();
            assertTrue(allowCommit.await(5, TimeUnit.SECONDS));
            return predictor;
        });
        BatchItem blockingItem = new BatchItem(item.ctx(), item.future(), item.routeResponse(),
                item.prefill(), item.decode(), blockingEndpoint, null, item.enqueuedAtMs());
        scheduler.unregisterInflight(item);
        assertTrue(scheduler.registerInflight(blockingItem));

        CompletableFuture<Void> flush = CompletableFuture.runAsync(() ->
                scheduler.onDecisionGroupReady(List.of(blockingItem), new DecisionGroupMetadata("race", 0)));
        assertTrue(deliveryClaimed.await(2, TimeUnit.SECONDS));

        scheduler.onRequestExpired(blockingItem.requestId(), blockingItem.future());
        Response timedOut = blockingItem.future().get(2, TimeUnit.SECONDS);
        assertFalse(timedOut.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), timedOut.getCode());

        allowCommit.countDown();
        flush.get(2, TimeUnit.SECONDS);
        assertEquals(0, blockingEndpoint.getInflightBatchCount(),
                "delivery revalidation must prevent a commit after local timeout");

        SchedulingTestConfig.useBatchDispatcher(config)
                .setMaxInflightBatchesPerPrefillWorker(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        assertTrue(scheduler.submit(context(304)).get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, sentBatches.size(),
                "after authoritative settlement, maxInflight=1 admits the next batch");

        verify(cancelChannel, never()).cancel(any(), eq(303L), anyLong());
    }

    @Test
    void uncertainBatchDelivery_legacyNotFoundRetainsFutureAndBothLedgers() throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.notFound()));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = new BatchItem(context(305), new CompletableFuture<>(), successRoute(305),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 305),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 305),
                endpoint, null, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sentBatches.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sentBatches.getLast().getBatchId();

        Thread.sleep(50);

        assertFalse(item.future().isDone());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, endpoint.getInflightBatchCount());
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(0);
        scheduler.cleanupInflight();
        assertEquals(1, scheduler.getInflightSize(), "TTL cannot break uncertain ownership");
    }

    @Test
    void uncertainBatchDelivery_acceptedCancelWaitsForTypedPrefillFinished() throws Exception {
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineCancelChannel.CancelOutcome.accepted()));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.failedFuture(new TimeoutException("lost ack"));
                });
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = new BatchItem(context(306), new CompletableFuture<>(), successRoute(306),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 306),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 306),
                endpoint, null, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));
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
        scheduler.onWorkerStatusUpdate(status);

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, endpoint.getInflightBatchCount());
    }

    @Test
    void uncertainBatchDeliveryAfterConfirmedFutureDoesNotStartCancelReconciliation()
            throws Exception {
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BatchItem item = new BatchItem(context(307), new CompletableFuture<>(), successRoute(307),
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, 307),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, 307),
                endpoint, null, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));
        Response response = item.future().get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        long batchId = sentBatches.getLast().getBatchId();
        RequestLifecycleSnapshot acknowledged = scheduler.getRequestState(307L, batchId);
        assertEquals(RequestLifecycleState.ACKNOWLEDGED, acknowledged.state());

        scheduler.onUncertain(item, new RuntimeException("late callback"));

        verify(cancelChannel, never()).cancel(any(), eq(307L), anyLong());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(307L, batchId).state());
    }

    @Test
    void uncertainBatchDelivery_acceptedRetriesUntilTombstoned() throws Exception {
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
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));

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
    void uncertainBatchDelivery_synchronousCancelThrowIsRetried() throws Exception {
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
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));

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
    void uncertainBatchDelivery_onlyMatchingTypedCanceled8429IsTerminal() throws Exception {
        AtomicInteger cancelCalls = new AtomicInteger();
        CountDownLatch reconciliationStarted = new CountDownLatch(1);
        when(cancelChannel.cancel(any(), anyLong(), anyLong())).thenAnswer(inv -> {
            cancelCalls.incrementAndGet();
            reconciliationStarted.countDown();
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
        scheduler.onDecisionGroupReady(List.of(item), new DecisionGroupMetadata("test", 0));
        long deadline = System.currentTimeMillis() + 1_000;
        while (sentBatches.isEmpty() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        long batchId = sentBatches.getLast().getBatchId();
        assertTrue(reconciliationStarted.await(1, TimeUnit.SECONDS),
                "the uncertain EnqueueBatch callback must install its resource fence first");

        scheduler.onWorkerStatusUpdate(prefillFinished(
                309, batchId, 0, PriorityPreemptionProgress.NONE));
        scheduler.onWorkerStatusUpdate(prefillFinished(
                309, batchId, 500, PriorityPreemptionProgress.NONE));
        scheduler.onWorkerStatusUpdate(prefillFinished(
                309, batchId + 1, 8429, PriorityPreemptionProgress.CANCELED));
        assertFalse(item.future().isDone());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, endpoint.getInflightBatchCount());

        scheduler.onWorkerStatusUpdate(prefillFinished(
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

    // ==================== P0-3: close/delivery race (PR-D) ====================

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

        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
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

    private static Response successRoute(long requestId,
                                         ServerStatus prefill,
                                         ServerStatus decode) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(prefill, decode));
        return response;
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

    private static BalanceContext routeDecisionContext(long requestId) {
        BalanceContext context = context(requestId);
        SchedulingTestConfig.useNonBatchDispatcher(context.getConfig());
        return context;
    }

    private static BatchItem routeDecisionItem(long requestId,
                                               PrefillEndpoint prefillEndpoint) {
        return routeDecisionItem(requestId, prefillEndpoint, null);
    }

    private static BatchItem routeDecisionItem(long requestId,
                                               PrefillEndpoint prefillEndpoint,
                                               DecodeEndpoint decodeEndpoint) {
        BalanceContext context = routeDecisionContext(requestId);
        Response route = successRoute(requestId);
        return new BatchItem(context, new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                prefillEndpoint, decodeEndpoint, System.currentTimeMillis());
    }

    private static WorkerStatus workerStatus(String ip, int httpPort, int grpcPort) {
        WorkerStatus status = new WorkerStatus();
        status.setIp(ip);
        status.setPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setAlive(true);
        return status;
    }

    private PrefillEndpoint ensureSessionPrefill(String ip) {
        WorkerStatus status = workerStatus(ip, 8080, 8081);
        status.setRole(RoleType.PREFILL);
        status.setRunningTaskList(new HashMap<>());
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10_000);
        cacheStatus.setBlockSize(256);
        status.setCacheStatus(cacheStatus);
        return (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ip + ":8080", status);
    }

    private static void useFixedCandidatePool(FlexlbConfig config, int workers) {
        RoutingConfig.FixedCandidatePoolConfig pool = new RoutingConfig.FixedCandidatePoolConfig();
        pool.setWorkers(workers);
        RoutingConfig.LeastRecentlyUsedInPoolConfig choice =
                new RoutingConfig.LeastRecentlyUsedInPoolConfig();
        choice.setPool(pool);
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        selector.setCandidateChoice(choice);
    }

    private static WorkerStatusResponse finishedStatus(RoleType role,
                                                       long requestId,
                                                       long batchId,
                                                       long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setErrorCode(errorCode);
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setRole(role);
        status.setFinishedTaskInfo(Map.of(Long.toString(requestId), task));
        return status;
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
        ws.setIp(ip);
        ws.setPort(httpPort);
        ws.setGrpcPort(grpcPort);
        ws.setAlive(true);
        return (DecodeEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.DECODE, ip + ":" + httpPort, ws);
    }

    private void recreateScheduler(
            PriorityScheduler.EngineFencePolicy engineFencePolicy,
            PriorityScheduler.CompletionExecutorPolicy completionExecutorPolicy) {
        scheduler.shutdown();
        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new PriorityScheduler(
                configService,
                router,
                endpointRegistry,
                mock(BatchDispatcher.class),
                reporter,
                null,
                null,
                cancelChannel,
                engineFencePolicy,
                RouteDecisionDelivery.INSTANCE,
                completionExecutorPolicy);
    }

    private static void awaitCondition(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (!condition.getAsBoolean() && System.currentTimeMillis() < deadline) {
            Thread.sleep(1);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true before timeout");
    }

    private static void awaitDeliveryLifecycleClosed(PriorityScheduler target) {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (!target.isDeliveryLifecycleClosed() && System.nanoTime() < deadlineNanos) {
            Thread.onSpinWait();
        }
        assertTrue(target.isDeliveryLifecycleClosed(),
                "delivery lifecycle did not close before timeout");
    }

    private static void awaitLatch(CountDownLatch latch) {
        try {
            assertTrue(latch.await(2, TimeUnit.SECONDS), "latch did not open before timeout");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }
}
