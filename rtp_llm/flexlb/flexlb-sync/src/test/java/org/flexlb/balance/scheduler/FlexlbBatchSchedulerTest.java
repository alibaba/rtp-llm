package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointId;
import org.flexlb.balance.endpoint.EndpointOperationLease;
import org.flexlb.balance.endpoint.EndpointRetireCause;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
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
                endpointRegistry, dispatcher, reporter, null, null);

        // Create endpoint and batcher for the worker that successRoute() returns
        String ipPort = "10.0.0.1:8080";
        WorkerStatus ws = new WorkerStatus();
        ws.setIp("10.0.0.1");
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        ws.setRole(RoleType.PREFILL);
        ws.tryMarkReady();
        ServerStatus prefill = new ServerStatus();
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setRole(RoleType.PREFILL);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);

        WorkerStatus decodeStatus = readyStatus(RoleType.DECODE, "10.0.0.2", 8081, 8082);
        endpointRegistry.ensureEndpoint(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus);
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
        assertEquals(0, inputs.get(1).getGroupSize());
        assertFalse(inputs.get(0).hasGroupId());
        assertFalse(inputs.get(1).hasGroupId());
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

    @Test
    void retiring_prefill_generation_settles_queued_request_and_allows_new_generation()
            throws Exception {
        String ipPort = "10.0.0.1:8080";
        PrefillEndpoint oldEndpoint = endpointRegistry.getPrefill(ipPort);
        WorkerStatus oldStatus = oldEndpoint.getStatus();

        CompletableFuture<Response> queued = scheduler.submit(context(53));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (oldEndpoint.getBatcher().queueSize() == 0 && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(1, oldEndpoint.getBatcher().queueSize());
        assertEquals(1, scheduler.getInflightSize());

        assertTrue(endpointRegistry.retire(RoleType.PREFILL, ipPort, oldStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));

        Response retired = queued.get(1, TimeUnit.SECONDS);
        assertFalse(retired.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), retired.getCode());
        assertEquals(0, oldEndpoint.getBatcher().queueSize());
        assertEquals(0, oldEndpoint.getInflightBatchCount());
        assertEquals(0, scheduler.getInflightSize());

        WorkerStatus replacement = readyStatus(RoleType.PREFILL, "10.0.0.1", 8080, 8081);
        WorkerEndpoint newEndpoint = endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ipPort, replacement);
        assertNotNull(newEndpoint);
        assertTrue(newEndpoint.getEndpointId().generation()
                > oldEndpoint.getEndpointId().generation());

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            Response route = successRoute(ctx.getRequestId());
            FlexlbBatchScheduler.findServer(route, RoleType.PREFILL)
                    .setEndpointGeneration(newEndpoint.getEndpointId().generation());
            return route;
        });

        CompletableFuture<Response> first = scheduler.submit(context(54));
        CompletableFuture<Response> second = scheduler.submit(context(55));
        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void stale_route_generation_is_never_rebound_to_replacement_endpoint() throws Exception {
        String ipPort = "10.0.0.1:8080";
        PrefillEndpoint oldEndpoint = endpointRegistry.getPrefill(ipPort);
        WorkerStatus oldStatus = oldEndpoint.getStatus();
        Response staleRoute = successRoute(59);
        FlexlbBatchScheduler.findServer(staleRoute, RoleType.PREFILL)
                .setEndpointGeneration(oldEndpoint.getEndpointId().generation());

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            assertTrue(endpointRegistry.retire(RoleType.PREFILL, ipPort, oldStatus,
                    EndpointRetireCause.HEALTH_CHECK_FAILED));
            WorkerEndpoint replacement = endpointRegistry.ensureEndpoint(
                    RoleType.PREFILL, ipPort,
                    readyStatus(RoleType.PREFILL, "10.0.0.1", 8080, 8081));
            assertNotNull(replacement);
            assertTrue(replacement.getEndpointId().generation()
                    > oldEndpoint.getEndpointId().generation());
            return staleRoute;
        });

        Response response = scheduler.submit(context(59)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), response.getCode());
        PrefillEndpoint replacement = endpointRegistry.getPrefill(ipPort);
        assertEquals(0, replacement.getBatcher().queueSize());
        assertTrue(sentBatches.isEmpty());
    }

    @Test
    void route_during_decode_retirement_barrier_is_not_admitted_without_its_generation()
            throws Exception {
        DecodeEndpoint decode = endpointRegistry.getDecode("10.0.0.2:8081");
        WorkerStatus decodeStatus = decode.getStatus();
        EndpointOperationLease lease = EndpointOperationLease.acquire(decode).orElseThrow();
        CompletableFuture<Boolean> retirement = CompletableFuture.supplyAsync(() ->
                endpointRegistry.retire(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus,
                        EndpointRetireCause.HEALTH_CHECK_FAILED));

        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (endpointRegistry.getDecode(decodeStatus.getIpPort()) != null
                && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertNull(endpointRegistry.getDecode(decodeStatus.getIpPort()));

        Response response = scheduler.submit(context(60)).get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        assertEquals(0, endpointRegistry.getPrefill("10.0.0.1:8080").getBatcher().queueSize());
        assertTrue(sentBatches.isEmpty());

        lease.close();
        assertTrue(retirement.get(1, TimeUnit.SECONDS));
    }

    @Test
    void legacy_handoff_is_atomic_with_decode_retirement() throws Exception {
        long requestId = 61L;
        DecodeEndpoint decode = endpointRegistry.getDecode("10.0.0.2:8081");
        WorkerStatus decodeStatus = decode.getStatus();
        decode.reserve(requestId, 10L, 20L);
        BlockingRouteSubmittedContext ctx = blockingContext(requestId);

        CompletableFuture<CompletableFuture<Response>> submission =
                CompletableFuture.supplyAsync(() -> scheduler.submit(ctx));
        assertTrue(ctx.handoffReached.await(1, TimeUnit.SECONDS));
        assertEquals(1, scheduler.getInflightSize());

        CompletableFuture<Boolean> retirement = CompletableFuture.supplyAsync(() ->
                endpointRegistry.retire(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus,
                        EndpointRetireCause.HEALTH_CHECK_FAILED));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (endpointRegistry.getDecode(decodeStatus.getIpPort()) != null
                && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertNull(endpointRegistry.getDecode(decodeStatus.getIpPort()));
        assertFalse(retirement.isDone(),
                "retirement must wait for the route-to-batcher generation lease");

        ctx.releaseHandoff.countDown();
        CompletableFuture<Response> scheduled = submission.get(1, TimeUnit.SECONDS);
        assertTrue(retirement.get(1, TimeUnit.SECONDS));

        Response retired = scheduled.get(1, TimeUnit.SECONDS);
        assertFalse(retired.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), retired.getCode());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, decode.getInflightCount());
        assertEquals(0,
                endpointRegistry.getPrefill("10.0.0.1:8080").getBatcher().queueSize());
        assertTrue(sentBatches.isEmpty());
    }

    @Test
    void retiring_decode_generation_removes_request_from_counterpart_prefill_queue()
            throws Exception {
        DecodeEndpoint decode = endpointRegistry.getDecode("10.0.0.2:8081");
        WorkerStatus decodeStatus = decode.getStatus();
        decode.reserve(58L, 10L, 20L);

        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        CompletableFuture<Response> queued = scheduler.submit(context(58));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (prefill.getBatcher().queueSize() == 0 && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(1, prefill.getBatcher().queueSize());
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, decode.getInflightCount());

        assertTrue(endpointRegistry.retire(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));

        Response retired = queued.get(1, TimeUnit.SECONDS);
        assertFalse(retired.isSuccess());
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(), retired.getCode());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, decode.getInflightCount());
        assertEquals(0, prefill.getBatcher().queueSize());
        assertEquals(0, prefill.getInflightBatchCount());
    }

    @Test
    void retiring_decode_generation_settles_dispatched_scheduler_and_prefill_state()
            throws Exception {
        DecodeEndpoint decode = endpointRegistry.getDecode("10.0.0.2:8081");
        WorkerStatus decodeStatus = decode.getStatus();
        decode.reserve(56L, 10L, 20L);
        decode.reserve(57L, 10L, 20L);

        CompletableFuture<Response> first = scheduler.submit(context(56));
        CompletableFuture<Response> second = scheduler.submit(context(57));
        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(2, scheduler.getInflightSize());
        assertEquals(2, decode.getInflightCount());
        PrefillEndpoint prefill = endpointRegistry.getPrefill("10.0.0.1:8080");
        assertEquals(1, prefill.getInflightBatchCount());

        assertTrue(endpointRegistry.retire(RoleType.DECODE, decodeStatus.getIpPort(), decodeStatus,
                EndpointRetireCause.HEALTH_CHECK_FAILED));

        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, decode.getInflightCount());
        assertEquals(0, decode.getTotalLoad());
        assertEquals(0, prefill.getInflightBatchCount());
    }

    @Test
    void retirement_settles_other_entries_when_one_entry_cleanup_fails_then_retries() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        String ipPort = "10.0.0.1:8080";
        PrefillEndpoint original = endpointRegistry.getPrefill(ipPort);
        FailOnceRepackPrefillEndpoint failing = new FailOnceRepackPrefillEndpoint(
                new EndpointId(RoleType.PREFILL, ipPort, 1L), original.getStatus(),
                config, scheduler, reporter);
        original.close();
        endpointRegistry.getPrefillEndpoints().put(ipPort, failing);

        Response accepted = scheduler.submit(context(59)).get(2, TimeUnit.SECONDS);
        assertTrue(accepted.isSuccess());
        assertEquals(1, scheduler.getInflightSize());

        assertThrows(IllegalStateException.class, () -> scheduler.retireEndpoint(
                failing, EndpointRetireCause.HEALTH_CHECK_FAILED));
        // No terminal/tombstone is published until every cleanup step has
        // succeeded, leaving this item available for the registry retry pass.
        assertEquals(1, scheduler.getInflightSize());

        failing.allowRepack();
        assertEquals(1, scheduler.retireEndpoint(failing, EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(RequestLifecycleState.FAILED, scheduler.getRequestState(59L, 0).state());
    }

    @Test
    void retirement_retries_decode_release_when_first_rollback_throws() throws Exception {
        config.setFlexlbBatchSizeMax(1);
        String decodeIpPort = "10.0.0.2:8081";
        WorkerStatus status = readyStatus(RoleType.DECODE, "10.0.0.2", 8081, 8082);
        FailOnceReleaseDecodeEndpoint failing = new FailOnceReleaseDecodeEndpoint(
                new EndpointId(RoleType.DECODE, decodeIpPort, 1L), status);
        endpointRegistry.getDecode(decodeIpPort).close();
        endpointRegistry.getDecodeEndpoints().put(decodeIpPort, failing);
        failing.reserve(63L, 10L, 20L);

        assertTrue(scheduler.submit(context(63)).get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(1, scheduler.getInflightSize());
        assertThrows(IllegalStateException.class, () -> scheduler.retireEndpoint(
                failing, EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertEquals(1, scheduler.getInflightSize());
        assertEquals(1, failing.getInflightCount());

        failing.allowRelease();
        assertEquals(1, scheduler.retireEndpoint(failing, EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, failing.getInflightCount());
        assertEquals(RequestLifecycleState.FAILED, scheduler.getRequestState(63L, 0).state());
    }

    @Test
    void decode_retirement_closes_handed_over_admission_lease_without_timeout_cancel() {
        Response route = successRoute(64);
        BatchItem item = new BatchItem(context(64), new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill("10.0.0.1:8080"),
                endpointRegistry.getDecode("10.0.0.2:8081"), System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        AtomicInteger activeLeaseCount = new AtomicInteger(1);
        AtomicInteger closeCallbacks = new AtomicInteger();
        InflightRegistrar leaseRegistrar = mock(InflightRegistrar.class);
        AdmissionLease lease = new AdmissionLease(item, item.decodeEp(),
                item.prefillEp().getBatcher().queueManager(), leaseRegistrar,
                10_000L, () -> {
                    closeCallbacks.incrementAndGet();
                    activeLeaseCount.decrementAndGet();
                });
        assertTrue(scheduler.bindAdmissionLease(item, lease));

        // Model the post-ACK state directly: the test's purpose is the
        // scheduler-owned lease handoff during endpoint retirement.
        lease.handoverToEngine();
        assertFalse(lease.isClosed());
        assertTrue(lease.hasSoftTimeoutRegistration());

        assertEquals(1, scheduler.retireEndpoint(item.decodeEp(),
                EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertTrue(lease.isClosed());
        assertFalse(lease.hasSoftTimeoutRegistration());
        assertEquals(0, activeLeaseCount.get());
        assertEquals(1, closeCallbacks.get());
        verify(leaseRegistrar, never()).finishYieldedById(anyLong(), anyString());

        assertEquals(0, scheduler.retireEndpoint(item.decodeEp(),
                EndpointRetireCause.HEALTH_CHECK_FAILED));
        assertEquals(1, closeCallbacks.get());
    }

    private static final class FailOnceRepackPrefillEndpoint extends PrefillEndpoint {
        private final AtomicBoolean failRepack = new AtomicBoolean(true);

        private FailOnceRepackPrefillEndpoint(EndpointId endpointId,
                                              WorkerStatus status,
                                              FlexlbConfig config,
                                              BatchDecisionHandler handler,
                                              BatchSchedulerReporter reporter) {
            super(endpointId, status, config, handler, reporter);
        }

        @Override
        public void repackBatch(long batchId, Set<Long> failedRequestIds) {
            if (failRepack.compareAndSet(true, false)) {
                throw new IllegalStateException("injected prefill repack failure");
            }
            super.repackBatch(batchId, failedRequestIds);
        }

        private void allowRepack() {
            failRepack.set(false);
        }
    }

    private static final class FailOnceReleaseDecodeEndpoint extends DecodeEndpoint {
        private final AtomicBoolean failRelease = new AtomicBoolean(true);

        private FailOnceReleaseDecodeEndpoint(EndpointId endpointId, WorkerStatus status) {
            super(endpointId, status);
        }

        @Override
        public void release(long requestId) {
            if (failRelease.compareAndSet(true, false)) {
                throw new IllegalStateException("injected decode release failure");
            }
            super.release(requestId);
        }

        private void allowRelease() {
            failRelease.set(false);
        }
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
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId), new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill("10.0.0.1:8080"), null,
                System.currentTimeMillis());
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

    private static BlockingRouteSubmittedContext blockingContext(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");

        BlockingRouteSubmittedContext ctx = new BlockingRouteSubmittedContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static final class BlockingRouteSubmittedContext extends BalanceContext {
        private final CountDownLatch handoffReached = new CountDownLatch(1);
        private final CountDownLatch releaseHandoff = new CountDownLatch(1);

        @Override
        public void setRouteSubmittedNanos(long routeSubmittedNanos) {
            super.setRouteSubmittedNanos(routeSubmittedNanos);
            handoffReached.countDown();
            try {
                if (!releaseHandoff.await(2, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("timed out waiting to release legacy handoff");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("legacy handoff interrupted", e);
            }
        }
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

    // ==================== P0-1: onTimeout terminal handling (PR-D) ====================

    @Test
    void onTimeout_settlesWithBatchSloExpired_andLateSuccessIsNoop() {
        // The dispatch timeout path (DispatchCallback.onTimeout) completes the
        // future with BATCH_SLO_EXPIRED, removes the inflight entry, and a late
        // onSuccess (stale ack) is a harmless no-op — the entry is already gone.
        BatchItem item = offerFailureItem(301);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onTimeout(item, new TimeoutException("test EnqueueBatch deadline"));

        Response response = item.future().getNow(null);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());

        // Late ack: entry already removed by finishEntry → entryFor returns null
        scheduler.onSuccess(item, 1L);
        Response unchanged = item.future().getNow(null);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), unchanged.getCode());

        // Idempotent: a second timeout is also a no-op
        scheduler.onTimeout(item, new TimeoutException("second"));
        Response stillUnchanged = item.future().getNow(null);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), stillUnchanged.getCode());
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
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
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

    private static WorkerStatus readyStatus(RoleType role,
                                            String ip,
                                            int httpPort,
                                            int grpcPort) {
        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp(ip);
        status.setPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.tryMarkReady();
        return status;
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
        // The fixture endpoints are first-generation registry publications.
        status.setEndpointGeneration(1);
        return status;
    }
}
