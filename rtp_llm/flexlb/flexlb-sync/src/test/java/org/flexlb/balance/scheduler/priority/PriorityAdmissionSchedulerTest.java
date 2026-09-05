package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.RequestIdFixtures;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
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
import org.flexlb.engine.grpc.RequestId;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 1 tests for {@link PriorityAdmissionScheduler} + {@link PlanCommitter}
 * wired through {@link PriorityScheduler#submit}:
 * switch-off parity, switch-on placement parity, infeasible failure,
 * offer-failure rollback and optimistic-concurrency retry semantics.
 */
class PriorityAdmissionSchedulerTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080@0";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081@0";

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private PrioritySchedulerReporter priorityReporter;
    private PriorityAdmissionScheduler priorityScheduler;
    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private final List<EngineRpcService.EnqueueBatchRequestPB> sentBatches = new CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(2);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(10_000);
        SchedulingTestConfig.usePriorityQueue(config);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .reserve(ctx.getRequestId(), 128, 136,
                            ctx.getPriority());
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    sentBatches.add(request);
                    return CompletableFuture.completedFuture(ackFor(request));
                });

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        priorityScheduler = spy(new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel()));
        scheduler = new PriorityScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null,
                new UnsupportedEngineCancelChannel());

        // Prefill worker matching successRoute()
        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setGrpcPort(8081);
        prefillWs.setRole(RoleType.PREFILL);
        prefillWs.setAlive(true);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

        // Decode worker matching successRoute()
        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setRole(RoleType.DECODE);
        decodeWs.setAlive(true);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
    }

    @AfterEach
    void tearDown() {
        priorityScheduler.shutdown();
        scheduler.shutdown();
    }

    // ==================== switch off: legacy path parity ====================

    @Test
    void switch_off_uses_legacy_path_and_never_invokes_priority_scheduler() throws Exception {
        SchedulingTestConfig.useFifoQueue(config);

        CompletableFuture<Response> first = scheduler.submit(context("11"));
        CompletableFuture<Response> second = scheduler.submit(context("12"));

        assertTrue(first.get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.get(2, TimeUnit.SECONDS).isSuccess());
        verify(priorityScheduler, never()).schedule(any(), any(), any());
        verify(router, times(2)).route(any(BalanceContext.class));
        assertEquals(1, sentBatches.size());
    }

    // ==================== switch on, no pressure: parity with legacy router ====================

    @Test
    void switch_on_no_pressure_places_on_router_selected_pd_pair() throws Exception {
        CompletableFuture<Response> first = scheduler.submit(context("1"));
        CompletableFuture<Response> second = scheduler.submit(context("2"));

        Response firstResponse = first.get(2, TimeUnit.SECONDS);
        Response secondResponse = second.get(2, TimeUnit.SECONDS);
        assertTrue(firstResponse.isSuccess());
        assertTrue(secondResponse.isSuccess());
        assertTrue(firstResponse.isEnqueuedByMaster());
        assertTrue(secondResponse.isEnqueuedByMaster());

        // Same P/D placement as the legacy path: one batch on the router-selected prefill
        assertEquals(1, sentBatches.size());
        assertEquals(2, batchInputs(sentBatches.getFirst()).size());
        // Route response carries the router-selected prefill/decode pair untouched
        ServerStatus prefill = PriorityScheduler.findServer(firstResponse, RoleType.PREFILL);
        ServerStatus decode = PriorityScheduler.findServer(firstResponse, RoleType.DECODE);
        assertEquals("10.0.0.1", prefill.getServerIp());
        assertEquals("10.0.0.2", decode.getServerIp());
        verify(priorityReporter, times(2)).reportNormalPlacement(eq(50));
        verify(router, times(2)).route(any(BalanceContext.class));
    }

    // ==================== hasPriority gate removed: every request takes the priority path ====================

    @Test
    void request_without_priority_field_still_takes_priority_path_when_switch_on() throws Exception {
        // hasPriority gate removed from PriorityScheduler.submit():
        // normalize() always assigns 1-100 in production, so the switch is
        // the sole gate — even a raw priority-0 context goes through the
        // priority scheduler and is placed normally.
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        Response response = scheduler.submit(context("61", 0)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        verify(priorityScheduler).schedule(any(), any(), any());
        // The envelope carries the context priority untouched (0 here).
        verify(priorityReporter).reportNormalPlacement(eq(0));
    }

    @Test
    void request_without_priority_field_and_no_worker_isResourceExhausted() throws Exception {
        when(router.route(any(BalanceContext.class))).thenReturn(null);

        Response response = scheduler.submit(context("62", 0)).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        verify(priorityScheduler).schedule(any(), any(), any());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    // ==================== request expiration ====================

    @Test
    void expired_priority_request_remains_resource_exhausted() throws Exception {
        BalanceContext ctx = context("71");
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1_000));

        Response response = scheduler.submit(ctx).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        assertTrue(response.getErrorMessage().contains("request expired"));
        verify(router, never()).route(any(BalanceContext.class));
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void expired_fifo_request_remains_batch_slo_expired() throws Exception {
        SchedulingTestConfig.useFifoQueue(config);
        BalanceContext ctx = context("72");
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1_000));

        Response response = scheduler.submit(ctx).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED,
                response.getAdmissionRejectReason());
        verify(priorityScheduler, never()).schedule(any(), any(), any());
        verify(router, never()).route(any(BalanceContext.class));
    }

    // ==================== admission permit hard limit ====================

    @Test
    void admissionPermitLimitIsAtomicAcrossConcurrentScheduling() throws Exception {
        config.queueScheduler().getLifecycle().setMaxDeliveredNotAcceptedRequestsGlobal(1);
        config.queueScheduler().getLifecycle().setDeliveredNotAcceptedTimeoutMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);

        CountDownLatch firstRouteEntered = new CountDownLatch(1);
        CountDownLatch allowFirstRoute = new CountDownLatch(1);
        AtomicInteger routeCalls = new AtomicInteger();
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            if (routeCalls.incrementAndGet() == 1) {
                firstRouteEntered.countDown();
                if (!allowFirstRoute.await(2, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("timed out waiting to release first route");
                }
            }
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .reserve(ctx.getRequestId(), 128, 136,
                            ctx.getPriority());
            return successRoute(ctx.getRequestId());
        });

        ExecutorService submitter = Executors.newSingleThreadExecutor();
        Future<CompletableFuture<Response>> firstSubmission =
                submitter.submit(() -> scheduler.submit(contextWithBudget("73")));
        try {
            assertTrue(firstRouteEntered.await(2, TimeUnit.SECONDS));
            assertEquals(1, priorityScheduler.activeAdmissionCount());

            Response second = scheduler.submit(contextWithBudget("74"))
                    .get(1, TimeUnit.SECONDS);

            assertFalse(second.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), second.getCode());
            assertEquals(1, routeCalls.get(),
                    "the rejected request must not start placement");
            assertEquals(1, priorityScheduler.activeAdmissionCount());

            allowFirstRoute.countDown();
            Response first = firstSubmission.get(2, TimeUnit.SECONDS)
                    .get(2, TimeUnit.SECONDS);
            assertTrue(first.isSuccess());
            assertEquals(1, priorityScheduler.activeAdmissionCount());

            reportDecodePhase("73", TaskPhase.KV_ALLOCATED);
            assertEquals(0, priorityScheduler.activeAdmissionCount());
        } finally {
            allowFirstRoute.countDown();
            submitter.shutdownNow();
        }
    }

    @Test
    void admissionPermitIsReleasedWhenLeaseAttachmentThrows() {
        config.queueScheduler().getLifecycle().setMaxDeliveredNotAcceptedRequestsGlobal(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);

        InflightRegistrar registrar = openRegistrar();
        when(registrar.registerInflight(any(BatchItem.class))).thenReturn(true);
        when(registrar.attachAdmissionLease(any(BatchItem.class), any(AdmissionLease.class)))
                .thenThrow(new IllegalStateException("attach failed"));

        CompletableFuture<Response> response = new CompletableFuture<>();
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> priorityScheduler.schedule(
                        contextWithBudget("75"), response, registrar));

        assertEquals("attach failed", error.getMessage());
        assertEquals(0, priorityScheduler.activeAdmissionCount());
        verify(registrar).unregisterInflight(any(BatchItem.class));
    }

    @Test
    void successfulPrefillOnlyAdmissionsDoNotConsumeTheHardLimit() throws Exception {
        config.queueScheduler().getLifecycle().setMaxDeliveredNotAcceptedRequestsGlobal(1);
        config.queueScheduler().getLifecycle().setDeliveredNotAcceptedTimeoutMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(1);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            BalanceContext ctx = invocation.getArgument(0);
            return prefillOnlyRoute(ctx.getRequestId());
        });

        Response first = scheduler.submit(contextWithBudget("78"))
                .get(2, TimeUnit.SECONDS);
        Response second = scheduler.submit(contextWithBudget("79"))
                .get(2, TimeUnit.SECONDS);

        assertTrue(first.isSuccess(), first.getErrorMessage());
        assertTrue(second.isSuccess(), second.getErrorMessage());
        assertEquals(0, priorityScheduler.activeAdmissionCount());
        assertEquals(0, priorityScheduler.pendingSoftTimeoutLeaseCount());
    }

    @Test
    void shutdownCancelsPendingLeaseTimeoutAndRejectsNewAdmission() throws Exception {
        config.queueScheduler().getLifecycle().setDeliveredNotAcceptedTimeoutMs(60_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);

        InflightRegistrar registrar = openRegistrar();
        AtomicReference<AdmissionLease> attachedLease = new AtomicReference<>();
        when(registrar.registerInflight(any(BatchItem.class))).thenReturn(true);
        when(registrar.attachAdmissionLease(any(BatchItem.class), any(AdmissionLease.class)))
                .thenAnswer(invocation -> {
                    attachedLease.set(invocation.getArgument(1));
                    return true;
                });

        CompletableFuture<Response> admitted = new CompletableFuture<>();
        priorityScheduler.schedule(contextWithBudget("76"), admitted, registrar);
        admitted.complete(successRoute("76"));
        awaitSoftTimeoutQueueSize(1);

        assertTrue(priorityScheduler.removesCanceledSoftTimeouts());
        assertEquals(1, priorityScheduler.activeAdmissionCount());
        assertEquals(1, priorityScheduler.pendingSoftTimeoutLeaseCount());
        priorityScheduler.shutdown();

        assertTrue(priorityScheduler.isShutdown());
        assertEquals(0, priorityScheduler.softTimeoutQueueSize());
        assertEquals(0, priorityScheduler.activeAdmissionCount());
        assertEquals(0, priorityScheduler.pendingSoftTimeoutLeaseCount());
        assertEquals(2, attachedLease.get().leaseState());
        verify(registrar, never())
                .fenceAfterDeliveryTimeout(any(BatchItem.class), anyString());

        CompletableFuture<Response> rejected = new CompletableFuture<>();
        priorityScheduler.schedule(contextWithBudget("77"), rejected, registrar);
        Response rejection = rejected.get(1, TimeUnit.SECONDS);
        assertFalse(rejection.isSuccess());
        assertTrue(rejection.getErrorMessage().contains("shut down"));
        verify(router, times(1)).route(any(BalanceContext.class));
    }

    @Test
    void softTimeoutCallbackRunsOutsideLifecycleMonitorAndShutdownWaitsForIt()
            throws Exception {
        // Mockito creates a copy-based spy whose field-initialized timeout
        // scheduler still captures the original instance. This lifecycle test
        // must observe the same concrete instance that owns the timer.
        priorityScheduler.shutdown();
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel());
        config.queueScheduler().getLifecycle().setDeliveredNotAcceptedTimeoutMs(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);

        long firstRequestId = 176L;
        long secondRequestId = 177L;
        CountDownLatch firstFenceEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstFence = new CountDownLatch(1);
        CountDownLatch shutdownStarted = new CountDownLatch(1);
        InflightRegistrar registrar = openRegistrar();
        when(registrar.registerInflight(any(BatchItem.class))).thenReturn(true);
        when(registrar.attachAdmissionLease(any(BatchItem.class), any(AdmissionLease.class)))
                .thenReturn(true);
        when(registrar.fenceAfterDeliveryTimeout(any(BatchItem.class), anyString()))
                .thenAnswer(invocation -> {
                    BatchItem item = invocation.getArgument(0);
                    if (item.requestId().equals(String.valueOf(firstRequestId))) {
                        firstFenceEntered.countDown();
                        awaitLatch(releaseFirstFence);
                    }
                    return InflightRegistrar.PostDeliveryFenceResult.STARTED;
                });

        CompletableFuture<Response> first = new CompletableFuture<>();
        priorityScheduler.schedule(contextWithBudget(String.valueOf(firstRequestId)), first, registrar);
        assertTrue(first.complete(successRoute(String.valueOf(firstRequestId))));
        assertTrue(firstFenceEntered.await(1, TimeUnit.SECONDS));
        assertEquals(1, priorityScheduler.activeSoftTimeoutCallbackCount());

        CompletableFuture<Response> second = new CompletableFuture<>();
        priorityScheduler.schedule(contextWithBudget(String.valueOf(secondRequestId)), second, registrar);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Boolean> secondCompletion = executor.submit(
                    () -> second.complete(successRoute(String.valueOf(secondRequestId))));
            assertTrue(secondCompletion.get(1, TimeUnit.SECONDS),
                    "a running timeout callback must not retain the lifecycle monitor");
            assertEquals(1, priorityScheduler.pendingSoftTimeoutLeaseCount());

            Future<?> shutdown = executor.submit(() -> {
                shutdownStarted.countDown();
                priorityScheduler.shutdown();
            });
            assertTrue(shutdownStarted.await(1, TimeUnit.SECONDS));
            assertThrows(TimeoutException.class,
                    () -> shutdown.get(100, TimeUnit.MILLISECONDS),
                    "shutdown must wait for a callback which already crossed the gate");

            releaseFirstFence.countDown();
            shutdown.get(1, TimeUnit.SECONDS);
            assertEquals(0, priorityScheduler.activeSoftTimeoutCallbackCount());
            assertEquals(0, priorityScheduler.pendingSoftTimeoutLeaseCount());
            assertEquals(0, priorityScheduler.activeAdmissionCount());
        } finally {
            releaseFirstFence.countDown();
            executor.shutdownNow();
        }
    }

    // ==================== infeasible: no available worker ====================

    @Test
    void null_route_fails_with_no_available_worker() throws Exception {
        when(router.route(any(BalanceContext.class))).thenReturn(null);

        Response response = scheduler.submit(context("21")).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void routerCapacityFailureIsResourceExhaustedWithoutCausalSnapshot() throws Exception {
        when(router.route(any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_PREFILL_WORKER));

        Response response = scheduler.submit(context("22")).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
    }

    @Test
    void decodeCapacityBlockedByUnattributedEngineOccupantIsAdmissionUnavailable()
            throws Exception {
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests((long) (1));
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        WorkerStatus decodeStatus = decodeEp.getStatus();
        TaskInfo untrackedRunning = new TaskInfo();
        untrackedRunning.setRequestId("900");
        untrackedRunning.setPhase(TaskPhase.RUNNING);
        untrackedRunning.setInputLength(128L);
        WorkerStatusResponse workerStatus = new WorkerStatusResponse();
        workerStatus.setRunningTaskInfo(Map.of("900", untrackedRunning));
        decodeEp.onWorkerStatusUpdate(decodeStatus, workerStatus);

        // The worker-status task was never reserved by this Master: its
        // numeric sentinel/value is not trusted priority provenance.
        assertFalse(decodeEp.layeredAdmissionView().confirmed().getFirst().priorityKnown());
        when(router.route(any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        Response response = scheduler.submit(context("23")).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED,
                response.getAdmissionRejectReason());
    }

    // ==================== offer failure: decode reservation rollback ====================

    @Test
    void offer_failure_releases_decode_reservation_and_fails_explicitly() throws Exception {
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(1);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        // Route performs the decode reservation (D reserve first), like production Router
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            decodeEp.reserve(ctx.getRequestId(), 128, 136);
            return successRoute(ctx.getRequestId());
        });

        // Fill the single queue slot so tryOffer() must fail (P offer second)
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        assertTrue(batcher.tryOffer(dummyItem("999")));

        Response response = scheduler.submit(context("31")).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                response.getAdmissionRejectReason());
        // Primary offer plus one fallback route, then capacity fast-reject.
        verify(router, times(2)).route(any(BalanceContext.class));
        // Rollback: every decode reservation released (shadow load/KV restored)
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        assertEquals(0, decodeEp.getTotalLoad());
    }

    // ==================== helpers ====================

    private BatchItem dummyItem(String requestId) {
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId), new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL),
                PriorityScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill(PREFILL_IP_PORT), null,
                System.currentTimeMillis());
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        for (EngineRpcService.GenerateInputPB input : batchInputs(request)) {
            response.addSuccesses(RequestIdFixtures.write(EngineRpcService.EnqueueBatchSuccessPB.newBuilder(), RequestId.parse(input))
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

    private BalanceContext context(String requestId) {
        // Production requests always carry a normalized 1-100 priority
        // (normalize() default is 50); the raw-0 overload above documents
        // the removed hasPriority gate.
        return context(requestId, 50);
    }

    private BalanceContext context(String requestId, int priority) {
        Request request = new Request();
        request.setRequestId(String.valueOf(requestId));
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private BalanceContext contextWithBudget(String requestId) {
        BalanceContext ctx = context(requestId);
        long nowMs = System.currentTimeMillis();
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(
                ctx.getPriority(), nowMs + 60_000));
        return ctx;
    }

    private static InflightRegistrar openRegistrar() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        when(registrar.isAdmissionOpen(ArgumentMatchers.anyString(), any())).thenReturn(true);
        when(registrar.claimAdmissionMutation(ArgumentMatchers.anyString(), any())).thenReturn(true);
        return registrar;
    }

    private void reportDecodePhase(String requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setPhase(phase);
        task.setInputLength(128);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setRunningTaskInfo(Map.of(String.valueOf(requestId), task));
        scheduler.onWorkerStatusUpdate(response);
    }

    private void awaitSoftTimeoutQueueSize(int expected) throws InterruptedException {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (priorityScheduler.softTimeoutQueueSize() != expected
                && System.nanoTime() < deadlineNanos) {
            Thread.sleep(1);
        }
        assertEquals(expected, priorityScheduler.softTimeoutQueueSize());
    }

    private static void awaitLatch(CountDownLatch latch) {
        boolean interrupted = false;
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        try {
            while (latch.getCount() != 0) {
                long remainingNanos = deadlineNanos - System.nanoTime();
                assertTrue(remainingNanos > 0, "latch did not open before timeout");
                try {
                    assertTrue(latch.await(remainingNanos, TimeUnit.NANOSECONDS),
                            "latch did not open before timeout");
                } catch (InterruptedException shutdownInterrupt) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private static byte[] generateInputBytes(String requestId) {
        EngineRpcService.GenerateInputPB input = RequestIdFixtures.write(EngineRpcService.GenerateInputPB.newBuilder(), requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static Response prefillOnlyRoute(String requestId) {
        Response response = new Response();
        response.setSuccess(true);

        ServerStatus prefill = new ServerStatus();
        prefill.setSuccess(true);
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setGroup("g1");
        prefill.setRequestId(String.valueOf(requestId));
        response.setServerStatus(List.of(prefill));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(String.valueOf(requestId));
        return status;
    }
}
