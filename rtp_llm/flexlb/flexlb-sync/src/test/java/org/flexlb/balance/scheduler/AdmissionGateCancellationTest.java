package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AdmissionGateCancellationTest {

    private PriorityScheduler scheduler;
    private PriorityAdmissionScheduler admissionScheduler;
    private EngineCancelChannel cancelChannel;
    private ConfigService configService;
    private FlexlbConfig config;
    private EndpointRegistry endpointRegistry;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        when(configService.loadBalanceConfig()).thenReturn(config);
        admissionScheduler = mock(PriorityAdmissionScheduler.class);
        cancelChannel = mock(EngineCancelChannel.class);
        endpointRegistry = mock(EndpointRegistry.class);
        scheduler = new PriorityScheduler(
                configService,
                mock(Router.class),
                endpointRegistry,
                mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class),
                admissionScheduler,
                null,
                cancelChannel);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void cancelBeforeInflightRegistrationClosesTheExistingGeneration()
            throws Exception {
        long requestId = 20_001L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> scheduleResult = scheduler.submit(context);

        RequestLifecycleSnapshot cancelled = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);

        assertNotNull(cancelled);
        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
        Response response = scheduleResult.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(8504, response.getCode());
        assertFalse(scheduler.registerInflight(item(context, scheduleResult)),
                "a cancel-owned admission gate must reject a later commit");
        assertEquals(cancelled,
                scheduler.cancelRequest(
                        requestId, 0, CancelReason.CLIENT_CANCELLED));
        verify(admissionScheduler).schedule(context, scheduleResult, scheduler);
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void cancelAndRegistrationShareOneLatchLinearization() throws Exception {
        long requestId = 20_002L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> scheduleResult = scheduler.submit(context);
        BatchItem item = item(context, scheduleResult);
        CountDownLatch start = new CountDownLatch(1);

        CompletableFuture<Boolean> registration = CompletableFuture.supplyAsync(() -> {
            await(start);
            return scheduler.registerInflight(item);
        });
        CompletableFuture<RequestLifecycleSnapshot> cancellation =
                CompletableFuture.supplyAsync(() -> {
                    await(start);
                    return scheduler.cancelRequest(
                            requestId, 0, CancelReason.CLIENT_CANCELLED);
                });

        start.countDown();
        registration.get(1, TimeUnit.SECONDS);
        RequestLifecycleSnapshot cancelled =
                cancellation.get(1, TimeUnit.SECONDS);
        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
        assertEquals(8504, scheduleResult.get(1, TimeUnit.SECONDS).getCode());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void admissionDeadlineTombstoneRejectsLateRegistrationAfterGateRemoval()
            throws Exception {
        long requestId = 20_003L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> scheduleResult = scheduler.submit(context);

        scheduler.onRequestExpired(requestId, scheduleResult);

        Response response = scheduleResult.get(1, TimeUnit.SECONDS);
        assertEquals(8511, response.getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.getRequestState(requestId, 0).state());
        assertEquals(0, scheduler.generationGateCount());
        assertFalse(scheduler.registerInflight(item(context, scheduleResult)),
                "a removed gate must not let its timed-out generation resurrect");
        assertEquals(RequestLifecycleState.TIMED_OUT,
                scheduler.cancelRequest(
                        requestId, 0, CancelReason.CLIENT_CANCELLED).state());
    }

    @Test
    void clientCancelWinsAgainstQueuedAdmissionDeadline() throws Exception {
        long requestId = 20_004L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> scheduleResult = scheduler.submit(context);

        RequestLifecycleSnapshot cancelled = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);
        scheduler.onRequestExpired(requestId, scheduleResult);

        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
        assertEquals(8504, scheduleResult.get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(requestId, 0).state());
    }

    @Test
    void activeGenerationCannotBeOverlaidByDuplicateSubmit() throws Exception {
        long requestId = 20_005L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> original = scheduler.submit(context);
        assertTrue(scheduler.registerInflight(item(context, original)));

        Response duplicate = scheduler.submit(context).get(1, TimeUnit.SECONDS);

        assertFalse(duplicate.isSuccess());
        assertEquals(1, scheduler.generationGateCount());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.cancelRequest(
                        requestId, 0, CancelReason.CLIENT_CANCELLED).state());
        assertEquals(8504, original.get(1, TimeUnit.SECONDS).getCode());
        assertEquals(0, scheduler.generationGateCount());
    }

    @Test
    void externalFutureCleanupRemovesExactGenerationGate() {
        long requestId = 20_006L;
        BalanceContext context = context(requestId);
        CompletableFuture<Response> result = scheduler.submit(context);
        BatchItem item = item(context, result);
        assertTrue(scheduler.registerInflight(item));
        AdmissionLease lease = new AdmissionLease(item, null, null, scheduler,
                0, null, null);
        assertTrue(scheduler.attachAdmissionLease(item, lease));
        lease.bindTo(result);

        assertTrue(result.cancel(false));

        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, scheduler.generationGateCount());
        CompletableFuture<Response> reused = scheduler.submit(context);
        assertFalse(reused.isDone(), "the detached non-tombstoned id is reusable");
        scheduler.cancelRequest(requestId, 0, CancelReason.CLIENT_CANCELLED);
    }

    @Test
    void legacyRoutingCannotCommitAfterCancelClosesGeneration() throws Exception {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        CountDownLatch routing = new CountDownLatch(1);
        CountDownLatch finishRouting = new CountDownLatch(1);
        Router router = mock(Router.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        Response route = item(context(20_007L), new CompletableFuture<>()).routeResponse();
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.0.2");
        decodeStatus.setHttpPort(8081);
        decodeStatus.setRequestId(20_007L);
        route.setServerStatus(new java.util.ArrayList<>(route.getServerStatus()));
        route.getServerStatus().add(decodeStatus);
        AtomicBoolean decodeReserved = new AtomicBoolean();
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            decodeReserved.set(true);
            routing.countDown();
            await(finishRouting);
            return route;
        });
        when(registry.getPrefill("10.0.0.1:8080")).thenReturn(prefill);
        when(registry.getDecode("10.0.0.2:8081")).thenReturn(decode);
        when(prefill.getBatcher()).thenReturn(batcher);
        doAnswer(invocation -> {
            decodeReserved.set(false);
            return null;
        }).when(decode).release(20_007L);
        scheduler = new PriorityScheduler(
                configService, router, registry, mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class), admissionScheduler, null, cancelChannel);
        BalanceContext context = context(20_007L);

        CompletableFuture<CompletableFuture<Response>> submission =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context));
        assertTrue(routing.await(1, TimeUnit.SECONDS));
        RequestLifecycleSnapshot cancelled = scheduler.cancelRequest(
                20_007L, 0, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.CANCEL_REQUESTED, cancelled.state());
        assertEquals(cancelled, scheduler.getRequestState(20_007L, 0));
        assertTrue(decodeReserved.get());
        finishRouting.countDown();
        CompletableFuture<Response> result = submission.get(1, TimeUnit.SECONDS);

        assertEquals(8504, result.get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.CANCELLED,
                scheduler.getRequestState(20_007L, 0).state());
        assertFalse(decodeReserved.get(),
                "Cancel must not become terminal before route reservation cleanup");
        verify(decode).release(20_007L);
        verify(batcher, never()).tryOffer(any());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, scheduler.generationGateCount());
    }

    @Test
    void legacyQueuedExternalFutureCancelUsesSchedulerReducer() {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        Router router = mock(Router.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        PrefillQueueManager queueManager = mock(PrefillQueueManager.class);
        long requestId = 20_008L;
        Response route = item(context(requestId), new CompletableFuture<>()).routeResponse();
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.0.2");
        decodeStatus.setHttpPort(8081);
        decodeStatus.setRequestId(requestId);
        route.setServerStatus(new java.util.ArrayList<>(route.getServerStatus()));
        route.getServerStatus().add(decodeStatus);
        when(router.route(any(BalanceContext.class))).thenReturn(route);
        when(registry.getPrefill("10.0.0.1:8080")).thenReturn(prefill);
        when(registry.getDecode("10.0.0.2:8081")).thenReturn(decode);
        when(prefill.getBatcher()).thenReturn(batcher);
        when(batcher.queueManager()).thenReturn(queueManager);
        when(batcher.tryOffer(any())).thenReturn(true);
        scheduler = new PriorityScheduler(
                configService, router, registry, mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class), admissionScheduler, null, cancelChannel);

        CompletableFuture<Response> result = scheduler.submit(context(requestId));
        assertEquals(1, scheduler.getInflightSize());

        assertTrue(result.cancel(false));

        verify(decode).release(requestId);
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, scheduler.generationGateCount());
        verify(cancelChannel, never()).cancel(any(), anyLong(), anyLong());
    }

    @Test
    void legacyOfferFailureKeepsOwnershipWhileCancelRacesResourceUnwind()
            throws Exception {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        Router router = mock(Router.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        PrefillQueueManager queueManager = mock(PrefillQueueManager.class);
        long requestId = 20_009L;
        Response route = item(context(requestId), new CompletableFuture<>()).routeResponse();
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.0.2");
        decodeStatus.setHttpPort(8081);
        decodeStatus.setRequestId(requestId);
        route.setServerStatus(new java.util.ArrayList<>(route.getServerStatus()));
        route.getServerStatus().add(decodeStatus);
        when(router.route(any(BalanceContext.class))).thenReturn(route);
        when(registry.getPrefill("10.0.0.1:8080")).thenReturn(prefill);
        when(registry.getDecode("10.0.0.2:8081")).thenReturn(decode);
        when(prefill.getBatcher()).thenReturn(batcher);
        when(batcher.queueManager()).thenReturn(queueManager);
        when(batcher.tryOffer(any())).thenThrow(new IllegalStateException("offer failed"));
        CountDownLatch cleanupStarted = new CountDownLatch(1);
        CountDownLatch releaseCleanup = new CountDownLatch(1);
        doAnswer(invocation -> {
            cleanupStarted.countDown();
            await(releaseCleanup);
            return null;
        }).when(queueManager).tryRemove(requestId, "TERMINAL_RELEASE");
        scheduler = new PriorityScheduler(
                configService, router, registry, mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class), admissionScheduler, null, cancelChannel);

        CompletableFuture<CompletableFuture<Response>> submission =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context(requestId)));
        assertTrue(cleanupStarted.await(1, TimeUnit.SECONDS));

        RequestLifecycleSnapshot observed = scheduler.cancelRequest(
                requestId, 0, CancelReason.CLIENT_CANCELLED);
        assertEquals(RequestLifecycleState.QUEUED, observed.state(),
                "the earlier offer failure owns terminal publication");

        releaseCleanup.countDown();
        CompletableFuture<Response> result = submission.get(1, TimeUnit.SECONDS);
        assertEquals(8510, result.get(1, TimeUnit.SECONDS).getCode());
        assertEquals(RequestLifecycleState.FAILED,
                scheduler.getRequestState(requestId, 0).state());
        verify(decode).release(requestId);
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, scheduler.generationGateCount());
    }

    @Test
    void shutdownCompletesGenerationHeldInLegacyRouteAndStillRollsBack()
            throws Exception {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        Router router = mock(Router.class);
        EndpointRegistry registry = mock(EndpointRegistry.class);
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        WorkerBatcher batcher = mock(WorkerBatcher.class);
        long requestId = 20_010L;
        Response route = item(context(requestId), new CompletableFuture<>()).routeResponse();
        ServerStatus decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setServerIp("10.0.0.2");
        decodeStatus.setHttpPort(8081);
        decodeStatus.setRequestId(requestId);
        route.setServerStatus(new java.util.ArrayList<>(route.getServerStatus()));
        route.getServerStatus().add(decodeStatus);
        CountDownLatch routeReserved = new CountDownLatch(1);
        CountDownLatch releaseRoute = new CountDownLatch(1);
        AtomicBoolean decodeReserved = new AtomicBoolean();
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation -> {
            decodeReserved.set(true);
            routeReserved.countDown();
            await(releaseRoute);
            return route;
        });
        when(registry.getPrefill("10.0.0.1:8080")).thenReturn(prefill);
        when(registry.getDecode("10.0.0.2:8081")).thenReturn(decode);
        when(prefill.getBatcher()).thenReturn(batcher);
        doAnswer(invocation -> {
            decodeReserved.set(false);
            return null;
        }).when(decode).release(requestId);
        scheduler = new PriorityScheduler(
                configService, router, registry, mock(BatchDispatcher.class),
                mock(BatchSchedulerReporter.class), admissionScheduler, null, cancelChannel);

        CompletableFuture<CompletableFuture<Response>> submission =
                CompletableFuture.supplyAsync(() -> scheduler.submit(context(requestId)));
        assertTrue(routeReserved.await(1, TimeUnit.SECONDS));

        scheduler.shutdown();
        releaseRoute.countDown();
        CompletableFuture<Response> result = submission.get(1, TimeUnit.SECONDS);

        assertEquals(8510, result.get(1, TimeUnit.SECONDS).getCode());
        assertFalse(decodeReserved.get());
        verify(decode).release(requestId);
        verify(batcher, never()).tryOffer(any());
        assertEquals(0, scheduler.getInflightSize());
        assertEquals(0, scheduler.generationGateCount());
    }

    @Test
    void orphanCleanupRetainsReservationOwnedByActiveAdmissionMutation() {
        long requestId = 20_011L;
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(0);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        RequestInflight reservation = new RequestInflight(
                128,
                136,
                System.currentTimeMillis() - 1_000,
                50,
                DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN);
        when(decode.reservedView()).thenReturn(Map.of(requestId, reservation));
        when(decode.releaseReservationIfCurrent(requestId, reservation))
                .thenReturn(true);
        when(endpointRegistry.getDecodeEndpoints())
                .thenReturn(new ConcurrentHashMap<>(
                        Map.of("10.0.0.2:8081", decode)));

        CompletableFuture<Response> result = scheduler.submit(context(requestId));
        assertTrue(scheduler.claimAdmissionMutation(requestId, result));

        scheduler.cleanupInflight();

        verify(decode, never()).releaseReservationIfCurrent(
                requestId, reservation);
        scheduler.completeAdmissionMutation(requestId, result);
        scheduler.cleanupInflight();
        verify(decode).releaseReservationIfCurrent(requestId, reservation);
        result.cancel(false);
    }

    @Test
    void orphanCleanupCannotOvertakeAdmissionMutationCommit() throws Exception {
        long requestId = 20_012L;
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(0);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        RequestInflight reservation = new RequestInflight(
                128,
                136,
                System.currentTimeMillis() - 1_000,
                50,
                DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN);
        CountDownLatch snapshotCaptured = new CountDownLatch(1);
        CountDownLatch continueCleanup = new CountDownLatch(1);
        when(decode.reservedView()).thenAnswer(invocation -> {
            snapshotCaptured.countDown();
            await(continueCleanup);
            return Map.of(requestId, reservation);
        });
        when(endpointRegistry.getDecodeEndpoints())
                .thenReturn(new ConcurrentHashMap<>(
                        Map.of("10.0.0.2:8081", decode)));
        BalanceContext context = context(requestId);
        CompletableFuture<Response> result = scheduler.submit(context);
        assertTrue(scheduler.claimAdmissionMutation(requestId, result));

        CompletableFuture<Void> cleanup = CompletableFuture.runAsync(
                scheduler::cleanupInflight);
        assertTrue(snapshotCaptured.await(1, TimeUnit.SECONDS));
        assertTrue(scheduler.registerInflight(item(context, result)));
        scheduler.completeAdmissionMutation(requestId, result);
        continueCleanup.countDown();
        cleanup.get(1, TimeUnit.SECONDS);

        verify(decode, never()).releaseReservationIfCurrent(
                requestId, reservation);
        assertEquals(1, scheduler.getInflightSize());
        result.cancel(false);
        assertEquals(0, scheduler.getInflightSize());
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(50);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return context;
    }

    private static BatchItem item(
            BalanceContext context,
            CompletableFuture<Response> future) {
        Response route = new Response();
        route.setSuccess(true);
        ServerStatus prefill = new ServerStatus();
        prefill.setSuccess(true);
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        route.setServerStatus(java.util.List.of(prefill));
        return new BatchItem(
                context,
                future,
                route,
                prefill,
                null,
                null,
                null,
                System.currentTimeMillis());
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(1, TimeUnit.SECONDS));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AssertionError(error);
        }
    }
}
