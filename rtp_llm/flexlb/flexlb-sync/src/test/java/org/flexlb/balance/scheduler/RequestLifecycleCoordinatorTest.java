package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.scheduler.RequestLifecycleTestSupport.Registered;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.commitRoute;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Canonical request-generation ownership tests, independent of the facade. */
class RequestLifecycleCoordinatorTest {

    private FlexlbConfig config;
    private RequestRegistry lifecycle;
    private EngineCancelChannel engineCancelChannel;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        engineCancelChannel = mock(EngineCancelChannel.class);
        lifecycle = new RequestRegistry(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class),
                engineCancelChannel);
    }

    @AfterEach
    void tearDown() {
        if (lifecycle.closeAdmissionAndAwaitMutations()) {
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
        }
    }

    @Test
    void duplicateRegistrationCannotReplaceTheCanonicalExactGeneration() {
        BalanceContext context = context(101L);
        CompletableFuture<Response> canonical = lifecycle.register(context, 8);

        CompletableFuture<Response> duplicate = lifecycle.register(context(101L), 8);

        assertFalse(canonical.isDone());
        assertTrue(duplicate.isDone());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                duplicate.join().getCode());
        assertSame(canonical, lifecycle.requestSlot(101L).future());
        assertEquals(1, lifecycle.liveRequestCount());
    }

    @Test
    void slotLockContractIsEnforcedWithoutJvmAssertions() {
        lifecycle.register(context(102L), 8);
        RequestSlot slot = lifecycle.requestSlot(102L);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class, slot::activeItem);

        assertTrue(failure.getMessage().contains("requires slot lock"));
    }

    @Test
    void globalOutstandingPermitIsAtomicAndReusableAfterLocalTerminal() {
        CompletableFuture<Response> first = lifecycle.register(context(201L), 1);
        CompletableFuture<Response> rejected = lifecycle.register(context(202L), 1);

        assertFalse(first.isDone());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                rejected.join().getCode());

        RequestState.Snapshot cancellation = lifecycle.cancelRequest(
                201L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(cancellation);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                first.join().getCode());

        CompletableFuture<Response> admittedAgain =
                lifecycle.register(context(203L), 1);
        assertFalse(admittedAgain.isDone(),
                "the exact terminal must release its one outstanding permit");
    }

    @Test
    void decodeAcceptanceLimitIsAtomicAndReusableAfterLocalTerminal() {
        Registered first = registerItem(211L);
        Registered second = registerItem(212L);

        assertEquals(RequestRegistry.RouteCommitResult.PUBLISHED,
                commitRoute(lifecycle, first, 1, 30_000L));
        assertEquals(
                RequestRegistry.RouteCommitResult.ACCEPTANCE_LIMIT_REACHED,
                commitRoute(lifecycle, second, 1, 30_000L));
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest(211L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(0, lifecycle.decodeAcceptanceCount());
        assertEquals(RequestRegistry.RouteCommitResult.PUBLISHED,
                commitRoute(lifecycle, second, 1, 30_000L));
        assertEquals(1, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest(212L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void zeroDecodeAcceptanceLimitKeepsTheGuardUnbounded() {
        Registered first = registerItem(221L);
        Registered second = registerItem(222L);

        assertEquals(RequestRegistry.RouteCommitResult.PUBLISHED,
                commitRoute(lifecycle, first, 0, 30_000L));
        assertEquals(RequestRegistry.RouteCommitResult.PUBLISHED,
                commitRoute(lifecycle, second, 0, 30_000L));
        assertEquals(2, lifecycle.decodeAcceptanceCount());

        lifecycle.cancelRequest(221L, 0L, CancelReason.CLIENT_CANCELLED);
        lifecycle.cancelRequest(222L, 0L, CancelReason.CLIENT_CANCELLED);
        assertEquals(0, lifecycle.decodeAcceptanceCount());
    }

    @Test
    void admissionMutationDefersCancellationUntilItsExactCapabilityCloses() {
        CompletableFuture<Response> future = lifecycle.register(context(301L), 4);
        RequestRegistry.AdmissionScope scope =
                lifecycle.beginAdmission(301L, future);
        assertNotNull(scope);

        RequestState.Snapshot requested = lifecycle.cancelRequest(
                301L, 0L, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestState.Phase.CANCEL_REQUESTED, requested.state());
        assertFalse(future.isDone(),
                "the admission mutation still owns rollback and terminal cleanup");

        scope.close();

        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
        assertEquals(RequestState.Phase.CANCELLED,
                lifecycle.getRequestState(301L, 0L).state());
    }

    @Test
    void shutdownGateWaitsForTheExactAdmissionMutationAndRejectsNewWork()
            throws Exception {
        CompletableFuture<Response> heldFuture =
                lifecycle.register(context(401L), 4);
        RequestRegistry.AdmissionScope held =
                lifecycle.beginAdmission(401L, heldFuture);
        assertNotNull(held);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Boolean> shutdownOwner =
                    executor.submit(lifecycle::closeAdmissionAndAwaitMutations);
            awaitCondition(lifecycle::isShuttingDown);
            assertFalse(shutdownOwner.isDone(),
                    "shutdown must not overtake an exact admission mutation");

            CompletableFuture<Response> rejected =
                    lifecycle.register(context(402L), 4);
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    rejected.join().getCode());

            held.close();
            assertTrue(shutdownOwner.get(5, TimeUnit.SECONDS));
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
            assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                    heldFuture.get(5, TimeUnit.SECONDS).getCode());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void cancelRequiresTheExpectedBatchGenerationAndUnknownIdsStayAbsent() {
        CompletableFuture<Response> future = lifecycle.register(context(501L), 4);

        assertNull(lifecycle.cancelRequest(
                999L, 0L, CancelReason.CLIENT_CANCELLED));
        assertNull(lifecycle.cancelRequest(
                501L, 91L, CancelReason.CLIENT_CANCELLED));
        assertFalse(future.isDone());

        RequestState.Snapshot exact = lifecycle.cancelRequest(
                501L, 0L, CancelReason.CLIENT_CANCELLED);
        assertNotNull(exact);
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                future.join().getCode());
    }

    @Test
    void workerActivityExtendsOnlyTheInactiveMaintenanceTtl() {
        Registered registered = registerItem(601L);
        RequestLifecycleTestSupport.bind(lifecycle, registered);
        RequestSlot slot = lifecycle.requestSlot(601L);
        long ttlMs = 300_000L;
        long heartbeatAtMs = slot.createdAtMs() + ttlMs + 1_000L;
        synchronized (slot) {
            slot.observeWorkerStatus(heartbeatAtMs);
        }

        assertFalse(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs - 1L, ttlMs));
        assertEquals(RequestState.Phase.QUEUED,
                lifecycle.getRequestState(601L, 0L).state());

        assertTrue(lifecycle.reduceStale(
                slot, heartbeatAtMs + ttlMs + 1L, ttlMs));
        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(601L, 0L).state());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void staleDeliveredRequestReclaimsLocalOwnershipWithoutEngineCancel() {
        Registered registered = registerItem(602L);
        assertEquals(RequestRegistry.RouteCommitResult.PUBLISHED,
                commitRoute(lifecycle, registered, 0, 30_000L));
        RequestSlot slot = lifecycle.requestSlot(602L);
        long ttlMs = 300_000L;

        assertTrue(lifecycle.reduceStale(
                slot, slot.createdAtMs() + ttlMs + 1L, ttlMs));

        assertEquals(RequestState.Phase.TIMED_OUT,
                lifecycle.getRequestState(602L, 0L).state());
        verify(registered.item().decodeEp()).releasePlacementExact(
                registered.item().decodeReservation());
        verify(engineCancelChannel, never()).cancel(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyLong());
    }

    private BalanceContext context(long requestId) {
        return RequestLifecycleTestSupport.context(config, requestId);
    }

    private Registered registerItem(long requestId) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future = lifecycle.register(context, 4);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(1L, requestId, 1L);
        return new Registered(
                new ScheduledRequest(
                        context,
                        future,
                        new Response(),
                        null,
                        null,
                        null,
                        decode,
                        reservation,
                        System.currentTimeMillis()),
                future);
    }

}
