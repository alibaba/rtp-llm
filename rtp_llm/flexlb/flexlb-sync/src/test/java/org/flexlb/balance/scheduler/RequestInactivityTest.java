package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Matching Engine activity renews the lease; silence always ends local ownership. */
class RequestInactivityTest {
    private static final long REQUEST_ID = 101L;
    // Explicit observation/check timestamps advance virtual time without waiting for wall time.
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1L);

    private AbstractRequestScheduler registry;
    private RequestContext requestContext;
    private RequestRoute item;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private DeliveryClaim claim;
    private boolean deliveryCompleted;
    private final CompletableFuture<org.flexlb.balance.eviction.EngineCancelChannel.CancelAck> cleanup = new CompletableFuture<>();
    private long registeredAtMs;
    private long handedOffAtMs;

    @BeforeEach
    void setUp() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));

        var channel = mock(org.flexlb.balance.eviction.EngineCancelChannel.class);
        when(channel.cancel(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyLong())).thenReturn(cleanup);
        org.springframework.test.util.ReflectionTestUtils.setField(registry.runtime, "cancelChannel", channel);
        RequestContext context = RequestProtocolTestSupport.context(config, REQUEST_ID);
        CompletableFuture<Response> future = RequestProtocolTestSupport.register(registry, context);
        requestContext = registry.findRequestContext(REQUEST_ID);
        registeredAtMs = requestContext.createdAtMs();
        prefill = mock(PrefillEndpoint.class);
        decode = RequestProtocolTestSupport.decodeEndpoint();
        ServerStatus prefillStatus = new ServerStatus();
        prefillStatus.setRole(RoleType.PREFILL);
        prefillStatus.setServerIp("127.0.0.1");
        prefillStatus.setGrpcPort(8081);
        var reservation = new DecodeResources.ReservationHandle(1L, REQUEST_ID, 1L);
        context.setFuture(future);
        item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillStatus, null,
                prefill, decode, reservation, registeredAtMs);
        var registered = new RequestProtocolTestSupport.Registered(item, future);
        RequestProtocolTestSupport.bind(registry, registered);
        claim = RequestProtocolTestSupport.claimBatch(registry, item, 1L, () -> true);
        assertNotNull(claim);
        assertTrue(claim.item.ctx().scheduler().tryStartSend(claim));
        handedOffAtMs = (long) org.springframework.test.util.ReflectionTestUtils
                .getField(requestContext, "batchEnqueueStartedAtMs");
    }

    @AfterEach
    void tearDown() {
        if (registry != null && RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void activeStatusCancelsObsoleteVisibilityTimerWithoutEndingResourceTracking(RoleType source) throws Exception {
        acknowledgeDelivery();
        ExpirationTimer.DecisionDeadline deadline = RequestProtocolTestSupport.field(requestContext, "decisionDeadline");
        assertNotNull(deadline);
        ScheduledFuture<?> scheduled = (ScheduledFuture<?>)
                org.springframework.test.util.ReflectionTestUtils.getField(deadline, "scheduled");
        assertNotNull(scheduled);
        assertFalse(scheduled.isCancelled());
        if (source == RoleType.PREFILL) {
            registry.onPrefillStatus(requestContext, prefill, RoleType.PREFILL,
                    PrefillState.PrefillRequestStatus.active(item));
        } else {
            registry.onDecodeStatus(requestContext, decode,
                    DecodeResources.DecodeRequestStatus.active(item.decodeReservation()));
        }
        registry.runtime.continuations().awaitIdle();
        assertTrue(scheduled.isCancelled(), "engine evidence must cancel the obsolete scheduled callback");
        assertNull(RequestProtocolTestSupport.field(requestContext, "decisionDeadline"));
        requestContext.onDecisionVisibilityDeadline(deadline);
        assertLiveAndCharged();
        assertTrue(item.future().join().isSuccess());
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void repeatedActiveStatusRenewsLivenessWithoutSchedulingContinuation(RoleType source) throws Exception {
        acknowledgeDelivery();
        long firstAt = registeredAtMs + TIMEOUT_MS;
        Runnable first = acceptActive(source, firstAt);
        if (first != null) { first.run(); }
        for (int index = 1; index <= 1_000; index++) {
            assertNull(acceptActive(source, firstAt + index),
                    "repeated activity must not create a continuation without pending effects");
        }
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext,
                firstAt + 1_000 + TIMEOUT_MS - 1);
        assertTrue(registry.requests.isCurrent(requestContext), "the last matching heartbeat extends liveness");
        verify(prefill, never()).releaseRequest(item);
        verify(decode, never()).release(any(), any());
    }

    @Test
    void activeStatusPreservesPendingDecodeHandoffTimerInstallation() throws Exception {
        acknowledgeDelivery();
        long completedAt = registeredAtMs + TIMEOUT_MS;
        Runnable completed = requestContext.scheduler().acceptPrefillStatus(requestContext, prefill, RoleType.PREFILL,
                PrefillState.PrefillRequestStatus.terminal(item,
                        PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L), completedAt);
        assertNotNull(completed);
        // A second report can arrive while the completion's continuation is still queued.
        Runnable active = acceptActive(RoleType.PREFILL, completedAt + 1);
        assertNotNull(active, "pending timer installation is real work even when status does not advance");
        active.run();
        completed.run();
        assertNotNull(org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "decisionDeadline"));
        assertNull(acceptActive(RoleType.PREFILL, completedAt + 2),
                "installed handoff timer does not require a new continuation on every heartbeat");
    }

    @Test
    void singleStatusEntryRejectsForeignOwnerAndIsolatesMalformedStatus() throws Exception {
        acknowledgeDelivery();
        RequestContext foreign = mock(RequestContext.class);
        when(foreign.scheduler()).thenReturn(mock(AbstractRequestScheduler.class));
        registry.onDecodeStatus(foreign, decode, DecodeResources.DecodeRequestStatus.active(item.decodeReservation()));
        verify(foreign, never()).ownsDecodeReservationLocked(any(), any());
        registry.onPrefillStatus(requestContext, prefill, RoleType.PREFILL, null);
        registry.onDecodeStatus(requestContext, decode, null);
        registry.onDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(item.decodeReservation()));
        synchronized (requestContext) {
            assertTrue(requestContext.decodeAccepted(), "a malformed status must not prevent later valid activity");
        }
    }

    private Runnable acceptActive(RoleType source, long nowMs) {
        return source == RoleType.PREFILL
                ? requestContext.scheduler().acceptPrefillStatus(requestContext, prefill, RoleType.PREFILL,
                        PrefillState.PrefillRequestStatus.active(item), nowMs)
                : requestContext.scheduler().acceptDecodeStatus(requestContext, decode,
                        DecodeResources.DecodeRequestStatus.active(item.decodeReservation()), nowMs);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void repeatedStatusKeepsDeliveredRequestAliveBeyondItsOriginalDeadline(RoleType source) throws Exception {
        acknowledgeDelivery();
        for (int observation = 1; observation <= 6; observation++) {
            long observedAt = registeredAtMs + observation * (TIMEOUT_MS / 2L);
            recordWorkerActivity(source, observedAt);
            RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, observedAt + TIMEOUT_MS / 2L - 1L);
            assertLiveAndCharged();
        }
        assertTrue(item.future().isDone(), "the delivery response does not end activity tracking");
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void silenceRequestsCleanupAtTheLastMatchingStatusDeadline(RoleType source) throws Exception {
        acknowledgeDelivery();
        long lastStatusAt = registeredAtMs + 2L * TIMEOUT_MS;
        recordWorkerActivity(source, lastStatusAt);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, lastStatusAt + TIMEOUT_MS - 1L);
        assertLiveAndCharged();

        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, lastStatusAt + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);

        // A delayed status or timer callback cannot reopen or double-release this generation.
        RequestProtocolTestSupport.applyPrefillStatus(registry, prefill, RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(item));
        RequestProtocolTestSupport.applyDecodeStatus(registry, decode, DecodeResources.DecodeRequestStatus.active(item.decodeReservation()));
        RequestProtocolTestSupport.applyDecodeStatus(registry, decode, DecodeResources.DecodeRequestStatus.terminal(item.decodeReservation(), 0L));
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, lastStatusAt + 2L * TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @Test
    void missingDeliveryReplyRequiresSenderExitAndRemoteCleanup() throws Exception {
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handedOffAtMs + TIMEOUT_MS - 1L);
        assertFalse(item.future().isDone(), "no transport callback has confirmed delivery");
        assertLiveAndCharged();

        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handedOffAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
        assertFalse(item.future().get(1L, TimeUnit.SECONDS).isSuccess());

        // A late delivery acknowledgement must not resurrect the expired request.
        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, () -> claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered()));
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @Test
    void expiredSilenceRejectsAckWhileTheTimerContinuationIsBacklogged() throws Exception {
        ExpirationTimer.InactivityDeadline exact = (ExpirationTimer.InactivityDeadline) org.springframework.test.util.ReflectionTestUtils.getField(requestContext, "inactivityDeadline");
        assertNotNull(exact);
        RequestContinuationExecutor continuations = (RequestContinuationExecutor) org.springframework.test.util.ReflectionTestUtils.getField(registry, "continuations");
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        continuations.submit(requestContext, () -> {
            started.countDown();
            try {
                release.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        });
        assertTrue(started.await(1L, TimeUnit.SECONDS));
        try {
            long past = System.currentTimeMillis() - TIMEOUT_MS - 1_000L;
            org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "lastWorkerStatusAtMs", past);
            org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "batchEnqueueStartedAtMs", past);
            registry.enqueueInactivityDeadline(requestContext, exact, System.currentTimeMillis(), () -> {
            });
            completeDelivery(DeliveryResult.delivered());
            assertEquals(RequestState.Phase.TIMED_OUT, requestContext.snapshot().state());
            assertFalse(item.future().isDone(), "response publication follows the accepted cleanup");
        } finally {
            release.countDown();
        }
        registry.runtime.continuations().awaitIdle();
        assertFalse(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    @ParameterizedTest
    @EnumSource(value = DeliveryResult.Status.class, names = {"UNCERTAIN"})
    void uncertainDeliveryKeepsOnlyABoundedConfirmationWait(DeliveryResult.Status outcome) throws Exception {
        completeDelivery(new DeliveryResult(outcome, new IllegalStateException("reply was lost")));
        assertLiveAndCharged();
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handedOffAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
        assertFalse(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void anEarlierClientCancellationDoesNotDisableInactivityExpiration() throws Exception {
        acknowledgeDelivery();
        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                registry.cancel(REQUEST_ID, 0L, CancelReason.CLIENT_CANCELLED).state());
        assertEquals(1, registry.requests.liveRequestCount());
        assertFalse(claim.settlement().toCompletableFuture().isDone());

        synchronized (requestContext) {
            assertEquals(CancelReason.CLIENT_CANCELLED, RequestProtocolTestSupport.<CancelReason>inspect(registry, requestContext, "requireCancellationFirstCauseLocked"));
        }
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handedOffAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.CANCELLED);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, registeredAtMs + 2L * TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.CANCELLED);
    }

    @ParameterizedTest
    @EnumSource(StaleFact.class)
    void staleEndpointOrRequestGenerationCannotRenewInactivity(StaleFact source) {
        long lateStatusAt = registeredAtMs + 2L * TIMEOUT_MS;
        synchronized (requestContext) {
            Runnable observation = switch (source) {
                case PREFILL_ENDPOINT -> requestContext.scheduler().acceptPrefillStatus(requestContext, mock(PrefillEndpoint.class), RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(item), lateStatusAt);
                case PREFILL_ITEM -> requestContext.scheduler().acceptPrefillStatus(requestContext, prefill, RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(item.ctx()), SchedulingTestConfig.routeResponse(item), item.prefill(), null, prefill, decode, item.decodeReservation(), registeredAtMs)), lateStatusAt);
                case DECODE_ENDPOINT -> requestContext.scheduler().acceptDecodeStatus(requestContext, RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.active(item.decodeReservation()), lateStatusAt);
                case DECODE_GENERATION -> requestContext.scheduler().acceptDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(new DecodeResources.ReservationHandle(2L, REQUEST_ID, 1L)), lateStatusAt);
                case DECODE_RESERVATION -> requestContext.scheduler().acceptDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(new DecodeResources.ReservationHandle(1L, REQUEST_ID, 2L)), lateStatusAt);
            };
            org.junit.jupiter.api.Assertions.assertNull(observation);
            assertTrue(RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "requestInactiveLocked", lateStatusAt));
        }
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, lateStatusAt);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void statusBeforeCancellationCheckInvalidatesTheEarlierExpirationDecision(RoleType source) {
        long originalDeadline = handedOffAtMs + TIMEOUT_MS;
        synchronized (requestContext) {
            assertTrue(RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "requestInactiveLocked", originalDeadline), "the timer's earlier observation is expired");
        }

        recordWorkerActivity(source, originalDeadline - 1L);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, originalDeadline);

        assertLiveAndCharged();
        synchronized (requestContext) {
            assertFalse(RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "requestInactiveLocked", originalDeadline));
            assertTrue(RequestProtocolTestSupport.<Boolean>inspect(registry, requestContext, "requestInactiveLocked", originalDeadline - 1L + TIMEOUT_MS));
        }
    }

    private void recordWorkerActivity(RoleType source, long nowMs) {
        synchronized (requestContext) {
            if (source == RoleType.PREFILL) {
                requestContext.scheduler().acceptPrefillStatus(requestContext, prefill, RoleType.PREFILL, PrefillState.PrefillRequestStatus.active(item), nowMs);
            } else {
                requestContext.scheduler().acceptDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(item.decodeReservation()), nowMs);
            }
        }
    }

    private void assertLiveAndCharged() {
        assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        synchronized (requestContext) {
            assertSame(item, requestContext.activeRoute());
            assertFalse(requestContext.snapshot().state().isTerminal());
        }
        verify(decode, never()).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        verify(decode, never()).release(any(), eq(DecodeResources.ReleaseReason.REMOTE_CLEANUP));
        verify(prefill, never()).releaseRequest(any());
    }

    private void completeDelivery(DeliveryResult result) {
        claim.item.ctx().scheduler().completeDelivery(claim, result);
        deliveryCompleted = true;
    }

    private void acknowledgeDelivery() throws Exception {
        completeDelivery(DeliveryResult.delivered());
        assertTrue(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    private void assertExpiredAndReleased(RequestState.Phase expectedState) {
        if (!claim.settlement().toCompletableFuture().isDone()) {
            assertEquals(1, registry.requests.liveRequestCount(), "expiry retains cleanup ownership");
            cleanup.complete(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            if (!deliveryCompleted) {
                assertFalse(claim.settlement().toCompletableFuture().isDone(), "early cleanup cannot release the sender");
                completeDelivery(DeliveryResult.uncertain(new IllegalStateException("sender exited")));
            }
            claim.settlement().toCompletableFuture().join();
            registry.runtime.continuations().awaitIdle();
        }
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        assertEquals(expectedState, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(REQUEST_ID, 0L).state());
        synchronized (requestContext) {
            assertFalse(requestContext.isLiveGeneration());
        }
        verify(decode, times(1)).release(item.decodeReservation(), DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        verify(prefill, times(1)).releaseRequest(item);
    }

    private enum StaleFact {
        PREFILL_ENDPOINT,
        PREFILL_ITEM,
        DECODE_ENDPOINT,
        DECODE_GENERATION,
        DECODE_RESERVATION
    }
}
