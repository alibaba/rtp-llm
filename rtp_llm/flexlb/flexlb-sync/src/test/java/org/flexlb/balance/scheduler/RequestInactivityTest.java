package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import static org.flexlb.balance.scheduler.ContextResult.*;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
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
        requestContext = registry.requests.findActive(REQUEST_ID);
        registeredAtMs = requestContext.snapshot().createdAtMs();
        prefill = mock(PrefillEndpoint.class);
        decode = RequestProtocolTestSupport.decodeEndpoint();
        ServerStatus prefillStatus = new ServerStatus();
        prefillStatus.setRole(RoleType.PREFILL);
        prefillStatus.setServerIp("127.0.0.1");
        prefillStatus.setGrpcPort(8081);
        var reservation = new DecodeResources.ReservationHandle(1L, REQUEST_ID, 1L);
        context.data().setFuture(future);
        item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, new Response(), prefillStatus, null,
                prefill, decode, reservation, registeredAtMs);
        var registered = new RequestProtocolTestSupport.Registered(item, future);
        RequestProtocolTestSupport.bind(registry, registered);
        claim = RequestProtocolTestSupport.claimBatch(registry, item, 1L, () -> true);
        assertNotNull(claim);
        assertTrue(claim.item().ctx().data().getScheduler().tryStartSend(claim));
        handedOffAtMs = requestContext.snapshot().delivery().startedAtMs();
    }

    @AfterEach
    void tearDown() {
        if (registry != null && RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)).close();
            SchedulerTestSupport.closeRequestExecutors(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void activeStatusCancelsObsoleteVisibilityTimerWithoutEndingResourceTracking(RoleType source) throws Exception {
        acknowledgeDelivery();
        ExpirationTimer.Token deadline = requestContext.snapshot().deadlines().visibility();
        assertNotNull(deadline);
        ScheduledFuture<?> scheduled = scheduledVisibility();
        assertNotNull(scheduled);
        assertFalse(scheduled.isCancelled());
        if (source == RoleType.PREFILL) {
            active(RoleType.PREFILL, System.currentTimeMillis());
        } else {
            active(RoleType.DECODE, System.currentTimeMillis());
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertTrue(scheduled.isCancelled(), "engine evidence must cancel the obsolete scheduled callback");
        assertNull(requestContext.snapshot().deadlines().visibility());
        registry.onDeadline(deadline, deadline.atMs());
        assertLiveAndCharged();
        assertTrue(item.future().join().isSuccess());
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void repeatedActiveStatusRenewsLivenessWithoutSchedulingContinuation(RoleType source) throws Exception {
        acknowledgeDelivery();
        long firstAt = registeredAtMs + TIMEOUT_MS;
        active(source, firstAt);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        var oldPlan = requestContext.snapshot().deadlines();
        var workers = SchedulerTestSupport.requestWorkers(registry.runtime);
        var observer = org.mockito.Mockito.spy(workers);
        org.springframework.test.util.ReflectionTestUtils.setField(registry.runtime, "requestWorkers", observer);
        try {
            for (int index = 1; index <= 1_000; index++) {
                active(source, firstAt + index);
                assertSame(oldPlan.inactivity(), requestContext.snapshot().deadlines().inactivity(),
                        "repeated activity must not replace an earlier wake-up before its identity is consumed");
            }
            verify(observer, never()).execute(any(Runnable.class));
        } finally { org.springframework.test.util.ReflectionTestUtils.setField(registry.runtime, "requestWorkers", workers); }
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, firstAt + 1_000 + TIMEOUT_MS - 1);
        assertTrue(registry.requests.isCurrent(requestContext), "the last matching heartbeat extends liveness");
        verify(prefill, never()).release(null, item);
        verify(decode, never()).release(any(), any());
    }

    @Test
    void activeStatusPreservesPendingDecodeHandoffTimerInstallation() throws Exception {
        acknowledgeDelivery();
        long completedAt = registeredAtMs + TIMEOUT_MS;
        var completed = requestContext.recordEngineEvidence(new EngineEvidence(item, null, RoleType.PREFILL,
                EngineKind.COMPLETED, 0L, null, completedAt));
        assertEquals(UpdateKind.APPLIED, completed.kind());
        var active = requestContext.recordEngineEvidence(new EngineEvidence(item, null, RoleType.PREFILL,
                EngineKind.ACTIVE, 0L, null, completedAt + 1));
        assertNotNull(active.after().deadlines().visibility(), "pending timer installation is real work even when status does not advance");
        var timer = SchedulerTestSupport.timer(registry.runtime);
        timer.update(requestContext, active.after().deadlines());
        timer.update(requestContext, completed.after().deadlines());
        assertNotNull(scheduledVisibility());
        var installed = scheduledVisibility();
        var repeated = requestContext.recordEngineEvidence(new EngineEvidence(item, null, RoleType.PREFILL,
                EngineKind.ACTIVE, 0L, null, completedAt + 2));
        timer.update(requestContext, repeated.after().deadlines());
        assertSame(installed, scheduledVisibility(), "installed handoff timer does not require a new continuation on every heartbeat");
    }

    @Test
    void singleStatusEntryRejectsForeignOwnerAndIsolatesMalformedStatus() throws Exception {
        acknowledgeDelivery();
        RequestContext foreign = mock(RequestContext.class);
        RequestData foreignData = mock(RequestData.class);
        when(foreign.data()).thenReturn(foreignData);
        when(foreignData.getScheduler()).thenReturn(mock(AbstractRequestScheduler.class));
        registry.onEngineEvidence(foreign, new EngineEvidence(item, item.decodeReservation(), RoleType.DECODE,
                EngineKind.ACTIVE, 0L, null, System.currentTimeMillis()));
        verify(foreign, never()).recordEngineEvidence(any());
        registry.onEngineEvidence(requestContext, null);
        registry.onEngineEvidence(requestContext, new EngineEvidence(item, item.decodeReservation(), null,
                EngineKind.ACTIVE, 0L, null, System.currentTimeMillis()));
        active(RoleType.DECODE, System.currentTimeMillis());
        assertTrue(requestContext.snapshot().engine().decodeAccepted(), "a malformed status must not prevent later valid activity");
    }

    private void active(RoleType source, long nowMs) {
        registry.onEngineEvidence(requestContext, new EngineEvidence(item,
                source == RoleType.DECODE ? item.decodeReservation() : null, source, EngineKind.ACTIVE, 0L, null, nowMs));
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
        RequestProtocolTestSupport.applyPrefillStatus(registry, prefill, RoleType.PREFILL, new PrefillState.PrefillRequestStatus(item, PrefillState.PrefillRequestStatus.Kind.ACTIVE, 0L));
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
        var before = requestContext.snapshot();
        var late = new TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis());
        var rejected = requestContext.recordDeliveryResult(late);
        assertEquals(UpdateKind.STALE, rejected.kind());
        assertSame(before, rejected.before());
        assertSame(before, rejected.after());
        assertEquals(SendState.UNKNOWN, rejected.after().delivery().send(), "late contradictory ACK cannot replace the settled sender fact");
        registry.onDeliveryEvidence(claim, late);
        assertSame(before, requestContext.snapshot());
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @Test
    void expiredSilenceRejectsAckWhileTheTimerWorkIsBacklogged() throws Exception {
        ExpirationTimer.Token exact = requestContext.snapshot().deadlines().inactivity();
        assertNotNull(exact);
        int workers = Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8);
        CountDownLatch started = new CountDownLatch(workers);
        CountDownLatch release = new CountDownLatch(1);
        for (int i = 0; i < workers; i++) {
            registry.runtime.executeRequestWork(requestContext, () -> {
                started.countDown();
                RequestProtocolTestSupport.await(release);
            });
        }
        try {
            assertTrue(started.await(1L, TimeUnit.SECONDS));
            long past = System.currentTimeMillis() - TIMEOUT_MS - 1_000L;
            org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "lastWorkerAtMs", past);
            var old = requestContext.snapshot().delivery();
            var aged = new DeliveryFacts(old.identity(), old.send(), old.acknowledged(), old.abandonment(),
                    old.executionFinished(), old.prefillProof(), old.decodeProof(), old.transferred(), past, old.acknowledgedAtMs());
            org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "delivery", aged);
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(requestContext, "refresh", System.currentTimeMillis());
            registry.onDeadline(requestContext.snapshot().deadlines().inactivity(), System.currentTimeMillis());
            completeDelivery(DeliveryResult.delivered());
            assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(requestContext.snapshot()).state());
            assertFalse(item.future().isDone(), "response publication follows the accepted cleanup");
        } finally {
            release.countDown();
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
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
        assertFalse(registry.runtime.signals().delivery(claim).toCompletableFuture().isDone());

        synchronized (requestContext) {
            assertEquals(CancelReason.CLIENT_CANCELLED, requestContext.snapshot().cancellation().reason());
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
        RequestRoute route = item;
        var reservation = item.decodeReservation();
        RoleType role = source == StaleFact.PREFILL_ENDPOINT || source == StaleFact.PREFILL_ITEM ? RoleType.PREFILL : RoleType.DECODE;
        if (source == StaleFact.PREFILL_ENDPOINT || source == StaleFact.PREFILL_ITEM || source == StaleFact.DECODE_ENDPOINT) {
            route = SchedulingTestConfig.createRoute(item.ctx(), SchedulingTestConfig.routeResponse(item), item.prefill(), null,
                    source == StaleFact.PREFILL_ENDPOINT ? mock(PrefillEndpoint.class) : prefill,
                    source == StaleFact.DECODE_ENDPOINT ? RequestProtocolTestSupport.decodeEndpoint() : decode,
                    reservation, registeredAtMs);
        } else {
            reservation = source == StaleFact.DECODE_GENERATION ? new DecodeResources.ReservationHandle(2L, REQUEST_ID, 1L)
                    : new DecodeResources.ReservationHandle(1L, REQUEST_ID, 2L);
        }
        var before = requestContext.snapshot();
        registry.onEngineEvidence(requestContext, new EngineEvidence(route, role == RoleType.DECODE ? reservation : null,
                role, EngineKind.ACTIVE, 0L, null, lateStatusAt));
        assertSame(before, requestContext.snapshot());
        assertTrue(inactive(lateStatusAt));
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, lateStatusAt);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void statusBeforeCancellationCheckInvalidatesTheEarlierExpirationDecision(RoleType source) {
        long originalDeadline = handedOffAtMs + TIMEOUT_MS;
        synchronized (requestContext) {
            assertTrue(inactive(originalDeadline), "the timer's earlier observation is expired");
        }

        recordWorkerActivity(source, originalDeadline - 1L);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, originalDeadline);

        assertLiveAndCharged();
        synchronized (requestContext) {
            assertFalse(inactive(originalDeadline));
            assertTrue(inactive(originalDeadline - 1L + TIMEOUT_MS));
        }
    }

    private void recordWorkerActivity(RoleType source, long nowMs) { active(source, nowMs); }

    private void assertLiveAndCharged() {
        assertEquals(1, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        synchronized (requestContext) {
            assertSame(item, requestContext.snapshot().route());
            assertFalse(RequestState.from(requestContext.snapshot()).state().isTerminal());
        }
        verify(decode, never()).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        verify(decode, never()).release(any(), eq(DecodeResources.ReleaseReason.REMOTE_CLEANUP));
        verify(prefill, never()).release(org.mockito.ArgumentMatchers.isNull(), any());
    }

    private void completeDelivery(DeliveryResult result) {
        registry.onDeliveryEvidence(claim, new TransportResult(claim, result, System.currentTimeMillis()));
        deliveryCompleted = true;
    }

    private void acknowledgeDelivery() throws Exception {
        completeDelivery(DeliveryResult.delivered());
        assertTrue(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    private void assertExpiredAndReleased(RequestState.Phase expectedState) {
        if (!registry.runtime.signals().delivery(claim).toCompletableFuture().isDone()) {
            assertEquals(1, registry.requests.liveRequestCount(), "expiry retains cleanup ownership");
            cleanup.complete(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            if (!deliveryCompleted) {
                assertFalse(registry.runtime.signals().delivery(claim).toCompletableFuture().isDone(), "early cleanup cannot release the sender");
                completeDelivery(DeliveryResult.uncertain(new IllegalStateException("sender exited")));
            }
            registry.runtime.signals().delivery(claim).toCompletableFuture().join();
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        assertEquals(expectedState, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(REQUEST_ID, 0L).state());
        synchronized (requestContext) {
            assertFalse(requestContext.snapshot().phase() != Phase.CLOSED);
        }
        verify(decode, times(1)).release(item.decodeReservation(), DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        verify(prefill, times(1)).release(null, item);
    }

    private boolean inactive(long nowMs) { return nowMs >= requestContext.snapshot().inactivityExpiresAtMs(); }

    private ScheduledFuture<?> scheduledVisibility() {
        var timer = SchedulerTestSupport.timer(registry.runtime);
        var slots = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.getField(timer, "slots");
        var slot = slots.get(requestContext);
        if (slot == null) { return null; }
        var handles = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils.invokeMethod(slot, "handles");
        var handle = handles.get(ExpirationTimer.Kind.VISIBILITY);
        return handle == null ? null : (ScheduledFuture<?>) org.springframework.test.util.ReflectionTestUtils.getField(handle, "scheduled");
    }

    private enum StaleFact {
        PREFILL_ENDPOINT,
        PREFILL_ITEM,
        DECODE_ENDPOINT,
        DECODE_GENERATION,
        DECODE_RESERVATION
    }
}
