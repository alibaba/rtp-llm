package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
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

    private RequestRegistry registry;
    private RequestSlot slot;
    private ScheduledRequest item;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private RequestRegistry.DeliveryClaim claim;
    private long registeredAtMs;

    @BeforeEach
    void setUp() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));

        BalanceContext context = RequestLifecycleTestSupport.context(config, REQUEST_ID);
        CompletableFuture<Response> future = registry.register(context);
        slot = registry.requestSlot(REQUEST_ID);
        registeredAtMs = slot.createdAtMs();
        prefill = mock(PrefillEndpoint.class);
        decode = mock(DecodeEndpoint.class);
        ServerStatus prefillStatus = new ServerStatus();
        prefillStatus.setRole(RoleType.PREFILL);
        prefillStatus.setServerIp("127.0.0.1");
        prefillStatus.setGrpcPort(8081);
        var reservation = new DecodeEndpoint.ReservationHandle(1L, REQUEST_ID, 1L);
        item = new ScheduledRequest(context, future, new Response(), prefillStatus, null,
                prefill, decode, reservation, registeredAtMs);
        var registered = new RequestLifecycleTestSupport.Registered(item, future);
        RequestLifecycleTestSupport.bindRoute(registry, registered);
        claim = RequestLifecycleTestSupport.claimRoute(registry, item, () -> true);
        assertNotNull(claim);
    }

    @AfterEach
    void tearDown() {
        if (registry != null && registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void repeatedStatusKeepsDeliveredRequestAliveBeyondItsOriginalDeadline(RoleType source) throws Exception {
        acknowledgeDelivery();
        for (int observation = 1; observation <= 6; observation++) {
            long observedAt = registeredAtMs + observation * (TIMEOUT_MS / 2L);
            observeActive(source, observedAt);
            registry.expireInactiveRequest(slot, observedAt + TIMEOUT_MS / 2L - 1L);
            assertLiveAndCharged();
        }
        assertTrue(item.future().isDone(), "the public route response does not end activity tracking");
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void silenceExpiresLocallyAtTheLastMatchingStatusDeadline(RoleType source) throws Exception {
        acknowledgeDelivery();
        long lastStatusAt = registeredAtMs + 2L * TIMEOUT_MS;
        observeActive(source, lastStatusAt);
        registry.expireInactiveRequest(slot, lastStatusAt + TIMEOUT_MS - 1L);
        assertLiveAndCharged();

        registry.expireInactiveRequest(slot, lastStatusAt + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);

        // A delayed status or timer callback cannot reopen or double-release this generation.
        registry.onPrefillFact(prefill, RoleType.PREFILL, PrefillState.WorkerStatusFact.active(item));
        registry.onDecodeFact(decode, DecodeEndpoint.WorkerStatusFact.active(item.decodeReservation()));
        registry.onDecodeFact(decode, DecodeEndpoint.WorkerStatusFact.terminal(item.decodeReservation(), 0L));
        registry.expireInactiveRequest(slot, lastStatusAt + 2L * TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @Test
    void missingDeliveryReplyExpiresPurelyLocally() throws Exception {
        registry.expireInactiveRequest(slot, registeredAtMs + TIMEOUT_MS - 1L);
        assertFalse(item.future().isDone(), "no transport callback has confirmed delivery");
        assertLiveAndCharged();

        registry.expireInactiveRequest(slot, registeredAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
        assertFalse(item.future().get(1L, TimeUnit.SECONDS).isSuccess());

        // A late delivery acknowledgement must not resurrect the expired request.
        registry.complete(claim, DeliveryResult.delivered());
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @ParameterizedTest
    @EnumSource(value = DeliveryResult.Status.class, names = {"UNCERTAIN", "TIMED_OUT"})
    void uncertainDeliveryKeepsOnlyABoundedConfirmationWait(DeliveryResult.Status outcome) throws Exception {
        registry.complete(claim, new DeliveryResult(outcome, new IllegalStateException("reply was lost")));
        assertLiveAndCharged();
        registry.expireInactiveRequest(slot, registeredAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
        assertFalse(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void anEarlierClientCancellationDoesNotDisableInactivityExpiration() throws Exception {
        acknowledgeDelivery();
        assertEquals(RequestState.Phase.CANCEL_REQUESTED,
                registry.cancelRequest(REQUEST_ID, 0L, CancelReason.CLIENT_CANCELLED).state());
        assertLiveAndCharged();

        synchronized (slot) {
            assertEquals(CancelReason.CLIENT_CANCELLED, slot.requireCancellationFirstCause());
        }
        registry.expireInactiveRequest(slot, registeredAtMs + TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.CANCELLED);
        registry.expireInactiveRequest(slot, registeredAtMs + 2L * TIMEOUT_MS);
        assertExpiredAndReleased(RequestState.Phase.CANCELLED);
    }

    @ParameterizedTest
    @EnumSource(StaleFact.class)
    void staleEndpointOrRequestGenerationCannotRenewInactivity(StaleFact source) {
        long lateStatusAt = registeredAtMs + 2L * TIMEOUT_MS;
        synchronized (slot) {
            RequestSlot.EngineObservation observation = switch (source) {
                case PREFILL_ENDPOINT -> slot.observePrefillFact(mock(PrefillEndpoint.class), RoleType.PREFILL,
                        PrefillState.WorkerStatusFact.active(item), lateStatusAt);
                case PREFILL_ITEM -> slot.observePrefillFact(prefill, RoleType.PREFILL,
                        PrefillState.WorkerStatusFact.active(new ScheduledRequest(item.ctx(), item.future(),
                                item.routeResponse(), item.prefill(), null, prefill, decode,
                                item.decodeReservation(), registeredAtMs)), lateStatusAt);
                case DECODE_ENDPOINT -> slot.observeDecodeFact(mock(DecodeEndpoint.class),
                        DecodeEndpoint.WorkerStatusFact.active(item.decodeReservation()), lateStatusAt);
                case DECODE_GENERATION -> slot.observeDecodeFact(decode, DecodeEndpoint.WorkerStatusFact.active(
                        new DecodeEndpoint.ReservationHandle(2L, REQUEST_ID, 1L)), lateStatusAt);
                case DECODE_RESERVATION -> slot.observeDecodeFact(decode, DecodeEndpoint.WorkerStatusFact.active(
                        new DecodeEndpoint.ReservationHandle(1L, REQUEST_ID, 2L)), lateStatusAt);
            };
            assertSame(RequestSlot.EngineObservation.STALE, observation);
            assertTrue(slot.requestInactive(lateStatusAt));
        }
        registry.expireInactiveRequest(slot, lateStatusAt);
        assertExpiredAndReleased(RequestState.Phase.TIMED_OUT);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void statusBeforeCancellationCheckInvalidatesTheEarlierExpirationDecision(RoleType source) {
        long originalDeadline = registeredAtMs + TIMEOUT_MS;
        synchronized (slot) {
            assertTrue(slot.requestInactive(originalDeadline), "the timer's earlier observation is expired");
        }

        observeActive(source, originalDeadline - 1L);
        registry.expireInactiveRequest(slot, originalDeadline);

        assertLiveAndCharged();
        synchronized (slot) {
            assertFalse(slot.requestInactive(originalDeadline));
            assertTrue(slot.requestInactive(originalDeadline - 1L + TIMEOUT_MS));
        }
    }

    private void observeActive(RoleType source, long nowMs) {
        synchronized (slot) {
            if (source == RoleType.PREFILL) {
                slot.observePrefillFact(prefill, RoleType.PREFILL,
                        PrefillState.WorkerStatusFact.active(item), nowMs);
            } else {
                slot.observeDecodeFact(decode,
                        DecodeEndpoint.WorkerStatusFact.active(item.decodeReservation()), nowMs);
            }
        }
    }

    private void assertLiveAndCharged() {
        assertEquals(1, registry.liveRequestCount());
        synchronized (slot) {
            assertSame(item, slot.activeItem());
            assertFalse(slot.snapshot().state().isTerminal());
        }
        verify(decode, never()).releaseReservationExact(any());
        verify(decode, never()).expireReservationExact(any());
        verify(prefill, never()).expireCommittedItem(any());
        verify(prefill, never()).releaseCommittedItem(any());
    }

    private void acknowledgeDelivery() throws Exception {
        registry.complete(claim, DeliveryResult.delivered());
        assertTrue(item.future().get(1L, TimeUnit.SECONDS).isSuccess());
    }

    private void assertExpiredAndReleased(RequestState.Phase expectedState) {
        assertEquals(0, registry.liveRequestCount());
        assertEquals(expectedState, registry.getRequestState(REQUEST_ID, 0L).state());
        synchronized (slot) {
            assertFalse(slot.isLiveGeneration());
        }
        verify(decode, times(1)).expireReservationExact(item.decodeReservation());
        verify(prefill, times(1)).expireCommittedItem(item);
    }

    private enum StaleFact {
        PREFILL_ENDPOINT,
        PREFILL_ITEM,
        DECODE_ENDPOINT,
        DECODE_GENERATION,
        DECODE_RESERVATION
    }
}
