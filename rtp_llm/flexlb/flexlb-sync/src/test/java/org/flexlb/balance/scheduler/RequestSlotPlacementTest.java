package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Executable documentation for the explicit slot placement mirror.
 *
 * <p>The placement field freezes the bound item's send-once identity at
 * publication bind, is cleared on rollback and tombstone, and the exact
 * ownership predicates read it as a derived view with unchanged results
 * (behaviour-zero pure addition, plan 3.1 item 1).</p>
 */
class RequestSlotPlacementTest {

    private FlexlbConfig config;
    private RequestLifecycleCoordinator lifecycle;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestLifecycleCoordinator(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(EngineCancelChannel.class));
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
    void placementIsNullBeforeThePublicationBind() {
        Registered registered = registerItem(101L);
        RequestSlot slot = lifecycle.requestSlot(101L);
        assertNotNull(slot);

        synchronized (slot) {
            assertNull(slot.placement());
            assertNull(slot.activeItem());
        }
        assertFalse(lifecycle.commitInflight(
                registered.item(), false, () -> true));
        // Without an admission mutation the item never binds, so the
        // placement mirror must stay empty.
        synchronized (slot) {
            assertNull(slot.placement());
        }
    }

    @Test
    void placementFreezesTheBoundItemIdentity() {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 201L, 11L);
        Registered registered =
                registerItem(201L, prefill, decode, reservation);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(201L);
        synchronized (slot) {
            RequestSlot.SlotPlacement placement = slot.placement();
            assertNotNull(placement);
            assertSame(prefill, placement.prefillEndpoint());
            assertSame(decode, placement.decodeEndpoint());
            assertEquals(reservation, placement.decodeReservation());
            assertEquals(11L, placement.decodeReservationToken());
            assertSame(slot.activeItem(), registered.item());
        }
    }

    @Test
    void placementIsClearedWhenPublicationDeclines() {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(211L, prefill, null, null);
        RequestSlot slot = lifecycle.requestSlot(211L);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(211L, registered.future())) {
            assertNotNull(admission);
            assertFalse(lifecycle.commitInflight(
                    registered.item(), false, () -> false));
            synchronized (slot) {
                assertNull(slot.activeItem());
                assertNull(slot.placement());
            }
        }
    }

    @Test
    void placementIsClearedWhenTheSlotReachesTombstone() throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(221L, prefill, null, null);
        assertTrue(bind(registered));

        // Drive the bound slot to a terminal tombstone through the
        // coordinator's outstanding-close path (the same terminal leaves
        // the shutdown path uses); terminal publication is asynchronous.
        assertTrue(lifecycle.closeAdmissionAndAwaitMutations());
        lifecycle.closeOutstandingAndTerminalize();
        lifecycle.closeExpiration();
        lifecycle.closePublisher();

        RequestSlot slot = lifecycle.requestSlot(221L);
        long deadline = System.nanoTime()
                + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            synchronized (slot) {
                if (slot.isTombstone()) {
                    break;
                }
            }
            Thread.sleep(1L);
        }
        synchronized (slot) {
            assertTrue(slot.isTombstone());
            assertNull(slot.placement());
            assertNull(slot.activeItem());
        }
    }

    @Test
    void ownsPrefillFactReadsThePlacementMirror() {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        PrefillEndpoint other = mock(PrefillEndpoint.class);
        Registered registered = registerItem(231L, prefill, null, null);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(231L);
        synchronized (slot) {
            assertTrue(slot.ownsPrefillFact(prefill, registered.item()));
            assertFalse(slot.ownsPrefillFact(other, registered.item()));
            assertFalse(slot.ownsPrefillFact(null, registered.item()));
        }
    }

    @Test
    void ownsDecodeFactReadsThePlacementMirror() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint other = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(3L, 241L, 19L);
        DecodeEndpoint.ReservationHandle staleToken =
                new DecodeEndpoint.ReservationHandle(3L, 241L, 20L);
        Registered registered =
                registerItem(241L, null, decode, reservation);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(241L);
        synchronized (slot) {
            assertTrue(slot.ownsDecodeFact(decode, reservation));
            assertFalse(slot.ownsDecodeFact(other, reservation));
            assertFalse(slot.ownsDecodeFact(decode, staleToken));
        }
    }

    private boolean bind(Registered registered) {
        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            return lifecycle.commitInflight(
                    registered.item(), false, () -> true);
        }
    }

    private Registered registerItem(long requestId) {
        return registerItem(requestId, null, null, null);
    }

    private Registered registerItem(
            long requestId,
            PrefillEndpoint prefillEndpoint,
            DecodeEndpoint decodeEndpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        BalanceContext context = context(requestId);
        CompletableFuture<Response> future =
                lifecycle.register(context, 8);
        BatchItem item = new BatchItem(
                context,
                future,
                new Response(),
                null,
                null,
                prefillEndpoint,
                decodeEndpoint,
                reservation,
                0L,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50,
                System.currentTimeMillis() + 60_000L));
        return context;
    }

    private record Registered(
            BatchItem item,
            CompletableFuture<Response> future) {
    }
}
