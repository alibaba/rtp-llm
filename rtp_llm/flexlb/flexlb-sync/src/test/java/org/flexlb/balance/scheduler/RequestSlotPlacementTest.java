package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodePlacementAuthorityPort;
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

    // ==================== stage-2 T7 S3: decode admission authority ====================

    @Test
    void authorityPreloadRowLivesOnlyInThePreBindWindow() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 301L, 11L);
        Registered registered =
                registerItem(301L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             301L, registered.future())) {
            assertNotNull(admission);
            // Reserve-wrapper stage: the authority carries the fence
            // plus the preloaded numeric row before any publication
            // bind (the admission body commits nothing here).
            lifecycle.executeUnderDecodeAdmission(
                    301L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 11L, 16L, 24L, 50, false),
                    () -> null,
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(11L, false, 0L, false));

            RequestSlot slot = lifecycle.requestSlot(301L);
            synchronized (slot) {
                RequestSlot.SlotDecodeAdmission authority =
                        slot.decodeAdmissionAuthorityView();
                assertNotNull(authority);
                assertEquals(16L, authority.preloadedKvTokens());
                assertEquals(24L, authority.preloadedExpectedKvTokens());
                assertEquals(50, authority.preloadedPriority());
                assertFalse(authority.masterQueued());
            }

            // Publication bind: the real pRow takes over as the
            // numeric carrier — the preload row must clear inside the
            // same tick (ruling 2(a) lifecycle point one).
            assertTrue(lifecycle.commitInflight(
                    registered.item(), false, () -> true));
            synchronized (slot) {
                RequestSlot.SlotDecodeAdmission authority =
                        slot.decodeAdmissionAuthorityView();
                assertNotNull(authority);
                assertEquals(0L, authority.preloadedKvTokens());
                assertEquals(0L, authority.preloadedExpectedKvTokens());
                assertEquals(0, authority.preloadedPriority());
                assertNotNull(slot.prefillRow());
            }
        }
    }

    @Test
    void publicationRollbackRestoresThePreloadedRow() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 311L, 12L);
        Registered registered =
                registerItem(311L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             311L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    311L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 12L, 16L, 24L, 50, false),
                    () -> null,
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(12L, false, 0L, false));
            RequestSlot slot = lifecycle.requestSlot(311L);
            synchronized (slot) {
                assertTrue(slot.tryBindItemForPublication(
                        registered.item(), false));
                assertEquals(0L,
                        slot.decodeAdmissionAuthorityView()
                                .preloadedKvTokens());
                // Publication did not commit: roll the bind back — the
                // preloaded numeric row regains charge, re-derived from
                // the exact item (ruling 2(a) lifecycle point two).
                slot.rollbackItemPublication(registered.item());
                RequestSlot.SlotDecodeAdmission authority =
                        slot.decodeAdmissionAuthorityView();
                assertNotNull(authority);
                assertEquals(16L, authority.preloadedKvTokens());
                assertEquals(0L, authority.preloadedExpectedKvTokens());
                assertEquals(50, authority.preloadedPriority());
                assertNull(slot.activeItem());
            }
        }
    }

    @Test
    void terminalizingClearsTheAuthorityUnconditionally() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 321L, 13L);
        Registered registered =
                registerItem(321L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             321L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    321L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 13L, 16L, 24L, 50, false),
                    () -> lifecycle.commitInflight(
                            registered.item(), false, () -> true),
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(13L, false, 0L, false));
            RequestSlot slot = lifecycle.requestSlot(321L);
            synchronized (slot) {
                assertNotNull(slot.decodeAdmissionAuthorityView());
                assertTrue(slot.markDecodeAccepted()
                        .acceptedBeforeCancel());
                // The decode-accepted death path force-clears the
                // authority inside its own monitor tick (ruling 2(a)
                // lifecycle point three) — a stale authority can never
                // outlive its slot.
                assertNull(slot.decodeAdmissionAuthorityView());
            }
        }
    }

    @Test
    void authorityClearIsFenceGuarded() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 331L, 14L);
        Registered registered =
                registerItem(331L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             331L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    331L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 14L, 16L, 24L, 50, false),
                    () -> null,
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(14L, false, 0L, false));
            // A foreign fence (a stale clear delivery after
            // request-id reuse) must not remove the newer authority.
            lifecycle.clearDecodeAdmission(331L, decode, 7L, 999L);
            RequestSlot slot = lifecycle.requestSlot(331L);
            synchronized (slot) {
                assertNotNull(slot.decodeAdmissionAuthorityView());
            }
            // The exact fence clears idempotently.
            lifecycle.clearDecodeAdmission(331L, decode, 7L, 14L);
            synchronized (slot) {
                assertNull(slot.decodeAdmissionAuthorityView());
            }
            lifecycle.clearDecodeAdmission(331L, decode, 7L, 14L);
            synchronized (slot) {
                assertNull(slot.decodeAdmissionAuthorityView());
            }
        }
    }

    // ==================== stage-2 T7 S2: channel-B authority view ====================

    @Test
    void channelBViewReturnsTheExactFenceSubState() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint other = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 341L, 15L);
        Registered registered =
                registerItem(341L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             341L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    341L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 15L, 16L, 24L, 50, true),
                    () -> null,
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(15L, true, 0L, false));

            // The exact fence resolves to the authority sub-state
            // snapshot (the read-source switch's channel B).
            DecodePlacementAuthorityPort.DecodeAdmissionEntry view =
                    lifecycle.decodeAdmissionView(341L, decode, 7L, 15L);
            assertNotNull(view);
            assertEquals(15L, view.reservationToken());
            assertTrue(view.masterQueued());
            assertEquals(0L, view.dispatchPermitToken());
            assertFalse(view.engineLifecycleOwned());

            // Every fence mismatch — wrong token, wrong generation,
            // wrong endpoint — is "no authority fact" (null).
            assertNull(lifecycle.decodeAdmissionView(
                    341L, decode, 7L, 999L));
            assertNull(lifecycle.decodeAdmissionView(
                    341L, decode, 8L, 15L));
            assertNull(lifecycle.decodeAdmissionView(
                    341L, other, 7L, 15L));
            // A request without a slot is likewise no authority fact.
            assertNull(lifecycle.decodeAdmissionView(
                    999L, decode, 7L, 15L));
        }
    }

    @Test
    void channelBViewFollowsAuthorityFlips() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 351L, 16L);
        Registered registered =
                registerItem(351L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             351L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    351L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 16L, 16L, 24L, 50, true),
                    () -> null,
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(16L, true, 0L, false));

            // The permit flip: queued → permit held (the post-commit
            // delivery class of the dispatch-permit acquisition).
            lifecycle.deliverDecodeAdmissionAfterCommit(
                    351L,
                    DecodePlacementAuthorityPort.Projection.flip(
                            decode, 7L, 16L, false, 5L, false));
            DecodePlacementAuthorityPort.DecodeAdmissionEntry view =
                    lifecycle.decodeAdmissionView(351L, decode, 7L, 16L);
            assertNotNull(view);
            assertFalse(view.masterQueued());
            assertEquals(5L, view.dispatchPermitToken());

            // The engine-lifecycle transfer: the queued bit stays off
            // and the lifecycle-owned bit turns on.
            lifecycle.deliverDecodeAdmissionAfterCommit(
                    351L,
                    DecodePlacementAuthorityPort.Projection.flip(
                            decode, 7L, 16L, false, 0L, true));
            view = lifecycle.decodeAdmissionView(351L, decode, 7L, 16L);
            assertNotNull(view);
            assertFalse(view.masterQueued());
            assertTrue(view.engineLifecycleOwned());
        }
    }

    @Test
    void channelBViewEndsWithTheSlotLifecycle() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(7L, 361L, 17L);
        Registered registered =
                registerItem(361L, null, decode, reservation);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             361L, registered.future())) {
            assertNotNull(admission);
            lifecycle.executeUnderDecodeAdmission(
                    361L,
                    DecodePlacementAuthorityPort.Projection.install(
                            decode, 7L, 17L, 16L, 24L, 50, true),
                    () -> lifecycle.commitInflight(
                            registered.item(), false, () -> true),
                    () -> new DecodePlacementAuthorityPort
                            .DecodeAdmissionEntry(17L, true, 0L, false));
            assertNotNull(lifecycle.decodeAdmissionView(
                    361L, decode, 7L, 17L));

            // The decode-accepted death path terminalizes the slot —
            // the channel-B view ends with the slot lifecycle (an
            // absent ACTIVE slot is "no authority fact").
            RequestSlot slot = lifecycle.requestSlot(361L);
            synchronized (slot) {
                assertTrue(slot.markDecodeAccepted()
                        .acceptedBeforeCancel());
            }
            assertNull(lifecycle.decodeAdmissionView(
                    361L, decode, 7L, 17L));
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
