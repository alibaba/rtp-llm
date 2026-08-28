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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Executable documentation for the slot-side pRow/dRow resource ledger
 * (plan 3.1 item 2, v2 C1/C8 two-road accounting).
 *
 * <p>The pRow mirrors the A-road master reservation from admission until
 * the KV_ALLOCATED critical point; the dRow mirrors the B-road engine
 * projection afterwards. The handover happens inside the exact
 * markDecodeAccepted slot mutation, so no intermediate snapshot can show
 * both rows or neither.</p>
 */
class RequestSlotResourceRowTest {

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
    void prefillRowMirrorsTheAdmissionReservation() {
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(5L, 301L, 77L);
        Registered registered = registerItem(
                301L, 16L, 24L, 3, mock(PrefillEndpoint.class), reservation);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(301L);
        synchronized (slot) {
            RequestSlot.SlotResourceRow row = slot.prefillRow();
            assertNotNull(row);
            assertEquals(
                    RequestSlot.SlotResourceRow.RowAuthority.MASTER_RESERVATION,
                    row.authority());
            assertEquals(16L, row.hardKvTokens());
            assertEquals(24L, row.expectedKvTokens());
            assertEquals(3, row.priority());
            assertEquals(77L, row.reservationToken());
            assertTrue(row.installedAtMs() > 0L);
            assertNull(slot.decodeRow());
            assertFalse(slot.decodeOwnsRequest());
        }
    }

    @Test
    void prefillRowIsClearedWhenPublicationDeclines() {
        Registered registered = registerItem(
                311L, 8L, 12L, 0, null, null);
        RequestSlot slot = lifecycle.requestSlot(311L);

        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(311L, registered.future())) {
            assertNotNull(admission);
            assertFalse(lifecycle.commitInflight(
                    registered.item(), false, () -> false));
            synchronized (slot) {
                assertNull(slot.prefillRow());
                assertNull(slot.decodeRow());
            }
        }
    }

    @Test
    void criticalPointHandoverSwapsAuthorityRowsAtomically() {
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(9L, 321L, 31L);
        Registered registered = registerItem(
                321L, 20L, 30L, 2, mock(PrefillEndpoint.class), reservation);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(321L);
        synchronized (slot) {
            assertNotNull(slot.prefillRow());
            assertNull(slot.decodeRow());
        }
        // The critical point (KV_ALLOCATED first report) arrives as one
        // exact slot mutation: the accepted fact flips the authority rows
        // and the ownership flag together.
        synchronized (slot) {
            DecodeAcceptance acceptance = slot.markDecodeAccepted();
            assertTrue(acceptance.acceptedBeforeCancel());
        }
        synchronized (slot) {
            assertNull(slot.prefillRow(),
                    "the master reservation row must end at the handover");
            RequestSlot.SlotResourceRow row = slot.decodeRow();
            assertNotNull(row,
                    "the engine projection row must begin at the handover");
            assertEquals(
                    RequestSlot.SlotResourceRow.RowAuthority.ENGINE_PROJECTION,
                    row.authority());
            assertEquals(31L, row.reservationToken());
            assertEquals(2, row.priority());
            assertEquals(0L, row.hardKvTokens(),
                    "numeric authority stays with the engine projection");
            assertTrue(row.installedAtMs() > 0L);
            assertTrue(slot.decodeOwnsRequest());
        }
    }

    @Test
    void repeatedAcceptanceKeepsTheFirstProjectionRow() {
        DecodeEndpoint.ReservationHandle reservation =
                new DecodeEndpoint.ReservationHandle(4L, 331L, 41L);
        Registered registered = registerItem(
                331L, 10L, 18L, 1, mock(PrefillEndpoint.class), reservation);
        assertTrue(bind(registered));

        RequestSlot slot = lifecycle.requestSlot(331L);
        synchronized (slot) {
            slot.markDecodeAccepted();
        }
        RequestSlot.SlotResourceRow first;
        synchronized (slot) {
            first = slot.decodeRow();
            assertNotNull(first);
        }
        synchronized (slot) {
            slot.markDecodeAccepted();
        }
        synchronized (slot) {
            assertEquals(first, slot.decodeRow(),
                    "repeated accepted facts keep the first handover row");
        }
    }

    @Test
    void rowsAreClearedWhenTheSlotReachesTombstone() throws Exception {
        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        Registered registered = registerItem(
                341L, 12L, 0L, 0, prefill, null);
        assertTrue(bind(registered));

        assertTrue(lifecycle.closeAdmissionAndAwaitMutations());
        lifecycle.closeOutstandingAndTerminalize();
        lifecycle.closeExpiration();
        lifecycle.closePublisher();

        RequestSlot slot = lifecycle.requestSlot(341L);
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
            assertNull(slot.prefillRow());
            assertNull(slot.decodeRow());
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

    private Registered registerItem(
            long requestId,
            long seqLen,
            long decodeExpectedKvTokens,
            int priority,
            PrefillEndpoint prefillEndpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        BalanceContext context = context(requestId, seqLen, priority);
        CompletableFuture<Response> future =
                lifecycle.register(context, 8);
        BatchItem item = new BatchItem(
                context,
                future,
                new Response(),
                null,
                null,
                prefillEndpoint,
                reservation == null ? null : mock(DecodeEndpoint.class),
                reservation,
                decodeExpectedKvTokens,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    private BalanceContext context(
            long requestId,
            long seqLen,
            int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority,
                System.currentTimeMillis() + 60_000L));
        return context;
    }

    private record Registered(
            BatchItem item,
            CompletableFuture<Response> future) {
    }
}
