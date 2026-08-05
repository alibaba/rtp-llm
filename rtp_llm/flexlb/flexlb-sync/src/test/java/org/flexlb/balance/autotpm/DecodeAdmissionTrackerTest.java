package org.flexlb.balance.autotpm;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link DecodeAdmissionTracker}.
 *
 * <p>Tests the core operations: reserve, state transitions, release,
 * and eviction candidate selection for both slot and KV shortages.
 */
class DecodeAdmissionTrackerTest {

    private static final String EP_KEY = "10.0.0.1:8080";

    // ==================== Basic operations ====================

    @Test
    void reserve_andRelease() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = reservation(1L, 50, 1000, "ep1");
        tracker.reserve("ep1", r);

        assertEquals(1, tracker.getReservations("ep1").size());
        assertSame(r, tracker.getReservation("ep1", 1L, true));

        tracker.release("ep1", 1L);
        assertTrue(tracker.getReservations("ep1").isEmpty());
    }

    @Test
    void release_nonExistent_isNoOp() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        assertNull(tracker.release("ep1", 999L));
    }

    // ==================== Capacity queries ====================

    @Test
    void availableSlots_noReservations() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        assertEquals(10, tracker.availableSlots(EP_KEY, 10));
    }

    @Test
    void availableSlots_withReservations() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));
        tracker.reserve(EP_KEY, reservation(2L, 60, 2000, EP_KEY));
        assertEquals(8, tracker.availableSlots(EP_KEY, 10));
    }

    @Test
    void availableKv_noReservations() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        assertEquals(100_000L, tracker.availableKv(EP_KEY, 100_000L));
    }

    @Test
    void availableKv_withReservations() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));
        tracker.reserve(EP_KEY, reservation(2L, 60, 2000, EP_KEY));
        assertEquals(97_000L, tracker.availableKv(EP_KEY, 100_000L));
    }

    @Test
    void availableKv_neverNegative() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 100_000, EP_KEY));
        assertEquals(0, tracker.availableKv(EP_KEY, 50_000L));
    }

    // ==================== State transitions ====================

    @Test
    void stateTransitions_reservedToAcceptedToRunning() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));

        assertEquals(DecodeAdmissionState.RESERVED_NOT_ACCEPTED,
                tracker.getReservation(EP_KEY, 1L, true).state());

        tracker.markAccepted(EP_KEY, 1L);
        assertEquals(DecodeAdmissionState.ACCEPTED_NOT_RUNNING,
                tracker.getReservation(EP_KEY, 1L, true).state());

        tracker.markRunning(EP_KEY, 1L);
        assertEquals(DecodeAdmissionState.RUNNING,
                tracker.getReservation(EP_KEY, 1L, true).state());
    }

    @Test
    void markRunning_fromReserved_directTransition() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));

        // Direct transition from RESERVED_NOT_ACCEPTED to RUNNING
        tracker.markRunning(EP_KEY, 1L);
        assertEquals(DecodeAdmissionState.RUNNING,
                tracker.getReservation(EP_KEY, 1L, true).state());
    }

    @Test
    void markAccepted_idempotent() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));

        tracker.markAccepted(EP_KEY, 1L);
        tracker.markAccepted(EP_KEY, 1L); // no-op
        assertEquals(DecodeAdmissionState.ACCEPTED_NOT_RUNNING,
                tracker.getReservation(EP_KEY, 1L, true).state());
    }

    @Test
    void markRunning_idempotent() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(1L, 50, 1000, EP_KEY));

        tracker.markRunning(EP_KEY, 1L);
        tracker.markRunning(EP_KEY, 1L); // no-op
        assertEquals(DecodeAdmissionState.RUNNING,
                tracker.getReservation(EP_KEY, 1L, true).state());
    }

    @Test
    void markAccepted_onNonExistent_isNoOp() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.markAccepted(EP_KEY, 999L); // should not throw
    }

    // ==================== Slot eviction candidates ====================

    @Test
    void slotEviction_lowerPriorityEvictedFirst() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(10L, 30, 1000, EP_KEY));
        tracker.reserve(EP_KEY, reservation(11L, 40, 1000, EP_KEY));
        tracker.reserve(EP_KEY, reservation(12L, 50, 1000, EP_KEY));

        // Incoming P70 → eligible victims: P30, P40, P50 (all lower)
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                EP_KEY, 70, 2);

        assertEquals(2, candidates.size());
        // P30 (lowest) first, then P40
        assertEquals(30, candidates.get(0).priority());
        assertEquals(40, candidates.get(1).priority());
    }

    @Test
    void slotEviction_samePriorityNotEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(10L, 50, 1000, EP_KEY));
        tracker.reserve(EP_KEY, reservation(11L, 50, 1000, EP_KEY));

        // Incoming P50 → no eligible victims (same priority)
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                EP_KEY, 50, 2);
        assertTrue(candidates.isEmpty());
    }

    @Test
    void slotEviction_higherPriorityNotEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(10L, 70, 1000, EP_KEY));

        // Incoming P50 → P70 is higher priority, NOT eligible
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                EP_KEY, 50, 1);
        assertTrue(candidates.isEmpty());
    }

    @Test
    void slotEviction_runningNeverEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r1 = reservation(10L, 30, 1000, EP_KEY);
        tracker.reserve(EP_KEY, r1);
        tracker.markRunning(EP_KEY, 10L); // transition to RUNNING

        tracker.reserve(EP_KEY, reservation(11L, 40, 1000, EP_KEY));

        // Incoming P70 → P30 is RUNNING (not evictable), only P40 eligible
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                EP_KEY, 70, 2);
        assertEquals(1, candidates.size());
        assertEquals(40, candidates.get(0).priority());
    }

    @Test
    void slotEviction_earlierStagePreferred() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // Two P30 reservations: one RESERVED, one ACCEPTED
        DecodeReservation reserved = new DecodeReservation(
                10L, 30, 10_000_000L, 1000, EP_KEY, 1L,
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED);
        DecodeReservation accepted = new DecodeReservation(
                11L, 30, 10_000_000L, 1000, EP_KEY, 2L,
                DecodeAdmissionState.ACCEPTED_NOT_RUNNING);
        tracker.reserve(EP_KEY, reserved);
        tracker.reserve(EP_KEY, accepted);

        // Incoming P70 → both P30 eligible, earlier stage (RESERVED) first
        List<DecodeReservation> candidates = tracker.findSlotEvictionCandidates(
                EP_KEY, 70, 2);
        assertEquals(2, candidates.size());
        assertEquals(DecodeAdmissionState.RESERVED_NOT_ACCEPTED,
                candidates.get(0).state());
        assertEquals(DecodeAdmissionState.ACCEPTED_NOT_RUNNING,
                candidates.get(1).state());
    }

    // ==================== KV eviction candidates ====================

    @Test
    void kvEviction_moreKvReleasedPreferred() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // Two P30 victims: one with 500 KV, one with 2000 KV
        tracker.reserve(EP_KEY, reservation(20L, 30, 500, EP_KEY));
        tracker.reserve(EP_KEY, reservation(21L, 30, 2000, EP_KEY));

        // Incoming P70, needs 1000 KV → greedy should pick the 2000-KV victim first
        List<DecodeReservation> candidates = tracker.findKvEvictionCandidates(
                EP_KEY, 70, 1000);
        assertEquals(1, candidates.size(), "Only one 2000-KV victim needed to satisfy 1000 KV");
        assertEquals(2000, candidates.get(0).kvTokensRequired());
    }

    @Test
    void kvEviction_greedyAccumulation() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // Three P30 victims: 300, 500, 800 KV
        tracker.reserve(EP_KEY, reservation(30L, 30, 300, EP_KEY));
        tracker.reserve(EP_KEY, reservation(31L, 30, 500, EP_KEY));
        tracker.reserve(EP_KEY, reservation(32L, 30, 800, EP_KEY));

        // Incoming P70, needs 1000 KV → greedy: 800 (not enough), then 500 (total 1300, enough)
        List<DecodeReservation> candidates = tracker.findKvEvictionCandidates(
                EP_KEY, 70, 1000);
        assertEquals(2, candidates.size(), "800+500=1300 >= 1000");
        assertEquals(800, candidates.get(0).kvTokensRequired());
        assertEquals(500, candidates.get(1).kvTokensRequired());
    }

    @Test
    void kvEviction_samePriorityNotEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(40L, 50, 1000, EP_KEY));

        List<DecodeReservation> candidates = tracker.findKvEvictionCandidates(
                EP_KEY, 50, 100);
        assertTrue(candidates.isEmpty());
    }

    @Test
    void kvEviction_runningNeverEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = reservation(50L, 30, 10000, EP_KEY);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 50L); // RUNNING

        tracker.reserve(EP_KEY, reservation(51L, 40, 100, EP_KEY));

        List<DecodeReservation> candidates = tracker.findKvEvictionCandidates(
                EP_KEY, 70, 1);
        assertEquals(1, candidates.size());
        assertEquals(40, candidates.get(0).priority());
    }

    // ==================== removeIfEvictable ====================

    @Test
    void removeIfEvictable_evictableReservation() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(60L, 30, 1000, EP_KEY));

        DecodeReservation removed = tracker.removeIfEvictable(EP_KEY, 60L);
        assertNotNull(removed);
        assertEquals(60L, removed.requestId());
        assertTrue(tracker.getReservations(EP_KEY).isEmpty());
    }

    @Test
    void removeIfEvictable_runningReservation_notRemoved() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, reservation(61L, 30, 1000, EP_KEY));
        tracker.markRunning(EP_KEY, 61L);

        DecodeReservation removed = tracker.removeIfEvictable(EP_KEY, 61L);
        assertNull(removed, "RUNNING reservations must not be evictable");
        assertEquals(1, tracker.getReservations(EP_KEY).size());
    }

    @Test
    void removeIfEvictable_nonExistent() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        assertNull(tracker.removeIfEvictable(EP_KEY, 999L));
    }

    // ==================== Helpers ====================

    private static DecodeReservation reservation(long requestId, int priority,
                                                 long kvTokensRequired, String epKey) {
        return new DecodeReservation(requestId, priority, 10_000_000L,
                kvTokensRequired, epKey, requestId,
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED);
    }
}
