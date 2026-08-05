package org.flexlb.balance.autotpm;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RunningPreemptPlanner} — pure planning, no mutation.
 */
class RunningPreemptPlannerTest {

    private static final String EP_KEY = "10.0.0.1:8080";
    private final RunningPreemptPlanner planner = new RunningPreemptPlanner();

    // ==================== Basic candidate finding ====================

    @Test
    void runningVictimFound_whenPriorityLower() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = res(10L, 30, 1000);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 10L);
        // Simulate running for 1 second — past critical section
        r.setRunningSinceMs(System.currentTimeMillis() - 1000);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        assertEquals(1, candidates.size());
        assertEquals(10L, candidates.get(0).requestId());
        assertEquals(30, candidates.get(0).priority());
    }

    @Test
    void samePriority_noVictim() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = res(10L, 50, 1000);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 10L);
        r.setRunningSinceMs(System.currentTimeMillis() - 1000);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 50, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty(), "Same-priority requests must never be preempted");
    }

    @Test
    void higherPriority_notPreempted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = res(10L, 70, 1000);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 10L);
        r.setRunningSinceMs(System.currentTimeMillis() - 1000);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 50, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty(), "Higher-priority requests must not be preempted");
    }

    // ==================== Critical section ====================

    @Test
    void criticalSection_notPreempted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = res(10L, 30, 1000);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 10L);
        // runningSinceMs is set to now by markRunning — within critical section (200ms)

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty(), "Request in critical section should not be preempted");
    }

    @Test
    void pastCriticalSection_preempted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = res(10L, 30, 1000);
        tracker.reserve(EP_KEY, r);
        tracker.markRunning(EP_KEY, 10L);
        // Set runningSinceMs to exactly criticalSectionMs ago (boundary)
        r.setRunningSinceMs(System.currentTimeMillis() - 200L);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        // At the boundary (>= criticalSectionMs), the request IS eligible
        assertFalse(candidates.isEmpty(), "Request past critical section should be preemptable");
    }

    // ==================== Sort order ====================

    @Test
    void multipleVictims_lowestPriorityFirst() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();

        DecodeReservation r1 = runningRes(10L, 40, 1000);
        DecodeReservation r2 = runningRes(11L, 30, 1000);
        tracker.reserve(EP_KEY, r1);
        tracker.reserve(EP_KEY, r2);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 5, 5000, 200L, 8);

        assertEquals(2, candidates.size());
        assertEquals(30, candidates.get(0).priority(), "P30 before P40");
        assertEquals(40, candidates.get(1).priority());
    }

    @Test
    void samePriority_moreKvReleasedFirst() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();

        DecodeReservation r1 = runningRes(10L, 30, 500);
        DecodeReservation r2 = runningRes(11L, 30, 2000);
        tracker.reserve(EP_KEY, r1);
        tracker.reserve(EP_KEY, r2);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 5, 5000, 200L, 8);

        assertEquals(2, candidates.size());
        assertEquals(2000, candidates.get(0).kvTokensRequired(),
                "More KV released should come first at same priority");
        assertEquals(500, candidates.get(1).kvTokensRequired());
    }

    // ==================== Edge cases ====================

    @Test
    void noRunningVictims_emptyResult() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // Only non-RUNNING reservations
        tracker.reserve(EP_KEY, res(10L, 30, 1000));
        tracker.reserve(EP_KEY, res(11L, 40, 1000));

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty(), "Non-RUNNING reservations should not be candidates");
    }

    @Test
    void emptyTracker_emptyResult() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty());
    }

    @Test
    void maxVictims_capsSelection() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        for (int i = 0; i < 5; i++) {
            tracker.reserve(EP_KEY, runningRes(100L + i, 30, 1000));
        }

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 5, 5000, 200L, 2);

        assertEquals(2, candidates.size(), "maxVictims=2 caps selection");
    }

    @Test
    void runningSinceZero_notPreempted() {
        // Edge case: reservation in RUNNING state but runningSinceMs=0
        // (never set via markRunning — should be protected by critical section)
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        DecodeReservation r = new DecodeReservation(
                10L, 30, 10_000_000L, 1000, EP_KEY, 1L,
                DecodeAdmissionState.RUNNING);
        // runningSinceMs defaults to 0 — don't set it
        tracker.reserve(EP_KEY, r);

        List<DecodeReservation> candidates = planner.findPreemptCandidates(
                tracker, EP_KEY, 70, 1, 100, 200L, 8);

        assertTrue(candidates.isEmpty(),
                "Reservations with runningSinceMs=0 should not be preempted (treated as critical section)");
    }

    // ==================== Helpers ====================

    /** Create a reservation in RESERVED_NOT_ACCEPTED state (for non-RUNNING tests). */
    private static DecodeReservation res(long requestId, int priority, long kvTokens) {
        return new DecodeReservation(requestId, priority, 10_000_000L,
                kvTokens, EP_KEY, requestId,
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED);
    }

    /** Create a reservation in RUNNING state with runningSinceMs set to 1 second ago. */
    private static DecodeReservation runningRes(long requestId, int priority, long kvTokens) {
        DecodeReservation r = new DecodeReservation(requestId, priority, 10_000_000L,
                kvTokens, EP_KEY, requestId,
                DecodeAdmissionState.RUNNING);
        r.setRunningSinceMs(System.currentTimeMillis() - 1000);
        return r;
    }
}
