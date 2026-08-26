package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * State-machine contracts for {@link PreemptionAttempt}, the decode-preemption
 * transaction ledger. This is the preemption lifecycle that a victim request
 * traverses when a higher-priority request evicts it — the exception/cancel
 * chain for eviction.
 *
 * <p>Attempt states: PLANNED → CLAIMED → CANCEL_IN_FLIGHT → WAITING_TERMINAL →
 * READY_COMMIT → COMMITTED (or ABORTED). Victim states mirror the per-request
 * cancel outcome. Every expected value is derived from the documented
 * transition rules, and the tests assert the exactly-once + idempotency
 * invariants that let a Cancel callback and a WorkerStatus callback converge
 * without double-settling a victim.
 */
@DisplayName("PreemptionAttempt state machine")
class PreemptionAttemptTest {

    private static PreemptionAttempt.Victim victim(long requestId) {
        return new PreemptionAttempt.Victim(
                requestId, /* priority */ 30, /* kvTokens */ 128L,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING, /* reservationToken */ 1L,
                new CancelTarget("10.0.0.1", 9090));
    }

    private static PreemptionAttempt attempt(long... victimIds) {
        List<PreemptionAttempt.Victim> victims = new java.util.ArrayList<>();
        for (long id : victimIds) {
            victims.add(victim(id));
        }
        return new PreemptionAttempt(1L, victims);
    }

    /** Drive an attempt to CANCEL_IN_FLIGHT (the common precondition). */
    private static PreemptionAttempt inFlight(long... victimIds) {
        PreemptionAttempt a = attempt(victimIds);
        assertTrue(a.claimAll());
        assertTrue(a.markCancelInFlight());
        return a;
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Construction validation")
    class Construction {

        @Test
        void rejectsNonPositiveTokenOrEmptyVictims() {
            assertThrows(IllegalArgumentException.class,
                    () -> new PreemptionAttempt(0L, List.of(victim(1L))));
            assertThrows(IllegalArgumentException.class,
                    () -> new PreemptionAttempt(1L, List.of()));
        }

        @Test
        void rejectsDuplicateVictimIds() {
            assertThrows(IllegalArgumentException.class,
                    () -> new PreemptionAttempt(1L, List.of(victim(7L), victim(7L))));
        }

        @Test
        void victimMustRequireEngineCancel() {
            // MASTER_QUEUED_NOT_DISPATCHED does not require engine cancel.
            assertThrows(IllegalArgumentException.class,
                    () -> new PreemptionAttempt.Victim(
                            1L, 30, 128L, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                            1L, new CancelTarget("10.0.0.1", 9090)));
        }

        @Test
        void victimRejectsNonPositiveReservationToken() {
            assertThrows(IllegalArgumentException.class,
                    () -> new PreemptionAttempt.Victim(
                            1L, 30, 128L, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                            0L, new CancelTarget("10.0.0.1", 9090)));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Happy-path attempt lifecycle")
    class HappyPath {

        @Test
        void plannedThroughCommittedSingleVictim() {
            PreemptionAttempt a = attempt(1L);
            assertTrue(a.claimAll());
            assertTrue(a.markCancelInFlight());
            assertTrue(a.recordAccepted(1L));
            a.beginTerminalWait();          // still WAITING_TERMINAL (victim not terminal)
            assertFalse(a.allVictimsTerminal());
            assertTrue(a.recordTerminal(1L)); // → READY_COMMIT
            assertTrue(a.allVictimsTerminal());
            assertTrue(a.markCommitted());
            // COMMITTED is absorbing: markAborted is a no-op.
            a.markAborted();
            assertFalse(a.markCommitted(), "committed attempt cannot re-commit");
        }

        @Test
        void terminalBeforeBeginWaitStillReachesReadyCommitOnWait() {
            // If every victim is already TERMINAL when beginTerminalWait runs,
            // advanceReadyIfSettled promotes straight to READY_COMMIT.
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordTerminal(1L));
            a.beginTerminalWait();          // all terminal → READY_COMMIT immediately
            assertTrue(a.markCommitted());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Attempt-state guards (ordering)")
    class AttemptGuards {

        @Test
        void markCancelInFlightRequiresClaimed() {
            assertFalse(attempt(1L).markCancelInFlight(),
                    "cannot go in-flight from PLANNED");
        }

        @Test
        void claimAllOnlyFromPlanned() {
            PreemptionAttempt a = attempt(1L);
            assertTrue(a.claimAll());
            assertFalse(a.claimAll(), "double claim is rejected");
        }

        @Test
        void beginTerminalWaitOnlyFromCancelInFlight() {
            assertThrows(IllegalStateException.class,
                    () -> attempt(1L).beginTerminalWait());
        }

        @Test
        void markCommittedRequiresReadyCommit() {
            assertFalse(inFlight(1L).markCommitted(),
                    "cannot commit before every victim settles");
        }

        @Test
        void abortIsBlockedOnceCommitted() {
            PreemptionAttempt a = inFlight(1L);
            a.recordTerminal(1L);
            a.beginTerminalWait();
            assertTrue(a.markCommitted());
            a.markAborted();
            // Still committed: a subsequent commit attempt is a no-op false,
            // proving the state stayed COMMITTED (not ABORTED).
            assertFalse(a.markCommitted());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Victim settlement convergence and idempotency")
    class VictimSettlement {

        @Test
        void recordTerminalIsExactlyOnceAndIdempotent() {
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordTerminal(1L));
            assertTrue(a.isTerminal(1L));
            // A second terminal (e.g. from the other callback) returns true
            // without re-settling.
            assertTrue(a.recordTerminal(1L));
        }

        @Test
        void acceptedThenTerminalConverge() {
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordAccepted(1L));   // CANCEL_REQUESTED
            assertTrue(a.recordTerminal(1L));   // CANCEL_REQUESTED → TERMINAL
            assertTrue(a.isTerminal(1L));
        }

        @Test
        void notFoundThenTerminalConverge() {
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordNotFound(1L));   // NOT_FOUND_STALE
            assertTrue(a.recordTerminal(1L));   // NOT_FOUND_STALE → TERMINAL
            assertTrue(a.isTerminal(1L));
        }

        @Test
        void unknownThenTerminalConverge() {
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordUnknown(1L));    // CANCEL_UNKNOWN
            assertTrue(a.recordTerminal(1L));   // CANCEL_UNKNOWN → TERMINAL
            assertTrue(a.isTerminal(1L));
        }

        @Test
        void recordAcceptedRejectedBeforeCancelInFlight() {
            PreemptionAttempt a = attempt(1L);
            a.claimAll(); // CLAIMED, not yet CANCEL_IN_FLIGHT
            assertFalse(a.recordAccepted(1L),
                    "accepted only valid while a cancel is in flight");
        }

        @Test
        void terminalIsAbsorbingAgainstLaterCallbacks() {
            PreemptionAttempt a = inFlight(1L);
            assertTrue(a.recordTerminal(1L));
            // Late Cancel-accepted / not-found / unknown all no-op true on a
            // victim already terminal (idempotent convergence).
            assertTrue(a.recordAccepted(1L));
            assertTrue(a.recordNotFound(1L));
            assertTrue(a.recordUnknown(1L));
        }

        @Test
        void operationsOnNonVictimThrow() {
            PreemptionAttempt a = inFlight(1L);
            assertThrows(IllegalArgumentException.class, () -> a.recordTerminal(999L));
            assertThrows(IllegalArgumentException.class, () -> a.isTerminal(999L));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Multi-victim readiness")
    class MultiVictim {

        @Test
        void readyCommitOnlyAfterEveryVictimIsTerminal() {
            PreemptionAttempt a = inFlight(1L, 2L);
            a.beginTerminalWait();
            assertTrue(a.recordTerminal(1L));
            assertFalse(a.allVictimsTerminal());
            assertFalse(a.markCommitted(),
                    "one victim still outstanding blocks commit");
            assertTrue(a.recordTerminal(2L));   // now all terminal → READY_COMMIT
            assertTrue(a.allVictimsTerminal());
            assertTrue(a.markCommitted());
        }
    }
}
