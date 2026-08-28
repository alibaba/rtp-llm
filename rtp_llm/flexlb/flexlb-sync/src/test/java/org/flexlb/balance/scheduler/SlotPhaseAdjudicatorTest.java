package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.SlotPhaseAdjudicator.CoarsePhase;
import org.flexlb.balance.scheduler.SlotPhaseAdjudicator.Event;
import org.flexlb.balance.scheduler.SlotPhaseAdjudicator.RefinedPhase;
import org.flexlb.balance.scheduler.SlotPhaseAdjudicator.State;
import org.flexlb.balance.scheduler.SlotPhaseAdjudicator.Verdict;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the partial-order adjudication layer
 * (plan 3.1 item 3, v2 S4 / 2.2 ruling matrix, invariants I1-I3).
 */
class SlotPhaseAdjudicatorTest {

    private static final long GENERATION = 7L;

    @Test
    void staleVersionEventsAreDiscardedByLastWriteWins() {
        State current = new State(GENERATION, 10L, RefinedPhase.P_RUNNING);

        Verdict older = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 9L, RefinedPhase.P_WAITING_UNLOADED, false));
        assertTrue(older instanceof Verdict.DiscardStaleVersion);
        assertEquals(9L, ((Verdict.DiscardStaleVersion) older).eventVersion());
        assertEquals(10L, ((Verdict.DiscardStaleVersion) older).seenVersion());
        assertFalse(older.advancesState());

        // Same-tick duplicate intermediates are LWW discards too (dedup bucket).
        Verdict duplicate = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 10L, RefinedPhase.P_RUNNING, false));
        assertTrue(duplicate instanceof Verdict.DiscardStaleVersion);
    }

    @Test
    void lateFinishBelowSeenVersionIsAlsoLastWriteWinsDiscard() {
        State current = new State(GENERATION, 10L, RefinedPhase.D_RUNNING);

        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 9L, RefinedPhase.COMPLETED, true));
        assertTrue(ruling instanceof Verdict.DiscardStaleVersion);
    }

    @Test
    void lateIntermediateEventsAreDroppedMonotonically() {
        State current = new State(GENERATION, 10L, RefinedPhase.P_RUNNING);

        // Equal phase at a fresh version: a repeat, not an advance.
        Verdict samePhase = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 11L, RefinedPhase.P_RUNNING, false));
        assertTrue(samePhase instanceof Verdict.DiscardLateEvent);
        assertEquals(RefinedPhase.P_RUNNING,
                ((Verdict.DiscardLateEvent) samePhase).currentPhase());

        // Strictly older phase at a fresh version: a late intermediate.
        Verdict olderPhase = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 11L, RefinedPhase.P_RECEIVED, false));
        assertTrue(olderPhase instanceof Verdict.DiscardLateEvent);
        assertEquals(RefinedPhase.P_RECEIVED,
                ((Verdict.DiscardLateEvent) olderPhase).indicatedPhase());
        assertFalse(olderPhase.advancesState());
    }

    @Test
    void implicationClosureJumpsMissingWaypoints() {
        // v2 appendix L9: a D event arriving while parked on DISPATCHED
        // jumps straight to D_LOADING without waiting for P events.
        State current = new State(GENERATION, 4L, RefinedPhase.DISPATCHED);
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 5L, RefinedPhase.D_LOADING, false));

        assertTrue(ruling instanceof Verdict.Advance);
        Verdict.Advance advance = (Verdict.Advance) ruling;
        assertEquals(RefinedPhase.DISPATCHED, advance.from());
        assertEquals(RefinedPhase.D_LOADING, advance.to());
        assertFalse(advance.finishWins());
        assertEquals(
                List.of(
                        RefinedPhase.P_RECEIVED,
                        RefinedPhase.P_WAITING_UNLOADED,
                        RefinedPhase.P_WAITING_LOADED,
                        RefinedPhase.P_RUNNING,
                        RefinedPhase.PREFILL_DONE),
                advance.skippedWaypoints());
        assertEquals(5L, advance.nextState().seenVersion());
        assertEquals(RefinedPhase.D_LOADING, advance.nextState().phase());
        assertEquals(GENERATION, advance.nextState().generationId());
        assertTrue(advance.advancesState());

        // I2: the closure of a D_* advance must contain PREFILL_DONE and
        // start strictly after DISPATCHED.
        assertTrue(advance.skippedWaypoints().contains(RefinedPhase.PREFILL_DONE));
        assertTrue(advance.skippedWaypoints().stream()
                .noneMatch(RefinedPhase::isTerminal));
    }

    @Test
    void adjacentAdvanceHasNoSkippedWaypoints() {
        State current = new State(GENERATION, 4L, RefinedPhase.QUEUED);
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 5L, RefinedPhase.DISPATCHING, false));

        assertTrue(ruling instanceof Verdict.Advance);
        Verdict.Advance advance = (Verdict.Advance) ruling;
        assertEquals(RefinedPhase.DISPATCHING, advance.to());
        assertEquals(List.of(), advance.skippedWaypoints());
    }

    @Test
    void initialPhaseAdvancesToTheFirstRoutedEvent() {
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                State.initial(GENERATION),
                new Event(GENERATION, 1L, RefinedPhase.ROUTED, false));
        assertTrue(ruling instanceof Verdict.Advance);
        assertEquals(RefinedPhase.ROUTED, ((Verdict.Advance) ruling).to());
    }

    @Test
    void finishAtAFreshVersionAdvancesWithoutTheSafetyNetFlag() {
        State current = new State(GENERATION, 10L, RefinedPhase.D_RUNNING);
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 11L, RefinedPhase.COMPLETED, true));

        assertTrue(ruling instanceof Verdict.Advance);
        Verdict.Advance advance = (Verdict.Advance) ruling;
        assertEquals(RefinedPhase.COMPLETED, advance.to());
        assertFalse(advance.finishWins());
        assertEquals(
                List.of(),
                advance.skippedWaypoints());
    }

    @Test
    void sameTickFinishOverridesAnAppliedIntermediate() {
        // Rule 4 (v2 S4): a finish colliding with an already-applied
        // intermediate of the same version wins — WARN-worthy safety net.
        State current = new State(GENERATION, 10L, RefinedPhase.P_RUNNING);
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 10L, RefinedPhase.CANCELLED, true));

        assertTrue(ruling instanceof Verdict.Advance);
        Verdict.Advance advance = (Verdict.Advance) ruling;
        assertEquals(RefinedPhase.CANCELLED, advance.to());
        assertTrue(advance.finishWins());
        assertEquals(10L, advance.nextState().seenVersion());
    }

    @Test
    void finishJumpingFromAnEarlyPhaseSkipsEveryRemainingIntermediate() {
        State current = new State(GENERATION, 3L, RefinedPhase.P_WAITING_UNLOADED);
        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION, 4L, RefinedPhase.FAILED, true));

        assertTrue(ruling instanceof Verdict.Advance);
        Verdict.Advance advance = (Verdict.Advance) ruling;
        assertEquals(RefinedPhase.FAILED, advance.to());
        assertEquals(
                List.of(
                        RefinedPhase.P_WAITING_LOADED,
                        RefinedPhase.P_RUNNING,
                        RefinedPhase.PREFILL_DONE,
                        RefinedPhase.D_LOADING,
                        RefinedPhase.D_RUNNING),
                advance.skippedWaypoints());
    }

    @Test
    void terminalStatesAbsorbEveryLaterEvent() {
        State terminal = new State(GENERATION, 20L, RefinedPhase.COMPLETED);

        // A fresh-version intermediate after the terminal.
        Verdict intermediate = SlotPhaseAdjudicator.adjudicate(
                terminal,
                new Event(GENERATION, 21L, RefinedPhase.D_RUNNING, false));
        assertTrue(intermediate instanceof Verdict.TombstoneAbsorb);
        assertEquals(RefinedPhase.COMPLETED,
                ((Verdict.TombstoneAbsorb) intermediate).terminalPhase());

        // Even a fresh finish of a different terminal flavour: no
        // resurrection, no second settlement (I3).
        Verdict otherFinish = SlotPhaseAdjudicator.adjudicate(
                terminal,
                new Event(GENERATION, 22L, RefinedPhase.CANCELLED, true));
        assertTrue(otherFinish instanceof Verdict.TombstoneAbsorb);

        // And a same-tick finish after the terminal.
        Verdict sameTick = SlotPhaseAdjudicator.adjudicate(
                terminal,
                new Event(GENERATION, 20L, RefinedPhase.FAILED, true));
        assertTrue(sameTick instanceof Verdict.TombstoneAbsorb);
    }

    @Test
    void crossGenerationEventsAreRejectedBeforeOrderingRulings() {
        State current = new State(GENERATION, 10L, RefinedPhase.P_RUNNING);

        Verdict ruling = SlotPhaseAdjudicator.adjudicate(
                current,
                new Event(GENERATION + 1L, 11L, RefinedPhase.D_LOADING, false));
        assertTrue(ruling instanceof Verdict.RejectCrossGeneration);
        assertEquals(GENERATION,
                ((Verdict.RejectCrossGeneration) ruling).stateGeneration());
        assertEquals(GENERATION + 1L,
                ((Verdict.RejectCrossGeneration) ruling).eventGeneration());
        assertFalse(ruling.advancesState());

        // Identity is checked even against a terminal state: another
        // generation's fact must not feed the tombstone audit.
        State terminal = new State(GENERATION, 20L, RefinedPhase.COMPLETED);
        Verdict atTerminal = SlotPhaseAdjudicator.adjudicate(
                terminal,
                new Event(GENERATION + 1L, 21L, RefinedPhase.D_RUNNING, false));
        assertTrue(atTerminal instanceof Verdict.RejectCrossGeneration);
    }

    @Test
    void finishFactsMustCarryTerminalPhasesAndViceVersa() {
        assertThrows(IllegalArgumentException.class,
                () -> new Event(GENERATION, 1L, RefinedPhase.P_RUNNING, true));
        assertThrows(IllegalArgumentException.class,
                () -> new Event(GENERATION, 1L, RefinedPhase.COMPLETED, false));
    }

    @Test
    void refinedPhasesFormTheV2ImplicationChain() {
        // I2 chain: D_* ⇒ PREFILL_DONE ⇒ DISPATCHED.
        assertTrue(RefinedPhase.D_RUNNING.implies(RefinedPhase.PREFILL_DONE));
        assertTrue(RefinedPhase.PREFILL_DONE.implies(RefinedPhase.DISPATCHED));
        assertTrue(RefinedPhase.D_LOADING.exceeds(RefinedPhase.PREFILL_DONE));

        // No backward implication (I1).
        assertFalse(RefinedPhase.PREFILL_DONE.implies(RefinedPhase.D_RUNNING));
        assertFalse(RefinedPhase.DISPATCHED.implies(RefinedPhase.P_RECEIVED));

        // Terminals are pairwise-incomparable lattice maxima.
        assertFalse(RefinedPhase.COMPLETED.implies(RefinedPhase.CANCELLED));
        assertFalse(RefinedPhase.CANCELLED.implies(RefinedPhase.COMPLETED));
        assertFalse(RefinedPhase.FAILED.exceeds(RefinedPhase.SLO_TIMEOUT));

        // v2 implication direction: later ⇒ earlier.  A terminal implies
        // every intermediate (single-exit resource release); an
        // intermediate implies no terminal (reaching PREFILL_DONE does not
        // imply COMPLETED).
        for (RefinedPhase terminal : List.of(
                RefinedPhase.COMPLETED, RefinedPhase.CANCELLED,
                RefinedPhase.SLO_TIMEOUT, RefinedPhase.FAILED)) {
            assertTrue(terminal.implies(RefinedPhase.INIT));
            assertTrue(terminal.implies(RefinedPhase.D_RUNNING));
            assertFalse(RefinedPhase.INIT.implies(terminal));
            assertFalse(RefinedPhase.D_RUNNING.implies(terminal));
            assertTrue(terminal.isTerminal());
        }
        assertFalse(RefinedPhase.D_RUNNING.isTerminal());
    }

    @Test
    void coarsePhaseProjectionFollowsTheTwoPhaseDeathTrack() {
        assertEquals(CoarsePhase.ACTIVE,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.INIT));
        assertEquals(CoarsePhase.ACTIVE,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.P_RUNNING));
        assertEquals(CoarsePhase.ACTIVE,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.D_RUNNING));
        assertEquals(CoarsePhase.TERMINALIZING,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.COMPLETED));
        assertEquals(CoarsePhase.TERMINALIZING,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.CANCELLED));
        assertEquals(CoarsePhase.TERMINALIZING,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.SLO_TIMEOUT));
        assertEquals(CoarsePhase.TERMINALIZING,
                SlotPhaseAdjudicator.coarsePhaseOf(RefinedPhase.FAILED));
    }

    /**
     * Stage-1 fix B (wiring lock): the storage-track projection is a real
     * reference to {@code RequestSlot.SlotPhase}, not a parallel enum —
     * every storage phase projects onto exactly one coarse phase, and the
     * two tracks never disagree on the storage-only TOMBSTONE state that
     * the harness consumes on its lock-free fast path.
     */
    @Test
    void slotPhaseProjectionIsTheRealStorageTrackWiring() {
        assertEquals(CoarsePhase.ACTIVE,
                SlotPhaseAdjudicator.coarsePhaseOf(
                        RequestSlot.SlotPhase.ACTIVE));
        assertEquals(CoarsePhase.TERMINALIZING,
                SlotPhaseAdjudicator.coarsePhaseOf(
                        RequestSlot.SlotPhase.TERMINALIZING));
        assertEquals(CoarsePhase.TOMBSTONE,
                SlotPhaseAdjudicator.coarsePhaseOf(
                        RequestSlot.SlotPhase.TOMBSTONE));
        // The projection is total and refuses null — a wiring mistake
        // must fail loudly, not silently map to a default phase.
        assertEquals(3, RequestSlot.SlotPhase.values().length);
        assertThrows(NullPointerException.class,
                () -> SlotPhaseAdjudicator.coarsePhaseOf(
                        (RequestSlot.SlotPhase) null));
    }

    @Test
    void repeatedAdjudicationReplayIsIdempotentForTheAdoptedState() {
        // Feeding the same event twice against the adopted next state must
        // be a dedup discard: single-application semantics for replays.
        State current = new State(GENERATION, 4L, RefinedPhase.DISPATCHED);
        Event event = new Event(GENERATION, 5L, RefinedPhase.D_LOADING, false);
        Verdict.Advance first =
                (Verdict.Advance) SlotPhaseAdjudicator.adjudicate(current, event);
        Verdict replay = SlotPhaseAdjudicator.adjudicate(
                first.nextState(), event);
        assertTrue(replay instanceof Verdict.DiscardStaleVersion);
    }
}
