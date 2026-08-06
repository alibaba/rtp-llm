package org.flexlb.autotpm;

import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.LongPredicate;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link InflightVictimSelector} (decision D3).
 */
class InflightVictimSelectorTest {

    private static final long NOW = 100_000L;
    private static final long CRITICAL_SECTION_MS = 1_000L;
    private static final LongPredicate NO_CANCEL_INTENT = id -> false;

    /** Candidate factory: runs long enough to pass the grace-period filter. */
    private static VictimCandidate candidate(long requestId, int priority, long iterateCount, long kvTokens) {
        return new VictimCandidate(requestId, priority, iterateCount, kvTokens,
                NOW - CRITICAL_SECTION_MS - 1, "10.0.0.1:8080");
    }

    private static Optional<VictimCandidate> select(List<VictimCandidate> candidates, int incomingPriority) {
        return InflightVictimSelector.select(candidates, incomingPriority,
                CRITICAL_SECTION_MS, NOW, NO_CANCEL_INTENT);
    }

    // ---- D3 selection order: priority asc → iterateCount asc → kvTokens asc → requestId asc ----

    @Test
    void select_lowestPriorityWins_regardlessOfOtherKeys() {
        List<VictimCandidate> candidates = List.of(
                candidate(1, 30, 1, 1),   // higher priority, minimal progress
                candidate(2, 10, 999, 999)); // lowest priority, deep progress

        assertEquals(2L, select(candidates, 70).orElseThrow().requestId());
    }

    @Test
    void select_samePriority_shallowerIterateCountWins() {
        List<VictimCandidate> candidates = List.of(
                candidate(1, 10, 50, 1),
                candidate(2, 10, 5, 999));

        assertEquals(2L, select(candidates, 70).orElseThrow().requestId());
    }

    @Test
    void select_samePriorityAndIterate_fewerKvTokensWins() {
        List<VictimCandidate> candidates = List.of(
                candidate(1, 10, 5, 800),
                candidate(2, 10, 5, 200));

        assertEquals(2L, select(candidates, 70).orElseThrow().requestId());
    }

    @Test
    void select_allKeysEqual_smallerRequestIdWins() {
        List<VictimCandidate> candidates = List.of(
                candidate(9, 10, 5, 200),
                candidate(3, 10, 5, 200));

        assertEquals(3L, select(candidates, 70).orElseThrow().requestId());
    }

    // ---- strictly-lower boundary (iron rule 2) ----

    @Test
    void select_equalPriority_neverSelected() {
        List<VictimCandidate> candidates = List.of(candidate(1, 50, 5, 200));

        assertTrue(select(candidates, 50).isEmpty());
        // one level below the incoming priority qualifies again
        assertTrue(select(candidates, 51).isPresent());
    }

    @Test
    void select_higherPriority_neverSelected() {
        List<VictimCandidate> candidates = List.of(candidate(1, 80, 5, 200));

        assertTrue(select(candidates, 70).isEmpty());
    }

    // ---- criticalSection grace-period filter ----

    @Test
    void select_withinCriticalSection_filteredOut() {
        // running for exactly criticalSectionMs - 1: still inside the grace period
        VictimCandidate fresh = new VictimCandidate(1, 10, 5, 200,
                NOW - CRITICAL_SECTION_MS + 1, "10.0.0.1:8080");

        assertTrue(select(List.of(fresh), 70).isEmpty());
    }

    @Test
    void select_exactlyAtCriticalSectionBoundary_eligible() {
        // nowMs - runningSinceMs == criticalSectionMs qualifies (>= semantics)
        VictimCandidate boundary = new VictimCandidate(1, 10, 5, 200,
                NOW - CRITICAL_SECTION_MS, "10.0.0.1:8080");

        assertEquals(1L, select(List.of(boundary), 70).orElseThrow().requestId());
    }

    // ---- pending cancel-intent filter ----

    @Test
    void select_pendingCancelIntent_skipped_nextBestWins() {
        List<VictimCandidate> candidates = List.of(
                candidate(1, 10, 5, 200),  // would win, but already being cancelled
                candidate(2, 20, 5, 200));
        LongPredicate cancelIntent = Set.of(1L)::contains;

        Optional<VictimCandidate> picked = InflightVictimSelector.select(
                candidates, 70, CRITICAL_SECTION_MS, NOW, cancelIntent);

        assertEquals(2L, picked.orElseThrow().requestId());
    }

    // ---- empty result ----

    @Test
    void select_emptyCandidates_returnsEmpty() {
        assertTrue(select(Collections.emptyList(), 70).isEmpty());
    }
}
