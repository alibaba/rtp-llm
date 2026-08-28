package org.flexlb.balance.eviction;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.scheduler.WorkerBatcher.QueueSnapshot;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Requirement-driven exact-victim contracts for
 * {@link EvictionPlanner#planPrefillQueue}.
 *
 * <p>Requirements under test (from the domain, not the code):
 * <ul>
 *   <li>Eviction NEVER touches an equal-or-higher priority request (strict &lt;).</li>
 *   <li>Priority-0 requests are NEVER victims (even though 0 &lt; anything numerically).</li>
 *   <li>Exactly {@code deficit = queueSize + 1 − capacity} victims are chosen.</li>
 *   <li>Victims are the lowest priority first; among equal priority the NEWEST
 *       arrival first (protecting older, longer-waited requests).</li>
 *   <li>Insufficient candidates → null + typed reason, NEVER a partial set.</li>
 * </ul>
 */
@DisplayName("EvictionPlanner.planPrefillQueue exact-victim contracts")
class EvictionPlannerPrefillContractTest {

    private static final String EP = "prefill-a";

    private static PriorityRequestEnvelope incoming(int priority) {
        return new PriorityRequestEnvelope(9999L, priority, 0L, 0L, 0L, 0L, 0L);
    }

    private static ScheduledRequest item(long id, int priority, long enqueuedAtMs) {
        ScheduledRequest item = Mockito.mock(ScheduledRequest.class);
        Mockito.when(item.requestId()).thenReturn(id);
        Mockito.when(item.priority()).thenReturn(priority);
        Mockito.when(item.enqueuedAtMs()).thenReturn(enqueuedAtMs);
        Mockito.when(item.seqLen()).thenReturn(128L);
        return item;
    }

    private static PrefillEvictionProposal plan(
            PriorityRequestEnvelope envelope, int capacity,
            List<ScheduledRequest> items, Map<String, String> failures) {
        QueueSnapshot queue = new QueueSnapshot(EP, 1L, capacity, items);
        return EvictionPlanner.planPrefillQueue(envelope, List.of(queue), failures);
    }

    private static List<Long> victimIds(PrefillEvictionProposal proposal) {
        return proposal.victims().stream().map(ScheduledRequest::requestId).toList();
    }

    // ─── Priority eligibility ───────────────────────────────────────────

    @Nested
    @DisplayName("Priority eligibility")
    class PriorityEligibility {

        @Test
        void equalPriorityIsNeverAVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, List.of(item(1L, 70, 100L)), failures);
            assertNull(p, "an equal-priority request must never be evicted");
            assertEquals("insufficient_lower_priority_candidates", failures.get(EP));
        }

        @Test
        void noPriorityRequestIsNeverAVictimEvenWhenNumericallyLower() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, List.of(item(1L, 0, 100L)), failures);
            assertNull(p, "priority-0 must never be a victim");
            assertEquals("insufficient_lower_priority_candidates", failures.get(EP));
        }

        @Test
        void noPriorityItemsDoNotCountTowardCandidates() {
            // capacity=1, 3 items but two are priority-0. deficit=3.
            // Only 1 true candidate (P30) < deficit → infeasible.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1,
                    List.of(item(1L, 0, 100L), item(2L, 30, 100L), item(3L, 0, 100L)),
                    failures);
            assertNull(p);
            assertEquals("insufficient_lower_priority_candidates", failures.get(EP));
        }
    }

    // ─── Deficit boundary ───────────────────────────────────────────────

    @Nested
    @DisplayName("Deficit = queueSize + 1 − capacity")
    class DeficitBoundary {

        @Test
        void exactlyEnoughCandidatesForTheDeficitAreAllEvicted() {
            // capacity=1, 2 items → deficit=2, both are candidates → both evicted.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1,
                    List.of(item(1L, 30, 100L), item(2L, 30, 200L)), failures);
            assertEquals(2, p.victims().size());
        }

        @Test
        void oneCandidateShortOfTheDeficitIsInfeasible() {
            // capacity=1, 2 items but only 1 lower → deficit 2 > 1 candidates.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1,
                    List.of(item(1L, 30, 100L), item(2L, 70, 100L)), failures);
            assertNull(p, "insufficient candidates must yield no plan at all");
            assertEquals("insufficient_lower_priority_candidates", failures.get(EP));
        }

        @Test
        void aQueueThatIsNotYetFullIsInfeasible() {
            // capacity=5, 1 item → deficit = 1+1−5 < 0.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 5, List.of(item(1L, 30, 100L)), failures);
            assertNull(p);
            assertEquals("queue_not_full", failures.get(EP));
        }

        @Test
        void anUnboundedQueueIsInfeasible() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 0, List.of(item(1L, 30, 100L)), failures);
            assertNull(p);
            assertEquals("queue_unbounded", failures.get(EP));
        }
    }

    // ─── Victim ordering ────────────────────────────────────────────────

    @Nested
    @DisplayName("Victim ordering: lowest priority first, then newest arrival")
    class VictimOrdering {

        @Test
        void lowestPriorityCandidatesAreEvictedFirst() {
            // capacity=2, 3 items → deficit=2. P50,P30,P40 → evict P30,P40.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 2,
                    List.of(item(1L, 50, 100L), item(2L, 30, 100L), item(3L, 40, 100L)),
                    failures);
            assertEquals(List.of(2L, 3L), victimIds(p),
                    "the lowest priority candidates are evicted first");
        }

        @Test
        void amongEqualPriorityTheNewestArrivalIsEvictedFirst() {
            // capacity=1, 3 items → deficit=3. Two P30 with different arrival.
            // Newer (id2, arr200) before older (id1, arr100).
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1,
                    List.of(item(1L, 30, 100L), item(2L, 30, 200L), item(3L, 40, 50L)),
                    failures);
            assertEquals(List.of(2L, 1L, 3L), victimIds(p),
                    "protect the older request among equal priority");
        }
    }

    // ─── Cost ───────────────────────────────────────────────────────────

    @Nested
    @DisplayName("Cost arithmetic")
    class CostArithmetic {

        @Test
        void rawCostOfASingleP30VictimIsExactlyF30() {
            // f(30) = 1024^rank(30) = 1024^0 = 1.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, List.of(item(1L, 30, 100L)), failures);
            assertEquals(1L, p.rawCost());
        }

        @Test
        void rawCostOfASingleP40VictimIsExactlyF40() {
            // f(40) = 1024^rank(40) = 1024^1 = 1024.
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, List.of(item(1L, 40, 100L)), failures);
            assertEquals(1024L, p.rawCost());
        }
    }

    // ─── Fake ScheduledRequest ──────────────────────────────────────────────

}
