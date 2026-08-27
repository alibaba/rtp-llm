package org.flexlb.balance.eviction;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

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
 *   <li>A saturated valid snapshot plans one 1:1 victim; an over-limit snapshot waits.</li>
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

    private static DeliveryItem item(long id, int priority, long enqueuedAtMs) {
        return new FakeDeliveryItem(id, priority, enqueuedAtMs);
    }

    private static PrefillEvictionProposal plan(
            PriorityRequestEnvelope envelope, int capacity,
            List<DeliveryItem> items, Map<String, String> failures) {
        return plan(envelope, capacity, items.size(), Long.MAX_VALUE,
                items, failures);
    }

    private static PrefillEvictionProposal plan(
            PriorityRequestEnvelope envelope, int capacity,
            long pending, long maxPending,
            List<DeliveryItem> items, Map<String, String> failures) {
        return plan(envelope, capacity, items.size(), pending, maxPending,
                items, failures);
    }

    private static PrefillEvictionProposal plan(
            PriorityRequestEnvelope envelope, int capacity,
            long waiting, long pending, long maxPending,
            List<DeliveryItem> items, Map<String, String> failures) {
        QueueSnapshot queue = new QueueSnapshot(
                EP, 1L, capacity, waiting, pending, maxPending, items);
        return EvictionPlanner.planPrefillQueue(envelope, List.of(queue), failures);
    }

    private static List<Long> victimIds(PrefillEvictionProposal proposal) {
        return proposal.victims().stream().map(DeliveryItem::requestId).toList();
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
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 3,
                    List.of(item(1L, 0, 100L), item(2L, 30, 100L), item(3L, 0, 100L)),
                    failures);
            assertEquals(List.of(2L), victimIds(p),
                    "priority-0 items must not enter the victim candidate set");
        }
    }

    // ─── Deficit boundary ───────────────────────────────────────────────

    @Nested
    @DisplayName("One-for-one replacement boundary")
    class DeficitBoundary {

        @Test
        void aSaturatedQueuePlansExactlyOneVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 2,
                    List.of(item(1L, 30, 100L), item(2L, 30, 200L)), failures);
            assertEquals(1, p.victims().size());
        }

        @Test
        void saturatedQueueWithoutLowerPriorityVictimIsInfeasible() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 2,
                    List.of(item(1L, 70, 100L), item(2L, 80, 100L)), failures);
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
        void anUnboundedQueueWithPendingCapacityNeedsNoVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 0, List.of(item(1L, 30, 100L)), failures);
            assertNull(p);
            assertEquals("queue_not_full", failures.get(EP));
        }

        @Test
        void pendingAtItsCapRequiresOneVictimEvenWhenQueueIsNotFull() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 5, 2L, 2L,
                    List.of(item(1L, 30, 100L)), failures);

            assertEquals(List.of(1L), victimIds(p));
        }

        @Test
        void pendingAboveItsCapWaitsInsteadOfPlanningAReplacement() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, 3L, 2L,
                    List.of(item(1L, 30, 100L)), failures);

            assertNull(p);
            assertEquals("over_limit_wait", failures.get(EP));
        }

        @Test
        void waitingAboveItsCapWaitsInsteadOfPlanningMultipleVictims() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, 2L, 2L, 10L,
                    List.of(
                            item(1L, 20, 100L),
                            item(2L, 30, 200L)),
                    failures);

            assertNull(p,
                    "the endpoint 1:1 replacement CAS cannot commit a multi-victim plan");
            assertEquals("over_limit_wait", failures.get(EP));
        }

        @Test
        void queueAndPendingDeficitsShareTheSameVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 2, 3L, 3L,
                    List.of(item(1L, 30, 100L), item(2L, 40, 100L)),
                    failures);

            assertEquals(List.of(1L), victimIds(p));
        }

        @Test
        void preparedHoldCountsTowardWaitingDeficitButOnlyActiveCanBeVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 2, 2L, 2L, 10L,
                    List.of(item(1L, 30, 100L)), failures);

            assertEquals(List.of(1L), victimIds(p),
                    "one hold plus one ACTIVE item already consumes both seats");
        }

        @Test
        void holdOnlySaturationWaitsWhenThereIsNoExactActiveVictim() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 1, 1L, 1L, 10L,
                    List.of(), failures);

            assertNull(p);
            assertEquals("insufficient_lower_priority_candidates",
                    failures.get(EP));
        }
    }

    // ─── Victim ordering ────────────────────────────────────────────────

    @Nested
    @DisplayName("Victim ordering: lowest priority first, then newest arrival")
    class VictimOrdering {

        @Test
        void lowestPriorityCandidatesAreEvictedFirst() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 3,
                    List.of(item(1L, 50, 100L), item(2L, 30, 100L), item(3L, 40, 100L)),
                    failures);
            assertEquals(List.of(2L), victimIds(p),
                    "the lowest priority candidate is the single 1:1 victim");
        }

        @Test
        void amongEqualPriorityTheNewestArrivalIsEvictedFirst() {
            Map<String, String> failures = new HashMap<>();
            PrefillEvictionProposal p = plan(
                    incoming(70), 3,
                    List.of(item(1L, 30, 100L), item(2L, 30, 200L), item(3L, 40, 50L)),
                    failures);
            assertEquals(List.of(2L), victimIds(p),
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

    // ─── Fake DeliveryItem ──────────────────────────────────────────────

    private record FakeDeliveryItem(long requestId, int priority, long enqueuedAtMs)
            implements DeliveryItem {

        @Override
        public long seqLen() {
            return 128L;
        }

        @Override
        public long hitCache() {
            return 0L;
        }
    }
}
