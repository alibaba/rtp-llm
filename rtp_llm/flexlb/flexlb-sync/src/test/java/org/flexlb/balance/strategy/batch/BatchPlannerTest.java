package org.flexlb.balance.strategy.batch;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchPlannerTest {

    @Test
    @DisplayName("Regret ordering preserves the unique fast worker for the request that needs it")
    void preservesTheUniqueFastWorkerForTheRequestThatNeedsIt() {
        assertEquals(List.of(1, 0), BatchPlanner.plan(List.of(
                request(new BatchCandidate("w1", 120, 100, 0),
                        new BatchCandidate("w2", 125, 100, 0)),
                request(new BatchCandidate("w1", 30, 10, 90),
                        new BatchCandidate("w2", 300, 100, 0)))));
    }

    @Test
    @DisplayName("Joint planning enforces aggregate KV capacity for every shared worker")
    void sharesWorkerKvCapacityAcrossTheWholePlan() {
        assertEquals(List.of(0, 1), BatchPlanner.plan(List.of(
                request(new BatchCandidate("w1", 10, 0, 0, 60, 100, 10)),
                request(new BatchCandidate("w1", 10, 0, 0, 60, 100, 10),
                        new BatchCandidate("w2", 20, 0, 0)))));
    }

    @Test
    @DisplayName("TTFT local improvement may exchange two placements when the whole plan improves")
    void improvesTheWholePlanByExchangingTwoPlacements() {
        assertEquals(List.of(0, 0, 1), BatchPlanner.plan(List.of(
                request(new BatchCandidate("w1", 80, 90, 0),
                        new BatchCandidate("w2", 140, 80, 0)),
                request(new BatchCandidate("w1", 80, 90, 0),
                        new BatchCandidate("w2", 140, 40, 0)),
                request(new BatchCandidate("w1", 30, 90, 0),
                        new BatchCandidate("w2", 80, 30, 0)))));
    }

    @Test
    @DisplayName("Cache affinity selects additional cached tokens within the configured TTFT allowance")
    void choosesMoreCachedTokensWithinTheFixedTtftAllowance() {
        assertEquals(List.of(1), BatchPlanner.plan(List.of(new BatchPlanningRequest(List.of(
                new BatchCandidate("w1", 10, 10, 0),
                new BatchCandidate("w2", 25, 10, 80),
                new BatchCandidate("w3", 40, 10, 100)), 20, 5, 100, true))));
    }

    @Test
    @DisplayName("Cache affinity cannot consume another request's TTFT allowance")
    void cacheAffinityCannotSpendAnotherRequestsTtftAllowance() {
        assertEquals(List.of(0, 0), BatchPlanner.plan(List.of(
                new BatchPlanningRequest(List.of(
                        new BatchCandidate("w1", 10, 40, 0),
                        new BatchCandidate("w2", 20, 80, 80)), 20, 5, 100, true),
                new BatchPlanningRequest(List.of(
                        new BatchCandidate("w2", 10, 0, 90),
                        new BatchCandidate("w1", 100, 0, 0)), 10, 5, 100, true))));
    }

    @Test
    @DisplayName("Single-request planning skips a candidate without delivery capacity")
    void singleRequestSkipsCandidatesWithoutCapacity() {
        assertEquals(List.of(1), BatchPlanner.plan(List.of(request(
                new BatchCandidate("w1", 10, 0, 0, 60, 100, 0),
                new BatchCandidate("w2", 20, 0, 0)))));
    }

    @Test
    @DisplayName("Configured completion-search budget controls repair of an incomplete greedy assignment")
    void completionSearchBudgetControlsRepairOfAnIncompleteGreedyPlan() {
        List<BatchPlanningRequest> requests = List.of(
                request(new BatchCandidate("w3", 1, 0, 0, 0, 0, 1),
                        new BatchCandidate("w2", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w1", 4, 0, 0, 0, 0, 1)),
                request(new BatchCandidate("w2", 7, 0, 0, 0, 0, 1),
                        new BatchCandidate("w3", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w1", 1, 0, 0, 0, 0, 1)),
                request(new BatchCandidate("w1", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w3", 4, 0, 0, 0, 0, 1)));

        assertEquals(2L, BatchPlanner.plan(requests, 1).stream()
                .filter(candidate -> candidate >= 0)
                .count());
        assertEquals(3L, BatchPlanner.plan(requests, 4096).stream()
                .filter(candidate -> candidate >= 0)
                .count());
    }

    @Test
    @DisplayName("Joint planning enforces aggregate delivery credits for every shared worker")
    void sharesWorkerDeliveryCreditsAcrossTheWholePlan() {
        assertEquals(List.of(0, 1), BatchPlanner.plan(List.of(
                request(new BatchCandidate("w1", 10, 0, 0, 0, 0, 1)),
                request(new BatchCandidate("w1", 10, 0, 0, 0, 0, 1),
                        new BatchCandidate("w2", 20, 0, 0)))));
    }

    @Test
    @DisplayName("An infeasible request remains unplaced without displacing a feasible peer")
    void leavesPermanentlyInfeasibleRequestUnplaced() {
        assertEquals(List.of(-1, 0), BatchPlanner.plan(List.of(
                request(new BatchCandidate("w1", 10, 0, 0, 0, 0, 0)),
                request(new BatchCandidate("w2", 20, 0, 0)))));
    }

    @Test
    @DisplayName("Cache affinity rejects a higher-hit candidate below its minimum prefix-hit threshold")
    void cacheAffinityRequiresTheConfiguredMinimumPrefixHitRate() {
        assertEquals(List.of(0), BatchPlanner.plan(List.of(
                new BatchPlanningRequest(List.of(
                        new BatchCandidate("w1", 10, 0, 0),
                        new BatchCandidate("w2", 15, 0, 9)),
                        10, 10, 100, true))));
    }

    @Test
    @DisplayName("Planner trace identifies the completion repair that recovers a greedy miss")
    void traceIdentifiesCompletionRepair() {
        List<BatchPlanningRequest> requests = List.of(
                request(new BatchCandidate("w3", 1, 0, 0, 0, 0, 1),
                        new BatchCandidate("w2", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w1", 4, 0, 0, 0, 0, 1)),
                request(new BatchCandidate("w2", 7, 0, 0, 0, 0, 1),
                        new BatchCandidate("w3", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w1", 1, 0, 0, 0, 0, 1)),
                request(new BatchCandidate("w1", 3, 0, 0, 0, 0, 1),
                        new BatchCandidate("w3", 4, 0, 0, 0, 0, 1)));

        BatchPlan plan = BatchPlanner.planWithTrace(requests, 4096);

        assertEquals(2, plan.greedyPlacedCount());
        assertEquals(3, plan.finalPlacedCount());
        assertTrue(plan.completionSearch().invoked());
        assertEquals(1, plan.completionSearch().recoveredPlacements());
        assertTrue(plan.requestChanges().stream()
                .flatMap(List::stream)
                .anyMatch(change -> change == BatchPlan.RequestChange.COMPLETION_REPAIR));
    }

    @Test
    @DisplayName("Planner trace distinguishes a TTFT swap from a cache-affinity relocation")
    void traceDistinguishesOptimizationOperations() {
        BatchPlan ttftPlan = BatchPlanner.planWithTrace(List.of(
                request(new BatchCandidate("w1", 80, 90, 0),
                        new BatchCandidate("w2", 140, 80, 0)),
                request(new BatchCandidate("w1", 80, 90, 0),
                        new BatchCandidate("w2", 140, 40, 0)),
                request(new BatchCandidate("w1", 30, 90, 0),
                        new BatchCandidate("w2", 80, 30, 0))), 4096);
        BatchPlan cachePlan = BatchPlanner.planWithTrace(List.of(
                new BatchPlanningRequest(List.of(
                        new BatchCandidate("w1", 10, 10, 0),
                        new BatchCandidate("w2", 25, 10, 80)),
                        20, 5, 100, true)), 4096);

        assertEquals(1, ttftPlan.ttftOptimization().swapCount());
        assertTrue(ttftPlan.requestChanges().get(1)
                .contains(BatchPlan.RequestChange.TTFT_SWAP));
        assertTrue(ttftPlan.requestChanges().get(2)
                .contains(BatchPlan.RequestChange.TTFT_SWAP));
        assertEquals(1, cachePlan.cacheAffinityOptimization().moveCount());
        assertTrue(cachePlan.requestChanges().getFirst()
                .contains(BatchPlan.RequestChange.CACHE_AFFINITY_MOVE));
        assertEquals(1, cachePlan.finalChangedFromGreedyCount());
    }

    private static BatchPlanningRequest request(BatchCandidate... candidates) {
        return new BatchPlanningRequest(List.of(candidates), 0, 0, 100, false);
    }
}
