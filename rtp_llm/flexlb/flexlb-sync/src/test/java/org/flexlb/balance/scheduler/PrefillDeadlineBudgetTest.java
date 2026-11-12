package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig.DeadlineBudgetConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillDeadlineBudgetTest {
    private final DeadlineBudgetConfig config = new DeadlineBudgetConfig();
    PrefillDeadlineBudgetTest() { config.setTokensPerSecond(1000); config.setCapacityFactor(1); }

    private PrefillDeadlineBudget.Candidate<String> q(String id, long sequence, long tokens, boolean cold, long remaining) {
        return new PrefillDeadlineBudget.Candidate<>(id, 50, sequence, tokens, cold, remaining);
    }

    private final PrefillDeadlineBudget.Fleet full = new PrefillDeadlineBudget.Fleet(1, 1, 1000, true);

    @Test void shortBurstAndUnknownStatusNeverAuthorizeShedding() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("cold", 1, 100_000, true, 5000));
        assertTrue(control.victims(queue, full, config, false, 0).isEmpty());
        assertTrue(control.victims(queue, full, config, false, 1000).isEmpty());
        assertTrue(control.victims(queue, new PrefillDeadlineBudget.Fleet(1, 0, 1000, true), config, false, 2000).isEmpty());
        assertTrue(control.victims(queue, new PrefillDeadlineBudget.Fleet(1, 1, 100_000, false), config, false, 5000).isEmpty());
    }

    @Test void deadlineBudgetRemovesOnlyEnoughColdWorkToMakeLaterDispatchFeasible() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("large-cold", 1, 80_000, true, 100_000),
                q("small-cold", 2, 10_000, true, 100_000), q("hot", 3, 100, false, 50_000));
        control.victims(queue, full, config, false, 0);
        assertEquals(List.of("large-cold"), control.victims(queue, full, config, false, 2000));
        assertTrue(control.victims(queue.subList(1, 3), full, config, false, 3000).isEmpty());
    }

    @Test void aShrinkingQueueStillShedsIfTheRemainingWorkMissesDeadlines() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("cold", 1, 80_000, true, 100_000), q("hot", 2, 100, false, 50_000));
        var initial = new ArrayList<>(queue);
        initial.add(q("drained", 3, 10, false, 100_000));
        control.victims(initial, full, config, false, 0);
        assertEquals(List.of("cold"), control.victims(queue, full, config, false, 2000));
    }

    @Test void expansionRecomputesCapacityAndClearsPreviousSaturationEvidence() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("cold", 1, 80_000, true, 100_000), q("hot", 2, 100, false, 50_000));
        control.victims(queue, full, config, false, 0);
        var expanded = new PrefillDeadlineBudget.Fleet(4, 4, 1000, true);
        assertTrue(control.victims(queue, expanded, config, false, 2000).isEmpty());
        assertTrue(control.victims(queue, expanded, config, false, 4000).isEmpty());
        assertTrue(control.victims(queue, full, config, false, 5000).isEmpty());
        assertEquals(List.of("cold"), control.victims(queue, full, config, false, 7000));
    }

    @Test void dispatchBudgetExcludesTheRequestsOwnWorkAndNeverEvictsHotRequests() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("cold", 1, 200_000, true, 100_000));
        control.victims(queue, full, config, false, 0);
        assertTrue(control.victims(queue, full, config, false, 2000).isEmpty());
        var onlyHot = List.of(q("hot", 1, 200_000, false, 1));
        assertTrue(control.victims(onlyHot, full, config, false, 3000).isEmpty());
    }

    @Test void usesEachRequestsDeadlineInsteadOfAnAverageOrTheQueueTimeout() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(q("cold", 1, 10_000, true, 120_000), q("short", 2, 100, false, 5000));
        control.victims(queue, full, config, false, 0);
        assertEquals(List.of("cold"), control.victims(queue, full, config, false, 2000));
    }

    @Test void lowerPriorityDeadlineCannotDisplaceHigherPriorityColdWork() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(new PrefillDeadlineBudget.Candidate<>("high", 90, 1, 80_000, true, 100_000),
                new PrefillDeadlineBudget.Candidate<>("low-hot", 10, 2, 100, false, 5000));
        control.victims(queue, full, config, true, 0);
        assertTrue(control.victims(queue, full, config, true, 2000).isEmpty());
    }

    @Test void fifoReliefIgnoresPriorityAndRemovesTheLargestColdWorkFirst() {
        var control = new PrefillDeadlineBudget();
        var queue = List.of(new PrefillDeadlineBudget.Candidate<>("small-low", 10, 1, 10_000, true, 100_000),
                new PrefillDeadlineBudget.Candidate<>("large-high", 90, 2, 80_000, true, 100_000),
                q("hot", 3, 100, false, 50_000));
        control.victims(queue, full, config, false, 0);
        assertEquals(List.of("large-high"), control.victims(queue, full, config, false, 2000));
    }
}
