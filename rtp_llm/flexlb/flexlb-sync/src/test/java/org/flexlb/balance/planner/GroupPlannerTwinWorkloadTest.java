package org.flexlb.balance.planner;

import org.flexlb.balance.planner.GroupPlanner.Constraints;
import org.flexlb.balance.planner.GroupPlanner.Item;
import org.flexlb.balance.planner.GroupPlanner.Selection;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Twin r6/r7 inputs applied to the real pure planner, with a frozen clock. */
class GroupPlannerTwinWorkloadTest {

    @Test
    void identicalRequestsProduceFullBatchesOrSingletonTimeoutsDependingOnArrival() {
        Constraints limits = new Constraints(8, 1_000_000L, 22_000_000L, 0L, 400L);
        List<Item> burst = LongStream.rangeClosed(1, 16)
                .mapToObj(id -> item(id, 0L, 128L)).toList();
        Selection<Item> first = GroupPlanner.selectWithPrediction(burst, limits, null);
        Selection<Item> second = GroupPlanner.selectWithPrediction(burst.subList(8, 16), limits, null);
        assertEquals("batch_full", GroupPlanner.dispatchReason(first, limits, 0L));
        assertEquals("batch_full", GroupPlanner.dispatchReason(second, limits, 0L));
        assertEquals(8, first.items().size());
        assertEquals(8, second.items().size());

        List<Long> sparseDispatched = new ArrayList<>();
        for (long id = 1; id <= 16; id++) {
            long arrival = (id - 1) * 2_000L;
            List<Item> pending = List.of(item(id, arrival, 128L));
            Selection<Item> timeout = GroupPlanner.selectWithPrediction(pending, limits, null);
            assertEquals(null, GroupPlanner.dispatchReason(timeout, limits, arrival + 399L));
            assertEquals("fixed_window_timeout", GroupPlanner.dispatchReason(timeout, limits, arrival + 400L));
            assertEquals(1, timeout.items().size());
            sparseDispatched.add(timeout.items().getFirst().requestId());
        }
        assertEquals(burst.stream().map(Item::requestId).toList(), sparseDispatched,
                "both arrival profiles must dispatch the same 16 request IDs exactly once");
    }

    @Test
    void increasingDecisionGroupLimitChangesFullTriggerWithoutChangingRequests() {
        List<Item> arrivals = LongStream.rangeClosed(1, 8)
                .mapToObj(id -> item(id, (id - 1) * 50L, 128L)).toList();
        Constraints small = new Constraints(8, 1_000_000L, 22_000_000L, 0L, 400L);
        Constraints large = new Constraints(32, 1_000_000L, 22_000_000L, 0L, 400L);

        Selection<Item> full = GroupPlanner.selectWithPrediction(arrivals, small, null);
        assertEquals("batch_full", GroupPlanner.dispatchReason(full, small, 350L));
        Selection<Item> timeout = GroupPlanner.selectWithPrediction(arrivals, large, null);
        assertEquals(null, GroupPlanner.dispatchReason(timeout, large, 350L));
        assertEquals("fixed_window_timeout", GroupPlanner.dispatchReason(timeout, large, 400L));
        assertEquals(full.items(), timeout.items());
    }

    @Test
    void lowKvUsageMustNotAllowHeterogeneousPrefillBatchToExceedPaddedComputeBudget() {
        // Sum(tokens)=1,100 fits easily, but a three-row padded batch costs
        // 900*3=2,700 tokens and exceeds the 2,000-token compute limit.
        List<Item> arrivals = List.of(
                item(1L, 0L, 100L), item(2L, 0L, 900L), item(3L, 0L, 100L));
        Constraints limits = new Constraints(8, 2_000L, 22_000_000L, 0L, 400L);
        Selection<Item> result = GroupPlanner.selectWithPrediction(arrivals, limits, null);

        assertEquals(List.of(1L, 2L), result.items().stream().map(Item::requestId).toList());
        assertEquals(1_800L, result.paddedTokens());
        assertTrue(result.fitsCompute(2_000L));
        assertEquals("fixed_window_timeout", GroupPlanner.dispatchReason(result, limits, 400L));
    }

    private static Item item(long id, long arrivalMs, long tokens) {
        return new Item(id, 0, id, arrivalMs, Long.MAX_VALUE, tokens, 0L);
    }
}
