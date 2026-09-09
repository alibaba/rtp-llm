package org.flexlb.balance.planner;

import org.flexlb.balance.planner.GroupPlanner.Constraints;
import org.flexlb.balance.planner.GroupPlanner.Item;
import org.flexlb.balance.planner.GroupPlanner.Plan;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Twin r6/r7 inputs applied to the real pure planner, with a frozen clock. */
class GroupPlannerTwinWorkloadTest {

    @Test
    void identicalRequestsProduceFullBatchesOrSingletonTimeoutsDependingOnArrival() {
        Constraints limits = new Constraints(8, 1_000_000L, 22_000_000L, 0L, 400L);
        List<Item> burst = LongStream.rangeClosed(1, 16)
                .mapToObj(id -> item(id, 0L, 128L)).toList();
        Plan<Item> first = plan(burst, limits, 0L);
        Plan<Item> second = plan(burst.subList(8, 16), limits, 0L);
        assertEquals("batch_full", first.reason());
        assertEquals("batch_full", second.reason());
        assertEquals(8, first.items().size());
        assertEquals(8, second.items().size());

        List<Long> sparseDispatched = new ArrayList<>();
        for (long id = 1; id <= 16; id++) {
            long arrival = (id - 1) * 2_000L;
            List<Item> pending = List.of(item(id, arrival, 128L));
            assertFalse(plan(pending, limits, arrival + 399L).ready());
            Plan<Item> timeout = plan(pending, limits, arrival + 400L);
            assertEquals("fixed_window_timeout", timeout.reason());
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

        Plan<Item> full = plan(arrivals, small, 350L);
        assertEquals("batch_full", full.reason());
        assertFalse(plan(arrivals, large, 350L).ready());
        Plan<Item> timeout = plan(arrivals, large, 400L);
        assertEquals("fixed_window_timeout", timeout.reason());
        assertEquals(full.items(), timeout.items());
    }

    @Test
    void lowKvUsageMustNotAllowHeterogeneousPrefillBatchToExceedPaddedComputeBudget() {
        // Sum(tokens)=1,100 fits easily, but a three-row padded batch costs
        // 900*3=2,700 tokens and exceeds the 2,000-token compute limit.
        List<Item> arrivals = List.of(
                item(1L, 0L, 100L), item(2L, 0L, 900L), item(3L, 0L, 100L));
        Constraints limits = new Constraints(8, 2_000L, 22_000_000L, 0L, 400L);
        Plan<Item> result = plan(arrivals, limits, 400L);

        assertEquals(List.of(1L, 2L), result.items().stream().map(Item::requestId).toList());
        assertEquals(1_800L, result.shape().paddedTokens());
        assertTrue(result.shape().fitsCompute(2_000L));
        assertEquals("fixed_window_timeout", result.reason());
    }

    private static Plan<Item> plan(List<Item> items, Constraints limits, long nowMs) {
        return GroupPlanner.plan(items, GroupPlanner.itemAccess(), limits, nowMs, null);
    }

    private static Item item(long id, long arrivalMs, long tokens) {
        return new Item(id, 0, id, arrivalMs, Long.MAX_VALUE, tokens, 0L);
    }
}
