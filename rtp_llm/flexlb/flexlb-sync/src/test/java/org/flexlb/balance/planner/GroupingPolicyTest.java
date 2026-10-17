package org.flexlb.balance.planner;

import org.flexlb.balance.planner.GroupPlanner.Constraints;
import org.flexlb.balance.planner.GroupPlanner.Item;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GroupingPolicyTest {
    private static final long UNLIMITED = Long.MAX_VALUE;

    private static Item item(long id, long enqueuedAt, long tokens) {
        return new Item(id, 50, id, enqueuedAt, UNLIMITED, tokens, 0L);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void emptyInputsCannotDispatch(boolean single) {
        GroupingPolicy policy = single ? GroupingPolicy.SINGLE : GroupingPolicy.FIXED_WINDOW;
        var constraints = new Constraints(1, UNLIMITED, UNLIMITED, 0, 0);
        var selection = policy.select(List.<Item>of(), constraints, null);
        assertTrue(selection.items().isEmpty());
        assertNull(policy.dispatchReason(selection, constraints, 10));
    }

    @Test
    void singleConsumesOnlyTheHeadWithoutPredictionOrCollectionWait() {
        var head = item(1, 100, 1000);
        Iterable<Item> bounded = () -> new Iterator<>() {
            private boolean consumed;
            public boolean hasNext() { return true; }
            public Item next() {
                if (consumed) { throw new AssertionError("SINGLE read beyond its head"); }
                consumed = true;
                return head;
            }
        };
        var constraints = new Constraints(1, 10, 10, 0, 0);
        var selection = GroupingPolicy.SINGLE.select(bounded, constraints,
                (added, prefix) -> { throw new AssertionError("SINGLE does not predict a group"); });
        assertEquals(List.of(head), selection.items());
        assertEquals("single_request", GroupingPolicy.SINGLE.dispatchReason(selection, constraints, 100));
        assertFalse(selection.fitsKv(10));
    }

    @Test
    void fixedWindowUsesTheOldestSelectedMemberAndWakesAtItsExactDeadline() {
        var items = List.of(item(1, 100, 10), item(2, 90, 10));
        var constraints = new Constraints(3, UNLIMITED, UNLIMITED, 0, 20);
        var selection = GroupingPolicy.FIXED_WINDOW.select(items, constraints, null);
        assertNull(GroupingPolicy.FIXED_WINDOW.dispatchReason(selection, constraints, 109));
        assertEquals(110, GroupPlanner.collectionDeadlineMs(selection.windowOpenedAtMs(), constraints.collectionWindowMs()));
        assertEquals(GroupPlanner.FIXED_WINDOW_TIMEOUT,
                GroupingPolicy.FIXED_WINDOW.dispatchReason(selection, constraints, 110));
        assertEquals(items, selection.items());
    }

    @Test
    void fullGroupIsReadyBeforeItsWindowExpires() {
        var constraints = new Constraints(1, UNLIMITED, UNLIMITED, 0, 50);
        var selection = GroupingPolicy.FIXED_WINDOW.select(List.of(item(1, 100, 10)), constraints, null);
        assertEquals(GroupPlanner.BATCH_FULL, GroupingPolicy.FIXED_WINDOW.dispatchReason(selection, constraints, 100));
    }

    @Test
    void predictedBudgetExcludesTheAdditionalMemberAndRetainsTheIndivisibleHead() {
        var items = List.of(item(1, 100, 10), item(2, 100, 10), item(3, 100, 10));
        AtomicInteger predictions = new AtomicInteger();
        var constraints = new Constraints(10, UNLIMITED, UNLIMITED, 150, 1000);
        var selection = GroupingPolicy.FIXED_WINDOW.select(items, constraints,
                (added, prefix) -> { predictions.incrementAndGet(); return prefix.size() * 100; });
        assertEquals(List.of(items.getFirst()), selection.items());
        assertEquals(GroupPlanner.PREDICTED_EXECUTION_CAP,
                GroupingPolicy.FIXED_WINDOW.dispatchReason(selection, constraints, 100));
        assertEquals(100, selection.selectedPredictionMs().orElseThrow());
        assertEquals(2, predictions.get());
        var oversizedConstraints = new Constraints(10, 1, 1, 50, 1000);
        var oversized = GroupingPolicy.FIXED_WINDOW.select(items, oversizedConstraints, (added, prefix) -> 100);
        assertEquals(GroupPlanner.PREDICTED_EXECUTION_CAP,
                GroupingPolicy.FIXED_WINDOW.dispatchReason(oversized, oversizedConstraints, 100));
        assertEquals(List.of(items.getFirst()), oversized.items());
    }

    @Test
    void repeatedSelectionsUseCurrentCapacityAndDoNotRetainPredictionState() {
        var items = List.of(item(1, 100, 10), item(2, 100, 10));
        var before = GroupingPolicy.FIXED_WINDOW.select(items, new Constraints(2, 21, 20, 0, 0), null);
        var after = GroupingPolicy.FIXED_WINDOW.select(items, new Constraints(2, 20, 20, 0, 0), null);
        assertEquals(2, before.items().size());
        assertEquals(1, after.items().size());
        assertEquals(2, items.size());
    }

    @Test
    void waitDeadlineSaturatesAndInvalidInputsAreRejected() {
        var constraints = new Constraints(2, UNLIMITED, UNLIMITED, 0, 20);
        var selection = GroupingPolicy.FIXED_WINDOW.select(List.of(item(1, Long.MAX_VALUE - 1, 1)), constraints, null);
        assertNull(GroupingPolicy.FIXED_WINDOW.dispatchReason(selection, constraints, Long.MAX_VALUE - 1));
        assertEquals(Long.MAX_VALUE, GroupPlanner.collectionDeadlineMs(selection.windowOpenedAtMs(), 20));
        assertThrows(IllegalArgumentException.class, () -> GroupingPolicy.SINGLE.select(List.of(),
                new Constraints(2, UNLIMITED, UNLIMITED, 0, 0), null));
        var invalidDeadline = GroupingPolicy.FIXED_WINDOW.select(List.of(item(1, -100, 1)), constraints, null);
        assertThrows(IllegalArgumentException.class,
                () -> GroupingPolicy.FIXED_WINDOW.dispatchReason(invalidDeadline, constraints, -200));
    }
}
