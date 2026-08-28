package org.flexlb.balance.planner;

import org.flexlb.balance.planner.GroupPlanner.Constraints;
import org.flexlb.balance.planner.GroupPlanner.Item;
import org.flexlb.balance.planner.GroupPlanner.Plan;
import org.flexlb.balance.planner.GroupPlanner.Selection;
import org.flexlb.balance.planner.GroupPlanner.Shape;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalDouble;
import java.util.function.ToDoubleFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exact-value contracts for the pure {@link GroupPlanner} domain
 * (Item / Shape / Constraints / Selection / Plan) and its selection and
 * readiness algorithm. Every expected value is computed directly from the
 * documented rules, not asserted loosely.
 */
@DisplayName("GroupPlanner pure domain and algorithm")
class GroupPlannerTest {

    private static final long BIG = 1_000_000L;
    /** Each candidate group is predicted at exactly 100ms per member. */
    private static final ToDoubleFunction<List<Item>> HUNDRED_PER_MEMBER =
            items -> 100.0 * items.size();

    private static Item item(long id, long seqLen, long enqueuedAtMs) {
        return new Item(id, /* priority */ 0, /* enqueueSeq */ id,
                enqueuedAtMs, /* expiresAtMs */ BIG, seqLen, /* hitCache */ 0L);
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Item validation")
    class ItemValidation {

        @Test
        void acceptsHitCacheAtBothBounds() {
            assertEquals(0L, item(1L, 100L, 0L).hitCache());
            assertEquals(100L, new Item(1L, 0, 1L, 0L, BIG, 100L, 100L).hitCache());
        }

        @Test
        void rejectsNegativeSeqLen() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Item(1L, 0, 1L, 0L, BIG, -1L, 0L));
        }

        @Test
        void rejectsHitCacheAboveSeqLen() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Item(1L, 0, 1L, 0L, BIG, 100L, 101L));
        }

        @Test
        void rejectsNegativeHitCache() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Item(1L, 0, 1L, 0L, BIG, 100L, -1L));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Shape arithmetic")
    class ShapeArithmetic {

        @Test
        void emptyShapeIsAllZero() {
            Shape empty = Shape.empty();
            assertEquals(0, empty.size());
            assertEquals(0L, empty.maxSeqLen());
            assertEquals(0L, empty.paddedTokens());
            assertEquals(0L, empty.kvTokens());
        }

        @Test
        void paddedTokensAreMaxSeqLenTimesSize() {
            // add(100) -> size1,max100,padded100,kv100
            Shape one = Shape.empty().add(100L);
            assertEquals(1, one.size());
            assertEquals(100L, one.maxSeqLen());
            assertEquals(100L, one.paddedTokens());
            assertEquals(100L, one.kvTokens());

            // add(50) -> size2,max100,padded200,kv150
            Shape two = one.add(50L);
            assertEquals(2, two.size());
            assertEquals(100L, two.maxSeqLen());
            assertEquals(200L, two.paddedTokens());
            assertEquals(150L, two.kvTokens());

            // add(200) -> size3,max200,padded600,kv350
            Shape three = two.add(200L);
            assertEquals(3, three.size());
            assertEquals(200L, three.maxSeqLen());
            assertEquals(600L, three.paddedTokens());
            assertEquals(350L, three.kvTokens());
        }

        @Test
        void fitsComputeIsStrictlyLessThanCapacity() {
            Shape two = Shape.empty().add(100L).add(100L); // paddedTokens = 200
            assertTrue(two.fitsCompute(201L));
            assertFalse(two.fitsCompute(200L), "capacity is a strict upper bound");
            assertFalse(two.fitsCompute(199L));
            assertFalse(two.fitsCompute(0L), "non-positive capacity never fits");
        }

        @Test
        void fitsKvIsInclusiveAndTreatsMaxAsUnlimited() {
            Shape two = Shape.empty().add(100L).add(50L); // kvTokens = 150
            assertTrue(two.fitsKv(150L), "KV capacity is inclusive");
            assertFalse(two.fitsKv(149L));
            assertTrue(two.fitsKv(Long.MAX_VALUE), "MAX means unlimited KV");
        }

        @Test
        void multiplyAndAddSaturateAtLongMax() {
            Shape huge = Shape.empty().add(Long.MAX_VALUE).add(1L);
            assertEquals(2, huge.size());
            assertEquals(Long.MAX_VALUE, huge.maxSeqLen());
            assertEquals(Long.MAX_VALUE, huge.paddedTokens());
            assertEquals(Long.MAX_VALUE, huge.kvTokens());
            // Saturated compute never fits any real capacity; MAX KV is unlimited.
            assertFalse(huge.fitsCompute(Long.MAX_VALUE));
            assertTrue(huge.fitsKv(Long.MAX_VALUE));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Constraints validation")
    class ConstraintsValidation {

        @Test
        void acceptsBoundaryRequestCounts() {
            assertEquals(1, new Constraints(1, BIG, BIG, 0L, 0L).maxRequests());
            assertEquals(1024, new Constraints(1024, BIG, BIG, 0L, 0L).maxRequests());
        }

        @Test
        void rejectsZeroMaxRequests() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Constraints(0, BIG, BIG, 0L, 0L));
        }

        @Test
        void acceptsArbitrarilyLargeMaxRequests() {
            // No upper bound after Phase B; only positive required.
            assertEquals(100_000, new Constraints(100_000, BIG, BIG, 0L, 0L).maxRequests());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Record invariants: Selection / Plan")
    class RecordInvariants {

        @Test
        void emptySelectionCannotCarryAPrediction() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Selection<>(List.of(), Shape.empty(), Long.MAX_VALUE,
                            false, OptionalDouble.of(10.0)));
        }

        @Test
        void readyPlanRequiresAReason() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Plan<>(List.of(item(1L, 10L, 0L)),
                            Shape.empty().add(10L), 0L, 300L, false,
                            OptionalDouble.empty(),
                            GroupPlanner.WindowReadiness.READY, null));
        }

        @Test
        void waitingPlanMustNotCarryAReason() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Plan<>(List.of(item(1L, 10L, 0L)),
                            Shape.empty().add(10L), 0L, 300L, false,
                            OptionalDouble.empty(),
                            GroupPlanner.WindowReadiness.WAITING, "unexpected"));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("select(): largest feasible homogeneous prefix")
    class SelectAlgorithm {

        @Test
        void emptyInputSelectsNothing() {
            Selection<Item> selection = GroupPlanner.select(
                    List.of(), GroupPlanner.itemAccess(),
                    new Constraints(8, BIG, BIG, 0L, 300L), null);
            assertTrue(selection.items().isEmpty());
            assertEquals(0, selection.shape().size());
            assertEquals(Long.MAX_VALUE, selection.windowOpenedAtMs());
            assertFalse(selection.predictionBoundaryTriggered());
            assertTrue(selection.selectedPredictionMs().isEmpty());
        }

        @Test
        void stopsExactlyAtMaxRequests() {
            List<Item> items = List.of(
                    item(1L, 10L, 1000L), item(2L, 10L, 2000L),
                    item(3L, 10L, 2000L), item(4L, 10L, 2000L),
                    item(5L, 10L, 2000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(3, BIG, BIG, 0L, 300L), null);

            assertEquals(List.of(1L, 2L, 3L),
                    selection.items().stream().map(Item::requestId).toList());
            assertEquals(3, selection.shape().size());
            assertEquals(10L, selection.shape().maxSeqLen());
            assertEquals(30L, selection.shape().paddedTokens());
            assertEquals(30L, selection.shape().kvTokens());
            assertEquals(1000L, selection.windowOpenedAtMs(),
                    "window opens at the minimum enqueue time of the group");
            assertFalse(selection.predictionBoundaryTriggered());
        }

        @Test
        void maxRequestsOneReturnsHeadOnly() {
            List<Item> items = List.of(item(1L, 10L, 1000L), item(2L, 10L, 1000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(1, BIG, BIG, 0L, 300L), null);
            assertEquals(List.of(1L),
                    selection.items().stream().map(Item::requestId).toList());
        }

        @Test
        void computeCapacityStopsGrowthWithoutTriggering() {
            // seqLen=100 each; paddedTokens after n = 100*n. cap=250 (strict).
            // n=2 -> 200<250 ok; n=3 -> 300<250 fails.
            List<Item> items = List.of(
                    item(1L, 100L, 1000L), item(2L, 100L, 1000L),
                    item(3L, 100L, 1000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, 250L, BIG, 0L, 300L), null);
            assertEquals(2, selection.items().size());
            assertEquals(200L, selection.shape().paddedTokens());
            assertFalse(selection.predictionBoundaryTriggered(),
                    "compute pressure stops growth but does not dispatch");
        }

        @Test
        void kvCapacityStopsGrowthWithoutTriggering() {
            // kvTokens after n = 100*n. kvCap=250 inclusive -> n=2 ok(200), n=3 fails(300).
            List<Item> items = List.of(
                    item(1L, 100L, 1000L), item(2L, 100L, 1000L),
                    item(3L, 100L, 1000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, 250L, 0L, 300L), null);
            assertEquals(2, selection.items().size());
            assertEquals(200L, selection.shape().kvTokens());
            assertFalse(selection.predictionBoundaryTriggered());
        }

        @Test
        void predictionDispatchBoundaryKeepsTheEqualMember() {
            // budget=200; predicted = 100*size. head=100(<200), size2=200(==200).
            // 200 is not > 200 (growth ok) but >= 200 (dispatch) -> keep 2, trigger.
            List<Item> items = List.of(
                    item(1L, 10L, 1000L), item(2L, 10L, 2000L),
                    item(3L, 10L, 3000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 200L, 300L), HUNDRED_PER_MEMBER);
            assertEquals(2, selection.items().size());
            assertTrue(selection.predictionBoundaryTriggered());
            assertEquals(OptionalDouble.of(200.0), selection.selectedPredictionMs());
        }

        @Test
        void predictionGrowthLimitDropsTheOverBudgetMember() {
            // budget=150; head=100(<150 dispatch no). size2 predicted=200 > 150 (growth
            // exceeded) -> remove 2nd, keep only head, trigger. selectedPrediction stays
            // the singleton's 100.
            List<Item> items = List.of(
                    item(1L, 10L, 1000L), item(2L, 10L, 2000L),
                    item(3L, 10L, 3000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 150L, 300L), HUNDRED_PER_MEMBER);
            assertEquals(List.of(1L),
                    selection.items().stream().map(Item::requestId).toList());
            assertTrue(selection.predictionBoundaryTriggered());
            assertEquals(OptionalDouble.of(100.0), selection.selectedPredictionMs());
        }

        @Test
        void singletonAlreadyAtBudgetTriggersImmediately() {
            // budget=100; head predicted=100 (>=100) -> singleton triggers, size1.
            List<Item> items = List.of(
                    item(1L, 10L, 1000L), item(2L, 10L, 2000L));
            Selection<Item> selection = GroupPlanner.select(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 100L, 300L), HUNDRED_PER_MEMBER);
            assertEquals(1, selection.items().size());
            assertTrue(selection.predictionBoundaryTriggered());
            assertEquals(OptionalDouble.of(100.0), selection.selectedPredictionMs());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("evaluateReadiness(): dispatch reason")
    class ReadinessAlgorithm {

        @Test
        void batchFullWhenGroupReachesMaxRequests() {
            List<Item> items = List.of(
                    item(1L, 10L, 1000L), item(2L, 10L, 1000L),
                    item(3L, 10L, 1000L), item(4L, 10L, 1000L));
            // window NOT elapsed (now == open) but BATCH_FULL takes precedence.
            Plan<Item> plan = GroupPlanner.plan(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(3, BIG, BIG, 0L, 300L),
                    /* nowMs */ 1000L, null);
            assertTrue(plan.ready());
            assertEquals(GroupPlanner.BATCH_FULL, plan.reason());
            assertEquals(3, plan.items().size());
            assertEquals(1300L, plan.collectionDeadlineMs());
        }

        @Test
        void predictedExecutionCapReason() {
            List<Item> items = List.of(item(1L, 10L, 1000L), item(2L, 10L, 2000L));
            Plan<Item> plan = GroupPlanner.plan(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 200L, 300L),
                    1000L, HUNDRED_PER_MEMBER);
            assertTrue(plan.ready());
            assertEquals(GroupPlanner.PREDICTED_EXECUTION_CAP, plan.reason());
        }

        @Test
        void fixedWindowTimeoutWhenWindowElapsedAndNotFull() {
            List<Item> items = List.of(item(1L, 10L, 1000L), item(2L, 10L, 1000L));
            // 2 < maxRequests(10), no predictor. window opens at 1000, window=300.
            // now=1300 -> 1300-1000=300 >= 300 -> elapsed.
            Plan<Item> plan = GroupPlanner.plan(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 0L, 300L),
                    /* nowMs */ 1300L, null);
            assertTrue(plan.ready());
            assertEquals(GroupPlanner.FIXED_WINDOW_TIMEOUT, plan.reason());
            assertEquals(1300L, plan.collectionDeadlineMs());
        }

        @Test
        void waitingWhenWindowNotYetElapsed() {
            List<Item> items = List.of(item(1L, 10L, 1000L), item(2L, 10L, 1000L));
            // now=1299 -> 299 < 300 -> not elapsed, not full -> WAITING, no reason.
            Plan<Item> plan = GroupPlanner.plan(
                    items, GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 0L, 300L),
                    /* nowMs */ 1299L, null);
            assertFalse(plan.ready());
            assertSame(null, plan.reason());
        }

        @Test
        void emptyPlanWaitsWithMaxWindowAndSaturatedDeadline() {
            Plan<Item> plan = GroupPlanner.plan(
                    List.of(), GroupPlanner.itemAccess(),
                    new Constraints(8, BIG, BIG, 0L, 300L),
                    /* nowMs */ Long.MAX_VALUE, null);
            assertFalse(plan.ready());
            assertTrue(plan.items().isEmpty());
            assertEquals(Long.MAX_VALUE, plan.windowOpenedAtMs());
            assertEquals(Long.MAX_VALUE, plan.collectionDeadlineMs());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("window helpers")
    class WindowHelpers {

        @Test
        void windowElapsedIsInclusiveFromOpenTime() {
            assertTrue(GroupPlanner.windowElapsed(1000L, 1300L, 300L));
            assertFalse(GroupPlanner.windowElapsed(1000L, 1299L, 300L));
            assertFalse(GroupPlanner.windowElapsed(1000L, 900L, 300L),
                    "a clock before the open time has not elapsed");
            assertFalse(GroupPlanner.windowElapsed(Long.MAX_VALUE, 5_000L, 300L),
                    "an unopened window never elapses");
        }

        @Test
        void collectionDeadlineAddsBoundedWindowAndSaturates() {
            assertEquals(1300L, GroupPlanner.collectionDeadlineMs(1000L, 300L));
            assertEquals(1000L, GroupPlanner.collectionDeadlineMs(1000L, -5L),
                    "a negative window is clamped to zero");
            assertEquals(Long.MAX_VALUE,
                    GroupPlanner.collectionDeadlineMs(Long.MAX_VALUE, 5L));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Requirement-derived corner cases (not implementation mirroring).
    //
    // The domain rule is "a decision group's head is indivisible and mandatory":
    // a single request cannot be split across groups, so the planner must always
    // return its head even when that head alone violates the compute or KV bound.
    // Capacity governs GROWTH past the head, never the head itself. Dropping an
    // oversized head here would silently lose the request instead of letting the
    // downstream reject-or-dispatch-alone path handle it.
    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Corner cases: indivisible head and degenerate shapes")
    class IndivisibleHeadCornerCases {

        @Test
        void oversizedSoloHeadIsStillSelectedWhenItExceedsComputeCapacity() {
            // Head padded tokens (1000) are NOT < capacity (100), yet a lone head
            // must still be selected — it cannot be split or silently dropped.
            Selection<Item> selection = GroupPlanner.select(
                    List.of(item(1L, 1000L, 1000L)), GroupPlanner.itemAccess(),
                    new Constraints(10, /* batchTokenCapacity */ 100L, BIG, 0L, 300L),
                    null);
            assertEquals(List.of(1L),
                    selection.items().stream().map(Item::requestId).toList());
            assertEquals(1000L, selection.shape().paddedTokens());
        }

        @Test
        void oversizedSoloHeadIsStillSelectedWhenItExceedsKvCapacity() {
            // Head kvTokens (1000) exceed kvCapacity (100); the lone head remains.
            Selection<Item> selection = GroupPlanner.select(
                    List.of(item(1L, 1000L, 1000L)), GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, /* batchKvCapacity */ 100L, 0L, 300L),
                    null);
            assertEquals(List.of(1L),
                    selection.items().stream().map(Item::requestId).toList());
            assertEquals(1000L, selection.shape().kvTokens());
        }

        @Test
        void growthStopsAtAnOversizedSecondMemberButTheHeadIsKept() {
            // Head fits; the second member would push padded tokens over capacity,
            // so growth stops and exactly the head is selected.
            Selection<Item> selection = GroupPlanner.select(
                    List.of(item(1L, 50L, 1000L), item(2L, 5000L, 2000L)),
                    GroupPlanner.itemAccess(),
                    new Constraints(10, /* cap */ 1000L, BIG, 0L, 300L), null);
            assertEquals(List.of(1L),
                    selection.items().stream().map(Item::requestId).toList());
        }

        @Test
        void anOversizedSoloHeadIsReadyOnWindowTimeout() {
            // Beyond selection: an indivisible oversized head must still be able to
            // dispatch. With no larger group possible, the window timeout releases
            // it rather than deadlocking.
            Plan<Item> plan = GroupPlanner.plan(
                    List.of(item(1L, 1000L, 1000L)), GroupPlanner.itemAccess(),
                    new Constraints(10, 100L, 100L, 0L, 300L),
                    /* nowMs */ 1300L, null);
            assertTrue(plan.ready());
            assertEquals(GroupPlanner.FIXED_WINDOW_TIMEOUT, plan.reason());
            assertEquals(List.of(1L),
                    plan.items().stream().map(Item::requestId).toList());
        }

        @Test
        void zeroSeqLenHeadProducesAZeroShapeThatFitsAnyPositiveCapacity() {
            Selection<Item> selection = GroupPlanner.select(
                    List.of(item(1L, 0L, 1000L)), GroupPlanner.itemAccess(),
                    new Constraints(10, BIG, BIG, 0L, 300L), null);
            assertEquals(1, selection.shape().size());
            assertEquals(0L, selection.shape().paddedTokens());
            assertEquals(0L, selection.shape().kvTokens());
            assertTrue(selection.shape().fitsCompute(1L));
            assertTrue(selection.shape().fitsKv(0L));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Contract asymmetry, pinned deliberately (NOT an implementation echo):
    // compute capacity is a STRICT ceiling (paddedTokens < cap) while KV capacity
    // is INCLUSIVE (kvTokens <= cap). This asymmetry is easy to break during
    // refactors and changes which groups are admissible at the exact boundary,
    // so it is asserted explicitly at one shared numeric boundary.
    // ASSUMPTION: the strict-vs-inclusive split is intentional; confirm with the
    // capacity owner before relying on it in new admission logic.
    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Corner case: compute/KV boundary asymmetry")
    class BoundaryAsymmetry {

        @Test
        void atTheExactBoundaryComputeIsExclusiveButKvIsInclusive() {
            Shape shape = Shape.empty().add(200L); // paddedTokens = 200, kvTokens = 200
            assertEquals(200L, shape.paddedTokens());
            assertEquals(200L, shape.kvTokens());
            assertFalse(shape.fitsCompute(200L),
                    "a group exactly at compute capacity must NOT be admitted");
            assertTrue(shape.fitsKv(200L),
                    "a group exactly at KV capacity MUST be admitted");
        }
    }
}
