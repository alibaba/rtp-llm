package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * task61 L1 equivalence pins for {@link PriorityOrdering}: the primitive
 * {@link PriorityOrdering#compareStrict} and the boxed
 * {@link PriorityOrdering#STRICT} must implement exactly the historical
 * ordering rule (priority desc → enqueue-seq asc), and their tie semantics
 * must stay explicit — STRICT alone allows ties on (priority, enqueueSeq),
 * which is why every total-order consumer (the batcher queue comparator,
 * {@code PrefillQueueManager.ordersBefore}) appends a {@code requestId}
 * tie-break. If a future comparator change breaks that contract these tests
 * fail loudly instead of silently invalidating the top-k / peek shortcuts.
 */
class PriorityOrderingTest {

    private record TestItem(int priority, long enqueueSeq) implements Prioritized {
    }

    /** Hand-written historical rule: priority desc, then enqueue-seq asc. */
    private static int referenceCompare(int priorityA, long seqA, int priorityB, long seqB) {
        int byPriority = Integer.compare(priorityB, priorityA);
        return byPriority != 0 ? byPriority : Long.compare(seqA, seqB);
    }

    @Test
    void compareStrictMatchesHistoricalRuleOnRandomPairs() {
        Random random = new Random(61);
        for (int i = 0; i < 20_000; i++) {
            int priorityA = 1 + random.nextInt(100);
            int priorityB = random.nextBoolean() ? priorityA : 1 + random.nextInt(100);
            long seqA = random.nextInt(1_000);
            long seqB = random.nextBoolean() ? seqA : random.nextInt(1_000);

            int expected = referenceCompare(priorityA, seqA, priorityB, seqB);
            int primitive = PriorityOrdering.compareStrict(priorityA, seqA, priorityB, seqB);
            int boxed = PriorityOrdering.STRICT.compare(
                    new TestItem(priorityA, seqA), new TestItem(priorityB, seqB));

            assertEquals(Integer.signum(expected), Integer.signum(primitive));
            assertEquals(Integer.signum(expected), Integer.signum(boxed),
                    "STRICT must never diverge from compareStrict");
            // Antisymmetry of the shared rule
            assertEquals(Integer.signum(primitive),
                    -Integer.signum(PriorityOrdering.compareStrict(
                            priorityB, seqB, priorityA, seqA)));
        }
    }

    @Test
    void higherPriorityOrdersFirstThenFifoWithinPriority() {
        assertTrue(PriorityOrdering.compareStrict(70, 100, 50, 1) < 0,
                "higher priority must order first even with a later enqueue seq");
        assertTrue(PriorityOrdering.compareStrict(50, 1, 70, 100) > 0);
        assertTrue(PriorityOrdering.compareStrict(50, 1, 50, 2) < 0,
                "same priority must be FIFO by enqueue seq");
        assertTrue(PriorityOrdering.compareStrict(50, 2, 50, 1) > 0);
    }

    /**
     * Tie contract (leader-pinned): same priority + same enqueue seq is a
     * full STRICT tie — distinct requests in that state are ordered only by
     * the caller-appended {@code requestId} tie-break. This pin guards the
     * "queue comparator is a total order" premise behind the task61
     * peek≡sorted-head and top-k≡full-sort-prefix equivalences.
     */
    @Test
    void samePrioritySameSeqIsAStrictTieResolvedOnlyByCallerTieBreak() {
        assertEquals(0, PriorityOrdering.compareStrict(50, 123, 50, 123));
        assertEquals(0, PriorityOrdering.STRICT.compare(
                new TestItem(50, 123), new TestItem(50, 123)));
        // strict() is the same comparator instance typed to the subtype
        assertEquals(0, PriorityOrdering.<TestItem>strict().compare(
                new TestItem(50, 123), new TestItem(50, 123)));
    }
}
