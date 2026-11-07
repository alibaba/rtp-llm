package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PriorityOrderingTest {

    private static final Comparator<Node> REFERENCE =
            Comparator.comparingInt(Node::priority)
                    .reversed()
                    .thenComparingLong(Node::enqueueSeq)
                    .thenComparing(node -> Long.toString(node.requestId()));

    @Test
    void primitiveTotalOrderMatchesReferenceComparatorAcrossRandomKeys() {
        Random random = new Random(0x5EEDBEEFL);
        for (int i = 0; i < 100_000; i++) {
            Node left = new Node(random.nextInt(), random.nextLong(), random.nextLong());
            Node right = new Node(random.nextInt(), random.nextLong(), random.nextLong());
            assertEquivalent(left, right);
        }
    }

    @Test
    void primitiveTotalOrderMatchesStrictComparatorAtNumericBoundaries() {
        Node[] boundaries = {
            new Node(Integer.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE),
            new Node(Integer.MIN_VALUE, Long.MIN_VALUE, Long.MAX_VALUE),
            new Node(0, 0, 0),
            new Node(Integer.MAX_VALUE, Long.MAX_VALUE, Long.MIN_VALUE),
            new Node(Integer.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE)
        };
        for (Node left : boundaries) {
            for (Node right : boundaries) {
                assertEquivalent(left, right);
            }
        }
    }

    private static void assertEquivalent(Node left, Node right) {
        int expected = Integer.signum(REFERENCE.compare(left, right));
        int actual =
                Integer.signum(
                        PriorityOrdering.compareWithRequestId(
                                left.priority(),
                                left.enqueueSeq(),
                                Long.toString(left.requestId()),
                                right.priority(),
                                right.enqueueSeq(),
                                Long.toString(right.requestId())));
        assertEquals(expected, actual);
    }

    private record Node(int priority, long enqueueSeq, long requestId) {}
}
