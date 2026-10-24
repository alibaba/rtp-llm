package org.flexlb.balance.delivery;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchIdGeneratorTest {

    private static final int SEQUENCE_BITS = 21;
    private static final int NODE_ID_BITS = 12;

    @Test
    void idsAreStrictlyIncreasingWithinOneGenerator() {
        BatchIdGenerator generator = new BatchIdGenerator("10.0.0.1", 7001, 100L, 1_000L);
        long previous = generator.getAsLong();
        for (int i = 0; i < 1_000; i++) {
            long next = generator.getAsLong();
            assertTrue(next > previous);
            previous = next;
        }
    }

    @Test
    void nodeIdFitsInItsBitsAndDependsOnTheProcess() {
        long first = nodeIdOf(new BatchIdGenerator("10.0.0.1", 7001, 100L, 1_000L));
        long second = nodeIdOf(new BatchIdGenerator("10.0.0.2", 7001, 100L, 1_000L));
        assertTrue(first >= 0 && first < (1L << NODE_ID_BITS));
        assertTrue(second >= 0 && second < (1L << NODE_ID_BITS));
        assertNotEquals(first, second);
    }

    @Test
    void sameProcessIdentityGivesTheSameNodeId() {
        long first = nodeIdOf(new BatchIdGenerator("10.0.0.1", 7001, 100L, 1_000L));
        long second = nodeIdOf(new BatchIdGenerator("10.0.0.1", 7001, 100L, 1_000L));
        assertEquals(first, second);
    }

    private static long nodeIdOf(BatchIdGenerator generator) {
        return (generator.getAsLong() >>> SEQUENCE_BITS) & ((1L << NODE_ID_BITS) - 1L);
    }
}
