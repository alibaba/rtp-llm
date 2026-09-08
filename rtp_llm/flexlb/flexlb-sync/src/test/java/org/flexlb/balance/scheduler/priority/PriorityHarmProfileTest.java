package org.flexlb.balance.scheduler.priority;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PriorityHarmProfileTest {

    @Test
    void exact_priority_bucket_dominates_unbounded_lower_priority_harm() {
        PriorityHarmProfile lower = PriorityHarmProfile.builder()
                .add(49, BigInteger.TEN.pow(100))
                .build();
        PriorityHarmProfile higher = PriorityHarmProfile.builder()
                .add(50, BigInteger.ONE)
                .build();

        assertTrue(lower.compareTo(higher) < 0);
    }

    @Test
    void bucket_aggregation_and_combination_do_not_overflow() {
        BigInteger huge = BigInteger.valueOf(Long.MAX_VALUE);
        PriorityHarmProfile first = PriorityHarmProfile.builder().add(37, huge).build();
        PriorityHarmProfile second = PriorityHarmProfile.builder().add(37, huge).build();

        assertEquals(huge.multiply(BigInteger.TWO), first.plus(second).harmAt(37));
        assertEquals(Long.MAX_VALUE,
                PriorityCostFunction.saturatedAdd(Long.MAX_VALUE, 1));
        assertEquals(Long.MAX_VALUE,
                PriorityCostFunction.saturatedMultiply(Long.MAX_VALUE, 2));
    }
}
