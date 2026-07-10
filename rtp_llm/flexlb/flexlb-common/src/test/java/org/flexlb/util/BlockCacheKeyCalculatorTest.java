package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BlockCacheKeyCalculatorTest {

    @Test
    void matchesRtpLlmCppGoldenVectors() {
        assertEquals(
                List.of(455111481605203084L, 6902853672176602142L),
                BlockCacheKeyCalculator.calculate(
                        List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L), 4));
        assertEquals(
                List.of(555616881177985336L, 4238552175599314317L),
                BlockCacheKeyCalculator.calculate(
                        List.of(151644L, 8948L, 198L, 2610L, 525L, 264L, 1296L, 13L), 4));
        assertEquals(
                List.of(-8366447758780319272L, 707304046373051542L),
                BlockCacheKeyCalculator.calculate(
                        List.of(0L, 2147483647L, 2147483648L, -1L), 2));
    }

    @Test
    void dropsFinalPartialBlock() {
        assertEquals(
                List.of(455111481605203084L, 6902853672176602142L),
                BlockCacheKeyCalculator.calculate(
                        List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L), 4));
    }

    @Test
    void returnsEmptyListWhenThereIsNoCompleteBlock() {
        assertEquals(List.of(), BlockCacheKeyCalculator.calculate(List.of(1L, 2L), 4));
    }

    @Test
    void rejectsInvalidInput() {
        assertThrows(
                IllegalArgumentException.class,
                () -> BlockCacheKeyCalculator.calculate(List.of(1L), 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> BlockCacheKeyCalculator.calculate(Arrays.asList(1L, null), 2));
    }
}
