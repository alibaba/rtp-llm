package org.flexlb.util;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BlockCacheKeyCalculatorTest {

    @Test
    void matchesVllmSha256CborGoldenVectors() {
        List<Long> inputIds = new ArrayList<>();
        for (long tokenId = 0; tokenId < 128; tokenId++) {
            inputIds.add(tokenId);
        }

        assertEquals(
                List.of(-7527834946346035334L, -7860823284622341314L),
                BlockCacheKeyCalculator.calculate(inputIds, 64));
    }

    @Test
    void dropsFinalPartialBlock() {
        List<Long> inputIds = new ArrayList<>();
        for (long tokenId = 0; tokenId < 130; tokenId++) {
            inputIds.add(tokenId);
        }

        assertEquals(
                List.of(-7527834946346035334L, -7860823284622341314L),
                BlockCacheKeyCalculator.calculate(inputIds, 64));
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
