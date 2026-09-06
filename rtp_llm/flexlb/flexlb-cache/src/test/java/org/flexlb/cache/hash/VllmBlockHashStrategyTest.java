package org.flexlb.cache.hash;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VllmBlockHashStrategyTest {

    private final BlockHashStrategy strategy = new VllmBlockHashStrategy();

    @Test
    void matchesVllmSha256CborHash() {
        assertEquals(
                List.of(2164874634404590027L),
                strategy.calculate(new int[]{1, 2, 3, 4}, 4, 0));
    }

    @Test
    void matchesVllmEagleHashWithOneLookaheadToken() {
        assertEquals(
                List.of(2771287707320467766L, -4525836348354197114L),
                strategy.calculate(IntStream.rangeClosed(1, 9).toArray(), 4, 1));
    }

    @Test
    void dropsTheFinalPartialBlock() {
        assertEquals(
                List.of(-7527834946346035334L, -7860823284622341314L),
                strategy.calculate(IntStream.range(0, 130).toArray(), 64, 0));
        assertEquals(List.of(), strategy.calculate(new int[]{1, 2}, 4, 0));
    }

    @Test
    void keepsAllCalculatedBlocksInTheCacheablePrefix() {
        List<Long> hashes = List.of(11L, 22L);

        assertSame(hashes, strategy.cacheablePrefix(hashes, 9, 4, 1));
    }

    @Test
    void rejectsInvalidHashInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(null, 4, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(new int[]{1}, 0, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(new int[]{1}, 4, -1));
    }
}
