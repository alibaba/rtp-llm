package org.flexlb.cache.hash;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SglangBlockHashStrategyTest {

    private final BlockHashStrategy strategy = new SglangBlockHashStrategy();

    @Test
    void matchesPublishedTokenHashChainIncludingPartialPage() {
        assertEquals(
                List.of(-3488128144981237669L, -3787494577174227566L),
                strategy.calculate(new int[]{1, 2, 3, 4, 5}, 4, 0));
    }

    @Test
    void matchesPublishedEagleBigramHashChainAcrossPageBoundary() {
        assertEquals(
                List.of(-8847804484166691499L, 4989791362144317498L),
                strategy.calculate(new int[]{10, 20, 30, 40, 50}, 2, 1));
    }

    @Test
    void excludesTheFinalTokenFromEagleBigramHashing() {
        assertEquals(List.of(), strategy.calculate(new int[]{10}, 4, 1));
        assertEquals(
                List.of(-2735951481331064195L),
                strategy.calculate(new int[]{10, 20, 30, 40}, 4, 1));
    }

    @Test
    void returnsOnlyCompleteTokenPagesAsCacheablePrefix() {
        List<Long> hashes = strategy.calculate(new int[]{1, 2, 3, 4, 5}, 4, 0);

        assertEquals(
                List.of(-3488128144981237669L),
                strategy.cacheablePrefix(hashes, 5, 4, 0));
        assertEquals(
                List.of(-3488128144981237669L),
                strategy.cacheablePrefix(hashes.subList(0, 1), 4, 4, 0));
    }

    @Test
    void returnsOnlyCompleteBigramPagesAsCacheablePrefix() {
        List<Long> fullPageHashes = strategy.calculate(new int[]{10, 20, 30, 40, 50}, 4, 1);
        List<Long> partialPageHashes = strategy.calculate(new int[]{10, 20, 30, 40}, 4, 1);

        assertEquals(
                fullPageHashes,
                strategy.cacheablePrefix(fullPageHashes, 5, 4, 1));
        assertEquals(
                List.of(),
                strategy.cacheablePrefix(partialPageHashes, 4, 4, 1));
    }

    @Test
    void returnsEmptyForEmptyInputAndEmptyHashLists() {
        assertEquals(List.of(), strategy.calculate(new int[]{}, 4, 0));
        assertEquals(List.of(), strategy.cacheablePrefix(List.of(), 4, 4, 0));
        assertEquals(List.of(), strategy.cacheablePrefix(null, 4, 4, 0));
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
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(new int[]{1, 2}, 4, 2));
    }
}
