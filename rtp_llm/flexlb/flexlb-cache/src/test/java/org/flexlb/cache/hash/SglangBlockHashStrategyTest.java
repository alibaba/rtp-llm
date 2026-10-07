package org.flexlb.cache.hash;

import org.flexlb.dao.loadbalance.TokenIds;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SglangBlockHashStrategyTest {

    private final BlockHashStrategy strategy = new SglangBlockHashStrategy();

    @Test
    void matchesPublishedTokenHashChainForCompletePages() {
        assertEquals(
                List.of(-3488128144981237669L),
                strategy.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4, 5}), 4, 0));
        assertEquals(
                List.of(-3488128144981237669L, 5674439469042975057L),
                strategy.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4, 5, 6, 7, 8}), 4, 0));
    }

    @Test
    void ignoresTrailingTokensThatDoNotFillAPage() {
        assertEquals(List.of(), strategy.calculate(TokenIds.wrap(new int[]{1, 2, 3}), 4, 0));
        assertEquals(List.of(), strategy.calculate(TokenIds.wrap(new int[]{10, 20, 30, 40}), 4, 1));
    }

    @Test
    void matchesPublishedEagleBigramHashChainAcrossPageBoundary() {
        assertEquals(
                List.of(-8847804484166691499L, 4989791362144317498L),
                strategy.calculate(TokenIds.wrap(new int[]{10, 20, 30, 40, 50}), 2, 1));
    }

    @Test
    void excludesTheFinalTokenFromEagleBigramHashing() {
        assertEquals(List.of(), strategy.calculate(TokenIds.wrap(new int[]{10}), 4, 1));
        assertEquals(
                List.of(8258502975543156532L),
                strategy.calculate(TokenIds.wrap(new int[]{10, 20, 30, 40, 99}), 4, 1));
    }

    @Test
    void rejectsInvalidHashInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(null, 4, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(TokenIds.wrap(new int[]{1}), 0, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(TokenIds.wrap(new int[]{1}), (long) Integer.MAX_VALUE + 1L, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(TokenIds.wrap(new int[]{1}), 1L << 32, 0));
        assertEquals(List.of(),
                strategy.calculate(TokenIds.wrap(new int[]{1}), Integer.MAX_VALUE, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(TokenIds.wrap(new int[]{1}), 4, -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> strategy.calculate(TokenIds.wrap(new int[]{1, 2}), 4, 2));
    }
}
