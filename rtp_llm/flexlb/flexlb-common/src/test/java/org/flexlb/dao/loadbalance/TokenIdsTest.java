package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TokenIdsTest {

    @Test
    void wrapsArrayWithoutCopying() {
        int[] values = {1, 2, 3};
        TokenIds tokenIds = TokenIds.wrap(values);

        values[1] = 22;

        assertEquals(22, tokenIds.getInt(1));
    }

    @Test
    void indexedViewReadsLazily() {
        int[] values = {11, 22, 33};
        AtomicInteger reads = new AtomicInteger();
        TokenIds tokenIds = TokenIds.wrap(values.length, index -> {
            reads.incrementAndGet();
            return values[index];
        });

        assertEquals(3, tokenIds.size());
        assertEquals(0, reads.get());
        assertEquals(22, tokenIds.getInt(1));
        assertEquals(1, reads.get());
    }
}
