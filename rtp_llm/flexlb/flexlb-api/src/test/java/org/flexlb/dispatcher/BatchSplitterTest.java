package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchSplitterTest {

    @Test
    void splitsEvenlyPreservingOrder() {
        List<List<String>> chunks = BatchSplitter.split(List.of("a", "b", "c", "d"), 2);
        assertEquals(2, chunks.size());
        assertEquals(List.of("a", "b"), chunks.get(0));
        assertEquals(List.of("c", "d"), chunks.get(1));
    }

    @Test
    void lastChunkMayBeSmaller() {
        List<List<String>> chunks = BatchSplitter.split(List.of("a", "b", "c"), 2);
        assertEquals(2, chunks.size());
        assertEquals(List.of("c"), chunks.get(1));
    }

    @Test
    void singleChunkWhenBatchNotLargerThanK() {
        assertEquals(1, BatchSplitter.split(List.of("a", "b"), 5).size());
    }

    @Test
    void emptyInputYieldsNoChunks() {
        assertTrue(BatchSplitter.split(List.of(), 5).isEmpty());
    }

}
