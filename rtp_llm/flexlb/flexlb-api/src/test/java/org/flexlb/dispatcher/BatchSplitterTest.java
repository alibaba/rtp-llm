package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

    @Test
    void rejectsNonPositiveK() {
        assertThrows(IllegalArgumentException.class, () -> BatchSplitter.split(List.of("a"), 0));
    }

    @Test
    void chunkCountIsTheSingleSourceForSplitSize() {
        // chunkCount(N, K) MUST equal split(...).size() — pre-assign sizes /batch_schedule by it
        assertEquals(BatchSplitter.split(List.of("a", "b", "c"), 2).size(), BatchSplitter.chunkCount(3, 2));
        assertEquals(0, BatchSplitter.chunkCount(0, 5));
        assertEquals(1, BatchSplitter.chunkCount(5, 5));
        assertEquals(100, BatchSplitter.chunkCount(500, 5));
        assertThrows(IllegalArgumentException.class, () -> BatchSplitter.chunkCount(3, 0));
    }
}
