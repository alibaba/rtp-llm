package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
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

    @Test
    void splitArrayKeepsItemOrderAndShape() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode();
        for (int i = 0; i < 7; i++) {
            ObjectNode o = m.createObjectNode();
            o.put("i", i);
            arr.add(o);
        }
        List<ArrayNode> chunks = BatchSplitter.splitArray(arr, 3, m);
        assertEquals(3, chunks.size());
        assertEquals(3, chunks.get(0).size());
        assertEquals(3, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
        assertEquals(0, chunks.get(0).get(0).get("i").asInt());
        assertEquals(6, chunks.get(2).get(0).get("i").asInt());
    }

    @Test
    void splitArrayEmptyReturnsEmpty() {
        ObjectMapper m = new ObjectMapper();
        List<ArrayNode> chunks = BatchSplitter.splitArray(m.createArrayNode(), 5, m);
        assertTrue(chunks.isEmpty());
    }

    @Test
    void splitArraySingleChunkWhenArrayNotLargerThanK() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode().add(1).add(2);
        assertEquals(1, BatchSplitter.splitArray(arr, 5, m).size());
    }

}
