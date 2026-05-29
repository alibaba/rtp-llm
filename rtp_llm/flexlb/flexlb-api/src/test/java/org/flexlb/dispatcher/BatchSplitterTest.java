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

    @Test
    void splitByCountEvenlyDividesWhenDivisible() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode();
        for (int i = 0; i < 100; i++) arr.add(i);
        List<ArrayNode> chunks = BatchSplitter.splitByCount(arr, 5, m);
        assertEquals(5, chunks.size());
        for (ArrayNode c : chunks) {
            assertEquals(20, c.size());
        }
        assertEquals(0, chunks.get(0).get(0).asInt());
        assertEquals(99, chunks.get(4).get(19).asInt());
    }

    @Test
    void splitByCountDistributesRemainderToLeadingChunks() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode();
        for (int i = 0; i < 7; i++) arr.add(i);
        List<ArrayNode> chunks = BatchSplitter.splitByCount(arr, 5, m);
        assertEquals(5, chunks.size(), "7 items / 5 chunks → 5 chunks");
        assertEquals(2, chunks.get(0).size(), "remainder 2 lands on first 2 chunks");
        assertEquals(2, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
        assertEquals(1, chunks.get(3).size());
        assertEquals(1, chunks.get(4).size());
        assertEquals(0, chunks.get(0).get(0).asInt(), "order preserved");
        assertEquals(6, chunks.get(4).get(0).asInt());
    }

    @Test
    void splitByCountClampsCountWhenTotalSmallerThanRequested() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode().add(1).add(2).add(3);
        List<ArrayNode> chunks = BatchSplitter.splitByCount(arr, 10, m);
        assertEquals(3, chunks.size(),
                "3 items asked for 10 chunks → clamp to 3 to avoid empty chunks");
        for (ArrayNode c : chunks) {
            assertEquals(1, c.size());
        }
    }

    @Test
    void splitByCountSingleChunkPutsEverythingIn() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode().add(1).add(2).add(3).add(4);
        List<ArrayNode> chunks = BatchSplitter.splitByCount(arr, 1, m);
        assertEquals(1, chunks.size());
        assertEquals(4, chunks.get(0).size());
    }

    @Test
    void splitByCountEmptyArrayReturnsEmpty() {
        ObjectMapper m = new ObjectMapper();
        List<ArrayNode> chunks = BatchSplitter.splitByCount(m.createArrayNode(), 5, m);
        assertTrue(chunks.isEmpty());
    }

    // ───────────────────────── split(spec) dispatcher ─────────────────────────

    @Test
    void splitDispatchesToSplitArrayOnSizeMode() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode().add("a").add("b").add("c").add("d").add("e");
        List<ArrayNode> chunks = BatchSplitter.split(arr, SubBatchSpec.parse("size:2"), m);
        assertEquals(3, chunks.size(), "5 items @ size 2 → ceil(5/2)=3 chunks");
        assertEquals(2, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
    }

    @Test
    void splitDispatchesToSplitByCountOnCountMode() {
        ObjectMapper m = new ObjectMapper();
        ArrayNode arr = m.createArrayNode().add("a").add("b").add("c").add("d").add("e");
        List<ArrayNode> chunks = BatchSplitter.split(arr, SubBatchSpec.parse("count:3"), m);
        assertEquals(3, chunks.size(), "5 items @ count 3 → exactly 3 chunks");
        // Remainder 5%3=2 goes to leading chunks: [2,2,1]
        assertEquals(2, chunks.get(0).size());
        assertEquals(2, chunks.get(1).size());
        assertEquals(1, chunks.get(2).size());
    }

    @Test
    void splitEmptyArrayReturnsEmptyRegardlessOfMode() {
        ObjectMapper m = new ObjectMapper();
        assertTrue(BatchSplitter.split(m.createArrayNode(), SubBatchSpec.parse("size:5"), m).isEmpty());
        assertTrue(BatchSplitter.split(m.createArrayNode(), SubBatchSpec.parse("count:3"), m).isEmpty());
    }

}
