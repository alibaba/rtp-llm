package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingPostMergerTest {
    private final ObjectMapper m = new ObjectMapper();

    @Test
    void renumbersIndicesAndSumsUsage() {
        ObjectNode merged = m.createObjectNode();
        ArrayNode data = merged.putArray("data");
        data.add(itemWithIndex(0));  // chunk 0 item 0 (sub-relative was 0)
        data.add(itemWithIndex(1));  // chunk 0 item 1 (sub-relative was 1)
        data.add(itemWithIndex(0));  // chunk 1 item 0 (sub-relative was 0)
        data.add(itemWithIndex(1));  // chunk 1 item 1
        ObjectNode usage = merged.putObject("usage");
        usage.put("prompt_tokens", 5);
        usage.put("total_tokens", 5);

        ObjectNode sub0 = m.createObjectNode();
        sub0.putObject("usage").put("prompt_tokens", 5).put("total_tokens", 5);
        ObjectNode sub1 = m.createObjectNode();
        sub1.putObject("usage").put("prompt_tokens", 7).put("total_tokens", 7);
        var subs = List.of(
                SubBatchResult.ok(sub0, 2, 0),
                SubBatchResult.ok(sub1, 2, 2));

        new EmbeddingPostMerger().apply(merged, subs, List.of(), m);

        ArrayNode out = (ArrayNode) merged.get("data");
        assertEquals(0, out.get(0).get("index").asInt());
        assertEquals(1, out.get(1).get("index").asInt());
        assertEquals(2, out.get(2).get("index").asInt());
        assertEquals(3, out.get(3).get("index").asInt());
        assertEquals(12, merged.get("usage").get("prompt_tokens").asInt());
        assertEquals(12, merged.get("usage").get("total_tokens").asInt());
    }

    @Test
    void failedPositionsKeepEmbeddingNullIndex() {
        // Simulate what PartialFailureMerger produces: data is the merged array including
        // an EMBEDDING_NULL placeholder at the absolute failed index.
        ObjectNode merged = m.createObjectNode();
        ArrayNode data = merged.putArray("data");
        data.add(itemWithIndex(0));                                  // chunk 0 ok
        data.add(FailedItemFactory.EMBEDDING_NULL.build(1, "fe_timeout", m));  // chunk 1 failed
        data.add(itemWithIndex(0));                                  // chunk 2 ok (sub-relative was 0)
        ObjectNode sub0 = m.createObjectNode();
        sub0.putObject("usage").put("prompt_tokens", 3).put("total_tokens", 3);
        ObjectNode sub2 = m.createObjectNode();
        sub2.putObject("usage").put("prompt_tokens", 4).put("total_tokens", 4);
        var subs = List.of(
                SubBatchResult.ok(sub0, 1, 0),
                SubBatchResult.failed(1, 1, "fe_timeout"),
                SubBatchResult.ok(sub2, 1, 2));

        new EmbeddingPostMerger().apply(merged, subs, List.of(1), m);

        ArrayNode out = (ArrayNode) merged.get("data");
        assertEquals(0, out.get(0).get("index").asInt());
        assertEquals(1, out.get(1).get("index").asInt());
        assertTrue(out.get(1).get("embedding").isNull());
        assertEquals(2, out.get(2).get("index").asInt());
        assertEquals(7, merged.get("usage").get("prompt_tokens").asInt());
        assertEquals(7, merged.get("usage").get("total_tokens").asInt());
    }

    @Test
    void envelopeWithNullUsageIsReplacedNotCast() {
        ObjectNode merged = m.createObjectNode();
        merged.putArray("data").add(itemWithIndex(0));
        merged.putNull("usage");

        ObjectNode sub0 = m.createObjectNode();
        sub0.putObject("usage").put("prompt_tokens", 6).put("total_tokens", 6);
        var subs = List.of(SubBatchResult.ok(sub0, 1, 0));

        EmbeddingPostMerger.INSTANCE.apply(merged, subs, List.of(), m);

        assertEquals(6, merged.get("usage").get("prompt_tokens").asInt());
        assertEquals(6, merged.get("usage").get("total_tokens").asInt());
    }

    private ObjectNode itemWithIndex(int i) {
        ObjectNode o = m.createObjectNode();
        o.put("index", i);
        ArrayNode emb = o.putArray("embedding");
        emb.add(0.1).add(0.2);
        return o;
    }
}
