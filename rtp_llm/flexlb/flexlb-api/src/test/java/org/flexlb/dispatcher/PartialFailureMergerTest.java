package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PartialFailureMergerTest {
    private final ObjectMapper m = new ObjectMapper();
    private final BatchEndpointSpec batchInferSpec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

    @Test
    void allSuccessMergesArraysNoPartialField() {
        ObjectNode body1 = m.createObjectNode();
        ArrayNode arr1 = body1.putArray("response_batch");
        arr1.add(textNode("a0"));
        arr1.add(textNode("a1"));
        ObjectNode body2 = m.createObjectNode();
        ArrayNode arr2 = body2.putArray("response_batch");
        arr2.add(textNode("a2"));

        var subs = List.of(
                SubBatchResult.ok(body1, 2, 0),
                SubBatchResult.ok(body2, 1, 2));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertFalse(merged.allFailed());
        ArrayNode out = (ArrayNode) merged.body().get("response_batch");
        assertEquals(3, out.size());
        assertEquals("a0", out.get(0).get("v").asText());
        assertEquals("a2", out.get(2).get("v").asText());
        assertNull(merged.body().get("_partial_failure"));
    }

    @Test
    void partialFailurePadsFailedChunkAndAddsMetadata() {
        ObjectNode body1 = m.createObjectNode();
        body1.putArray("response_batch").add(textNode("a0")).add(textNode("a1"));

        var subs = List.of(
                SubBatchResult.ok(body1, 2, 0),
                SubBatchResult.failed(2, 2, "fe_timeout"));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertFalse(merged.allFailed());
        ArrayNode out = (ArrayNode) merged.body().get("response_batch");
        assertEquals(4, out.size());
        assertTrue(out.get(2).isNull());
        assertTrue(out.get(3).isNull());
        ObjectNode pf = (ObjectNode) merged.body().get("_partial_failure");
        assertEquals(2, pf.get("failed_count").asInt());
        assertEquals(4, pf.get("total_count").asInt());
        ArrayNode fi = (ArrayNode) pf.get("failed_indices");
        assertEquals(2, fi.get(0).asInt());
        assertEquals(3, fi.get(1).asInt());
    }

    @Test
    void allFailedFlagsAllFailedNoEnvelopeRequired() {
        var subs = List.of(
                SubBatchResult.failed(2, 0, "fe_down"),
                SubBatchResult.failed(2, 2, "fe_down"));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertTrue(merged.allFailed());
    }

    @Test
    void openAiSpecPadsFailedItemsWithErrorObjects() {
        BatchEndpointSpec spec = new BatchEndpointSpec(
                "/v1/batch/chat/completions", "requests", "responses",
                FailedItemFactory.OPENAI_ERROR, null);
        ObjectNode body1 = m.createObjectNode();
        ArrayNode r = body1.putArray("responses");
        ObjectNode ok = m.createObjectNode();
        ok.put("id", "ok0");
        r.add(ok);
        var subs = List.of(
                SubBatchResult.ok(body1, 1, 0),
                SubBatchResult.failed(1, 1, "fe_5xx"));
        MergedResponse merged = PartialFailureMerger.merge(subs, spec, m);

        ArrayNode out = (ArrayNode) merged.body().get("responses");
        assertEquals(2, out.size());
        assertEquals("ok0", out.get(0).get("id").asText());
        assertEquals(1, out.get(1).get("index").asInt());
        assertEquals("fe_5xx", out.get(1).get("error").get("message").asText());
    }

    @Test
    void successWithNonObjectBodyTreatedAsFailed() {
        var subs = List.of(SubBatchResult.ok(m.createArrayNode(), 2, 0));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertTrue(merged.allFailed());
        assertEquals(List.of(0, 1), merged.failedIndices());
    }

    @Test
    void successWithMissingResponseArrayIsPaddedAndReportedFailed() {
        ObjectNode body1 = m.createObjectNode();
        body1.putArray("response_batch").add(textNode("a0")).add(textNode("a1"));
        ObjectNode bodyNoArr = m.createObjectNode();
        bodyNoArr.put("note", "fe returned no response_batch");

        var subs = List.of(
                SubBatchResult.ok(body1, 2, 0),
                SubBatchResult.ok(bodyNoArr, 1, 2));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        ArrayNode out = (ArrayNode) merged.body().get("response_batch");
        assertEquals(3, out.size());
        assertTrue(out.get(2).isNull());
        ObjectNode pf = (ObjectNode) merged.body().get("_partial_failure");
        assertEquals(1, pf.get("failed_count").asInt());
        assertEquals(3, pf.get("total_count").asInt());
        assertEquals(2, ((ArrayNode) pf.get("failed_indices")).get(0).asInt());
    }

    @Test
    void successWithWrongLengthArrayIsPaddedAndReportedFailed() {
        ObjectNode body1 = m.createObjectNode();
        body1.putArray("response_batch").add(textNode("a0"));

        var subs = List.of(SubBatchResult.ok(body1, 2, 0));
        MergedResponse merged = PartialFailureMerger.merge(subs, batchInferSpec, m);

        assertTrue(merged.allFailed());
        assertEquals(List.of(0, 1), merged.failedIndices());
    }

    private ObjectNode textNode(String v) {
        ObjectNode n = m.createObjectNode();
        n.put("v", v);
        return n;
    }
}
