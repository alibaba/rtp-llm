package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ResponseMergerTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonNode batch(String... responses) throws Exception {
        StringBuilder sb = new StringBuilder("{\"response_batch\":[");
        for (int i = 0; i < responses.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("{\"response\":\"").append(responses[i]).append("\",\"finished\":true}");
        }
        sb.append("]}");
        return mapper.readTree(sb.toString());
    }

    @Test
    void concatenatesSuccessfulSubBatchesInOrder() throws Exception {
        MergedResponse m = ResponseMerger.merge(
                List.of(SubBatchResult.ok(batch("a", "b"), 2), SubBatchResult.ok(batch("c"), 1)), mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(3, arr.size());
        assertEquals("a", arr.get(0).get("response").asText());
        assertEquals("c", arr.get(2).get("response").asText());
        assertEquals(2, m.succeededChunks());
        assertEquals(2, m.totalChunks());
    }

    @Test
    void failedSubBatchIsPaddedPreservingOrderAndSize() throws Exception {
        // chunk0 ok (2), chunk1 FAILED (size 1), chunk2 ok (1) -> 4 entries, slot 2 is a placeholder
        MergedResponse m = ResponseMerger.merge(
                List.of(SubBatchResult.ok(batch("a", "b"), 2),
                        SubBatchResult.failed(1),
                        SubBatchResult.ok(batch("d"), 1)),
                mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(4, arr.size());
        assertEquals("", arr.get(2).get("response").asText());
        assertTrue(arr.get(2).get("finished").asBoolean());
        assertEquals("d", arr.get(3).get("response").asText());
        assertEquals(2, m.succeededChunks());
        assertEquals(3, m.totalChunks());
        assertFalse(m.allFailed());
    }

    @Test
    void wrongLengthSubBatchIsTreatedAsFailedAndPadded() throws Exception {
        // claims chunk size 2 but FE returned only 1 entry -> padded to 2 placeholders, counted as failed
        MergedResponse m = ResponseMerger.merge(List.of(SubBatchResult.ok(batch("only"), 2)), mapper);
        JsonNode arr = m.body().get("response_batch");
        assertEquals(2, arr.size());
        assertEquals("", arr.get(0).get("response").asText());
        assertEquals(0, m.succeededChunks());
        assertTrue(m.allFailed());
    }

    @Test
    void emptyListYieldsEmptyBatch() {
        MergedResponse m = ResponseMerger.merge(List.of(), mapper);
        assertTrue(m.body().get("response_batch").isArray());
        assertEquals(0, m.body().get("response_batch").size());
        assertFalse(m.allFailed()); // nothing attempted is not a failure
    }
}
