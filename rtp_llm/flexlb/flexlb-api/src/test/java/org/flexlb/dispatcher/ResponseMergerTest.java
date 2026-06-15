package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ResponseMergerTest {

    private static final BatchEndpointSpec BATCH_INFER = BatchEndpointSpec.BY_PATH.get("/batch_infer");
    private static final BatchEndpointSpec OPENAI_BATCH =
            BatchEndpointSpec.BY_PATH.get("/v1/batch/chat/completions");

    @Test
    void happyPathStitchesAllChunks() {
        SubBatchResult s0 = SubBatchResult.ok(envelopeBatchInfer("r0", "r1"), 2, 0);
        SubBatchResult s1 = SubBatchResult.ok(envelopeBatchInfer("r2", "r3"), 2, 2);
        ResponseMerger.MergedResponse merged = ResponseMerger.merge(List.of(s0, s1), BATCH_INFER);

        assertEquals(2, merged.succeededChunks());
        assertEquals(2, merged.totalChunks());
        assertTrue(merged.failedIndices().isEmpty());
        assertFalse(merged.allFailed());
        JSONArray out = merged.body().getJSONArray("response_batch");
        assertEquals(4, out.size());
        assertEquals("r0", out.getString(0));
        assertEquals("r3", out.getString(3));
        assertNull(merged.body().getJSONObject("_partial_failure"));
    }

    @Test
    void partialFailureFillsPlaceholderAtAbsoluteIndex() {
        SubBatchResult s0 = SubBatchResult.ok(envelopeBatchInfer("r0", "r1"), 2, 0);
        SubBatchResult s1 = SubBatchResult.failed(2, 2, "timeout");
        SubBatchResult s2 = SubBatchResult.ok(envelopeBatchInfer("r4", "r5"), 2, 4);
        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(s0, s1, s2), BATCH_INFER);

        assertEquals(2, merged.succeededChunks());
        assertEquals(List.of(2, 3), merged.failedIndices());
        JSONArray out = merged.body().getJSONArray("response_batch");
        assertEquals(6, out.size());
        assertNull(out.get(2));
        assertNull(out.get(3));
        JSONObject pf = merged.body().getJSONObject("_partial_failure");
        assertEquals(2, pf.getIntValue("failed_count"));
        assertEquals(6, pf.getIntValue("total_count"));
        assertEquals(List.of(2, 3), pf.getJSONArray("failed_indices").toJavaList(Integer.class));
    }

    @Test
    void allFailedReturnsEmptyEnvelopeWithAllIndices() {
        SubBatchResult s0 = SubBatchResult.failed(2, 0, "no_route");
        SubBatchResult s1 = SubBatchResult.failed(3, 2, "no_route");
        ResponseMerger.MergedResponse merged = ResponseMerger.merge(List.of(s0, s1), BATCH_INFER);

        assertTrue(merged.allFailed());
        assertEquals(0, merged.succeededChunks());
        assertEquals(2, merged.totalChunks());
        assertEquals(List.of(0, 1, 2, 3, 4), merged.failedIndices());
        assertEquals(List.of("no_route", "no_route"), merged.failedReasons());
        // No FE HTTP status available (transport-level failures) -> server error.
        assertEquals(500, merged.errorStatus());
    }

    @Test
    void allFailedSurfacesSharedFourxxStatus() {
        SubBatchResult s0 = SubBatchResult.failed(2, 0, "bad_request", 400);
        SubBatchResult s1 = SubBatchResult.failed(2, 2, "bad_request", 400);
        ResponseMerger.MergedResponse merged = ResponseMerger.merge(List.of(s0, s1), BATCH_INFER);

        assertTrue(merged.allFailed());
        // Every sub-batch failed with the same client error: surface it, don't mask as 500.
        assertEquals(400, merged.errorStatus());
    }

    @Test
    void allFailedWithMixedOrServerStatusFallsTo500() {
        // Mixed 4xx is ambiguous -> 500.
        ResponseMerger.MergedResponse mixed = ResponseMerger.merge(
                List.of(SubBatchResult.failed(1, 0, "bad", 400),
                        SubBatchResult.failed(1, 1, "conflict", 409)), BATCH_INFER);
        assertEquals(500, mixed.errorStatus());

        // Any non-4xx (transport failure with no status, or an FE 5xx) -> 500.
        ResponseMerger.MergedResponse server = ResponseMerger.merge(
                List.of(SubBatchResult.failed(1, 0, "no_route"),
                        SubBatchResult.failed(1, 1, "fe_500", 500)), BATCH_INFER);
        assertEquals(500, server.errorStatus());
    }

    @Test
    void openaiPartialFailureUsesErrorShapePlaceholder() {
        SubBatchResult s0 = SubBatchResult.ok(envelopeOpenai("ok-0"), 1, 0);
        SubBatchResult s1 = SubBatchResult.failed(1, 1, "timeout");
        ResponseMerger.MergedResponse merged = ResponseMerger.merge(List.of(s0, s1), OPENAI_BATCH);

        JSONArray out = merged.body().getJSONArray("responses");
        assertEquals(2, out.size());
        JSONObject placeholder = (JSONObject) out.get(1);
        assertEquals(1, placeholder.getIntValue("index"));
        assertEquals("timeout", placeholder.getJSONObject("error").getString("message"));
    }

    @Test
    void malformedSuccessfulSubFallsToFailurePath() {
        JSONObject malformed = new JSONObject();
        malformed.put("response_batch", "not-an-array");
        SubBatchResult s0 = SubBatchResult.ok(malformed, 2, 0);
        SubBatchResult s1 = SubBatchResult.ok(envelopeBatchInfer("r2", "r3"), 2, 2);
        ResponseMerger.MergedResponse merged = ResponseMerger.merge(List.of(s0, s1), BATCH_INFER);

        assertEquals(1, merged.succeededChunks());
        assertEquals(List.of(0, 1), merged.failedIndices());
        assertEquals(List.of("malformed_sub_batch"), merged.failedReasons());
    }

    private static JSONObject envelopeBatchInfer(String... items) {
        JSONObject body = new JSONObject();
        JSONArray arr = new JSONArray();
        for (String it : items) {
            arr.add(it);
        }
        body.put("response_batch", arr);
        return body;
    }

    private static JSONObject envelopeOpenai(String marker) {
        JSONObject body = new JSONObject();
        JSONArray arr = new JSONArray();
        JSONObject item = new JSONObject();
        item.put("index", 0);
        item.put("content", marker);
        arr.add(item);
        body.put("responses", arr);
        return body;
    }
}
