package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

class RerankerMergerTest {

    private static final BatchEndpointSpec RERANKER =
            BatchEndpointSpec.BY_PATH.get("/v1/reranker");

    @Test
    void chunkRewriteDisablesLocalSortAndTopKWithoutMutatingOriginal() {
        JSONObject original = JSONObject.of(
                "query", "cape pants",
                "documents", JSONArray.of("d0", "d1", "d2"),
                "sorted", true,
                "top_k", 2,
                "return_documents", false);
        List<JSONArray> chunks = BatchChunkAssembler.splitArray(
                original.getJSONArray("documents"), 2);
        List<JSONObject> bodies = BatchChunkAssembler.buildChunkBodies(
                original, chunks, "documents");

        RERANKER.prepareChunkBodies(original, bodies);

        assertEquals(true, original.getBoolean("sorted"));
        assertEquals(2, original.getIntValue("top_k"));
        for (JSONObject body : bodies) {
            assertFalse(body.getBooleanValue("sorted"));
            assertFalse(body.containsKey("top_k"));
            assertFalse(body.getBooleanValue("return_documents"));
        }
    }

    @Test
    void rebasesIndicesStableSortsGloballyAppliesTopKAndSumsTokens() {
        SubBatchResult first = SubBatchResult.ok(rerankerBody(11,
                item(0, "d0", 0.2), item(1, "d1", 0.9)), 2, 0);
        SubBatchResult second = SubBatchResult.ok(rerankerBody(17,
                item(0, "d2", 0.9), item(1, "d3", 0.4)), 2, 2);
        JSONObject request = JSONObject.of("query", "q", "documents", JSONArray.of(
                "d0", "d1", "d2", "d3"), "top_k", 2);

        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(first, second), RERANKER, request);

        JSONArray results = merged.body().getJSONArray("results");
        assertEquals(2, results.size());
        // Equal scores retain original document order across chunk boundaries.
        assertEquals(1, results.getJSONObject(0).getIntValue("index"));
        assertEquals("d1", results.getJSONObject(0).getString("document"));
        assertEquals(2, results.getJSONObject(1).getIntValue("index"));
        assertEquals("d2", results.getJSONObject(1).getString("document"));
        assertEquals(28L, merged.body().getLongValue("total_tokens"));
    }

    @Test
    void sortedFalseKeepsDocumentOrderBeforeTopK() {
        SubBatchResult first = SubBatchResult.ok(rerankerBody(3,
                item(0, "d0", 0.1), item(1, "d1", 0.9)), 2, 0);
        SubBatchResult second = SubBatchResult.ok(rerankerBody(5,
                item(0, "d2", 0.8), item(1, "d3", 0.7)), 2, 2);
        JSONObject request = JSONObject.of("query", "q", "documents", JSONArray.of(
                "d0", "d1", "d2", "d3"), "sorted", false, "top_k", 3);

        JSONArray results = ResponseMerger.merge(List.of(first, second), RERANKER, request)
                .body().getJSONArray("results");

        assertEquals(3, results.size());
        assertEquals(List.of(0, 1, 2), results.stream()
                .map(value -> ((JSONObject) value).getIntValue("index")).toList());
    }

    @Test
    void negativeTopKMatchesPythonSliceSemantics() {
        SubBatchResult only = SubBatchResult.ok(rerankerBody(4,
                item(0, "d0", 4), item(1, "d1", 3),
                item(2, "d2", 2), item(3, "d3", 1)), 4, 0);
        JSONObject request = JSONObject.of("query", "q", "documents", JSONArray.of(
                "d0", "d1", "d2", "d3"), "top_k", -1);

        JSONArray results = ResponseMerger.merge(List.of(only), RERANKER, request)
                .body().getJSONArray("results");

        assertEquals(3, results.size(), "Python values[:-1] drops the final result");
    }

    @Test
    void partialFailureIsLeftForHandlerFailClosedPolicy() {
        SubBatchResult first = SubBatchResult.ok(rerankerBody(3,
                item(0, "d0", 0.1), item(1, "d1", 0.9)), 2, 0);
        SubBatchResult failed = SubBatchResult.failed(2, 2, "timeout");
        JSONObject request = JSONObject.of("query", "q", "documents", JSONArray.of(
                "d0", "d1", "d2", "d3"));

        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(first, failed), RERANKER, request);

        assertEquals(List.of(2, 3), merged.failedIndices());
        assertEquals(4, merged.totalItems());
        assertEquals(4, merged.body().getJSONArray("results").size());
        assertNull(merged.body().getJSONArray("results").get(2));
    }

    @Test
    void validatesFieldsWhoseSemanticsDispatcherRewrites() {
        assertNull(RERANKER.validateForFanout(JSONObject.of(
                "query", "q", "documents", JSONArray.of("d"), "top_k", 2.0)));
        assertEquals("query must be a string", RERANKER.validateForFanout(JSONObject.of(
                "documents", JSONArray.of("d"))));
        assertEquals("sorted must be a boolean", RERANKER.validateForFanout(JSONObject.of(
                "query", "q", "documents", JSONArray.of("d"), "sorted", "true")));
        assertEquals("top_k must be an integer or null", RERANKER.validateForFanout(JSONObject.of(
                "query", "q", "documents", JSONArray.of("d"), "top_k", 1.5)));
    }

    private static JSONObject rerankerBody(long totalTokens, JSONObject... items) {
        JSONObject body = new JSONObject();
        body.put("results", new JSONArray(List.of(items)));
        body.put("total_tokens", totalTokens);
        return body;
    }

    private static JSONObject item(int index, String document, double score) {
        return JSONObject.of("index", index, "document", document, "relevance_score", score);
    }
}
