package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RerankerMergerTest {
    @ParameterizedTest
    @CsvSource(delimiter = '|', textBlock = """
            true | 2 | [1,2]
            false | 3 | [0,1,2]
            true | -1 | [1,2,0]
            true | -9 | []
            true | 0 | []
            true | 99999999999999999999999 | [1,2,0,3]
            """)
    void sortingAndTopKFollowPythonSemantics(boolean sorted, String topK, String expected) {
        JSONObject request = JSON.parseObject("{\"sorted\":" + sorted + ",\"top_k\":" + topK + "}");
        JSONObject body = JSON.parseObject("""
                {"total_tokens":4,"results":[{"index":0,"relevance_score":-0.0},
                 {"index":1,"relevance_score":1},{"index":2,"relevance_score":1},
                 {"index":3,"relevance_score":0.0}]}
                """);
        JSONArray results = ResponseMerger.merge(List.of(SubBatchResult.ok(body, 4, 0)),
                BatchEndpointSpec.RERANKER, request).body().getJSONArray("results");
        assertEquals(JSON.parseArray(expected), results.stream().map(i -> ((JSONObject) i).get("index")).toList());
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', textBlock = """
            {} | query must be a string
            {"query":"q","sorted":"true"} | sorted must be a boolean
            {"query":"q","top_k":1.5} | top_k must be an integer or null
            {"query":"q","top_k":2.0} |
            """)
    void validatesRewrittenRequestFields(String json, String error) {
        assertEquals(error, RerankerMerger.validate(JSON.parseObject(json)));
    }

    @ParameterizedTest
    @CsvSource({"0.5,1", "0,0", "1,1", "-1,1", "2147483648,1"})
    void rejectsInvalidOrDuplicateChunkIndices(String first, String second) {
        JSONObject body = JSON.parseObject("{\"total_tokens\":2,\"results\":[{\"index\":" + first
                + ",\"relevance_score\":1},{\"index\":" + second + ",\"relevance_score\":1}]}");
        assertThrows(IllegalStateException.class, () -> ResponseMerger.merge(
                List.of(SubBatchResult.ok(body, 2, 0)), BatchEndpointSpec.RERANKER, new JSONObject()));
    }
}
