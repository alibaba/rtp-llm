package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class EmbeddingMergerTest {

    private static final BatchEndpointSpec EMBEDDINGS = BatchEndpointSpec.BY_PATH.get("/v1/embeddings");

    @Test
    void renumbersIndicesAndSumsUsage() {
        SubBatchResult s0 = SubBatchResult.ok(embeddingBody(2, 3, 5), 2, 0);
        SubBatchResult s1 = SubBatchResult.ok(embeddingBody(2, 7, 11), 2, 2);

        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(s0, s1), EMBEDDINGS);

        JSONArray data = merged.body().getJSONArray("data");
        assertEquals(4, data.size());
        for (int i = 0; i < 4; i++) {
            assertEquals(i, data.getJSONObject(i).getIntValue("index"));
        }
        JSONObject usage = merged.body().getJSONObject("usage");
        assertEquals(3 + 7, usage.getLongValue("prompt_tokens"));
        assertEquals(5 + 11, usage.getLongValue("total_tokens"));
    }

    @Test
    void partialFailureFailedChunksDoNotContributeToUsage() {
        SubBatchResult s0 = SubBatchResult.ok(embeddingBody(2, 3, 5), 2, 0);
        SubBatchResult s1 = SubBatchResult.failed(2, 2, "no_route");
        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(s0, s1), EMBEDDINGS);

        JSONArray data = merged.body().getJSONArray("data");
        assertEquals(4, data.size());
        assertEquals(0, data.getJSONObject(0).getIntValue("index"));
        assertEquals(1, data.getJSONObject(1).getIntValue("index"));
        assertEquals(2, data.getJSONObject(2).getIntValue("index"));
        assertEquals(3, data.getJSONObject(3).getIntValue("index"));
        assertEquals("no_route", data.getJSONObject(2).getString("error"));
        JSONObject usage = merged.body().getJSONObject("usage");
        assertEquals(3, usage.getLongValue("prompt_tokens"));
        assertEquals(5, usage.getLongValue("total_tokens"));
    }

    @Test
    void omitsUsageWhenNoSuccessfulSubsAndEnvelopeMissingUsage() {
        JSONObject envelopeNoUsage = new JSONObject();
        envelopeNoUsage.put("data", new JSONArray());
        SubBatchResult ok = SubBatchResult.ok(envelopeNoUsage, 0, 0);
        ResponseMerger.MergedResponse merged =
                ResponseMerger.merge(List.of(ok), EMBEDDINGS);
        // No incoming usage and no totals to add → don't fabricate one.
        assertEquals(null, merged.body().getJSONObject("usage"));
    }

    private static JSONObject embeddingBody(int items, long prompt, long total) {
        JSONObject body = new JSONObject();
        JSONArray data = new JSONArray();
        for (int i = 0; i < items; i++) {
            JSONObject item = new JSONObject();
            item.put("index", i);
            item.put("embedding", JSONArray.of(0.1, 0.2));
            data.add(item);
        }
        body.put("data", data);
        JSONObject usage = new JSONObject();
        usage.put("prompt_tokens", prompt);
        usage.put("total_tokens", total);
        body.put("usage", usage);
        return body;
    }
}
