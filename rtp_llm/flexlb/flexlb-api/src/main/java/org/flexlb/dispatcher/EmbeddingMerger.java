package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.util.List;

/** Rebase embedding indices and add per-chunk usage after the ordered merge. */
public final class EmbeddingMerger {

    private EmbeddingMerger() {}

    public static void merge(JSONObject mergedBody, List<SubBatchResult> subs, JSONObject originalRequest) {
        BatchEndpointSpec spec = BatchEndpointSpec.EMBEDDING;
        if (mergedBody.getString("object") == null) {
            mergedBody.put("object", "list");
        }
        if (mergedBody.getString("model") == null) {
            String model = originalRequest == null ? null : originalRequest.getString("model");
            mergedBody.put("model", model == null ? "" : model);
        }
        JSONArray data = mergedBody.getJSONArray(spec.getResponseArrayField());
        if (data != null) {
            for (int i = 0; i < data.size(); i++) {
                Object item = data.get(i);
                if (item instanceof JSONObject on) {
                    on.put("index", i);
                }
            }
        }
        long promptTokens = 0;
        long totalTokens = 0;
        for (SubBatchResult s : subs) {
            if (!ResponseMerger.wellFormed(s, spec)) {
                continue;
            }
            JSONObject usage = s.body().getJSONObject("usage");
            if (usage == null) {
                continue;
            }
            promptTokens += usage.getLongValue("prompt_tokens", 0);
            totalTokens += usage.getLongValue("total_tokens", 0);
        }
        JSONObject u = mergedBody.getJSONObject("usage");
        if (u == null) {
            u = new JSONObject();
            mergedBody.put("usage", u);
        }
        u.put("prompt_tokens", promptTokens);
        u.put("total_tokens", totalTokens);
    }
}
