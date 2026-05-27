package org.flexlb.dispatcher;

import org.flexlb.dispatcher.FanoutService.SubBatchResult;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * Cross-chunk aggregation for OpenAI embedding-shaped responses. After
 * {@link PartialFailureMerger} has stitched the {@code data} array (including any
 * {@link FailedItemFactory#EMBEDDING_NULL} placeholders at absolute failed indices), this
 * post-merger renumbers each item's {@code index} to its absolute offset in the merged array
 * and sums {@code usage.prompt_tokens} / {@code usage.total_tokens} across all successful
 * sub-bodies. Failed sub-batches contribute zero to {@code usage}; their pre-assigned absolute
 * indices are preserved (the unconditional renumbering writes back the same value).
 */
public final class EmbeddingPostMerger implements BatchEndpointSpec.PostMerger {

    public static final EmbeddingPostMerger INSTANCE = new EmbeddingPostMerger();

    @Override
    public void apply(ObjectNode mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices, ObjectMapper mapper) {
        if (mergedBody.get("data") instanceof ArrayNode data) {
            for (int i = 0; i < data.size(); i++) {
                if (data.get(i) instanceof ObjectNode on) {
                    on.put("index", i);
                }
            }
        }
        long promptTokens = 0;
        long totalTokens = 0;
        for (SubBatchResult s : subs) {
            if (!s.isSuccess() || s.body() == null) {
                continue;
            }
            JsonNode usage = s.body().get("usage");
            if (usage == null) {
                continue;
            }
            promptTokens += usage.path("prompt_tokens").asLong(0);
            totalTokens += usage.path("total_tokens").asLong(0);
        }
        boolean envelopeHasUsage = mergedBody.get("usage") instanceof ObjectNode;
        if (envelopeHasUsage || promptTokens > 0 || totalTokens > 0) {
            ObjectNode u = mergedBody.get("usage") instanceof ObjectNode existing
                    ? existing
                    : mergedBody.putObject("usage");
            u.put("prompt_tokens", promptTokens);
            u.put("total_tokens", totalTokens);
        }
    }
}
