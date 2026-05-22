package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

public final class ResponseMerger {

    private ResponseMerger() {
    }

    /**
     * Stitch sub-batches back together in order. A failed sub-batch — or a successful one whose
     * {@code response_batch} length doesn't match its chunk size — is padded with placeholders so
     * the merged batch always has exactly sum(chunkSize) entries in the original order.
     */
    public static MergedResponse merge(List<SubBatchResult> subResults, ObjectMapper mapper) {
        ArrayNode merged = mapper.createArrayNode();
        int succeededChunks = 0;
        int succeededPrompts = 0;
        for (SubBatchResult sub : subResults) {
            JsonNode arr = sub.isSuccess() ? sub.response().get("response_batch") : null;
            if (arr != null && arr.isArray() && arr.size() == sub.chunkSize()) {
                merged.addAll((ArrayNode) arr);
                succeededChunks++;
                succeededPrompts += sub.chunkSize();
            } else {
                for (int i = 0; i < sub.chunkSize(); i++) {
                    merged.add(placeholder(mapper));
                }
            }
        }
        ObjectNode body = mapper.createObjectNode();
        body.set("response_batch", merged);
        return new MergedResponse(body, succeededChunks, subResults.size(), succeededPrompts);
    }

    private static ObjectNode placeholder(ObjectMapper mapper) {
        ObjectNode p = mapper.createObjectNode();
        p.put("response", "");
        p.put("finished", true);
        return p;
    }
}
