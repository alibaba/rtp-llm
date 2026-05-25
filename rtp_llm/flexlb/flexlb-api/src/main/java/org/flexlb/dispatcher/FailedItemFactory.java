package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

@FunctionalInterface
public interface FailedItemFactory {
    JsonNode build(int absoluteIndex, String reason, ObjectMapper mapper);

    FailedItemFactory NULL = (idx, reason, mapper) -> mapper.nullNode();

    FailedItemFactory OPENAI_ERROR = (idx, reason, mapper) -> {
        ObjectNode err = mapper.createObjectNode();
        err.put("code", DispatchProtocol.ERROR_CODE_SUB_BATCH_FAILED);
        err.put("message", reason);
        ObjectNode item = mapper.createObjectNode();
        item.put("index", idx);
        item.set("error", err);
        return item;
    };

    FailedItemFactory EMBEDDING_NULL = (idx, reason, mapper) -> {
        ObjectNode item = mapper.createObjectNode();
        item.put("index", idx);
        item.set("embedding", mapper.nullNode());
        item.put("error", reason);
        return item;
    };
}
