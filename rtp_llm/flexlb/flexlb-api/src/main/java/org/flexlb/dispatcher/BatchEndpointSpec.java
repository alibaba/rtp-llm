package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/** Wire contracts of the FE endpoints supported by batch fanout. */
@Getter
@RequiredArgsConstructor
public enum BatchEndpointSpec {
    ROOT("/", "prompt_batch", "response_batch"),
    BATCH_INFER("/batch_infer", "prompt_batch", "response_batch"),
    CHAT("/v1/batch/chat/completions", "requests", "responses"),
    EMBEDDING("/v1/embeddings", "input", "data"),
    RERANKER("/v1/reranker", "documents", "results");

    public static final String PROMPT_BATCH_FIELD = "prompt_batch";
    public static final List<BatchEndpointSpec> SPECS = List.of(values());
    public static final Map<String, BatchEndpointSpec> BY_PATH = SPECS.stream()
            .collect(Collectors.toUnmodifiableMap(BatchEndpointSpec::getPath, Function.identity()));

    private final String path;
    private final String requestArrayField;
    private final String responseArrayField;

    public boolean isPreAssignable() {
        return this == ROOT || this == BATCH_INFER;
    }

    public boolean isFailOnPartialFailure() {
        return this == RERANKER;
    }

    public boolean isSplittableBatch(JSONObject body, JSONArray arr) {
        return arr != null && canSplit(arr) && !requiresWholeBody(body);
    }

    /** Embedding arrays of content parts represent one multimodal input, not a batch. */
    public boolean canSplit(JSONArray arr) {
        if (this == EMBEDDING || this == RERANKER) {
            for (Object item : arr) {
                if (!(item instanceof String)) {
                    return false;
                }
            }
        }
        return true;
    }

    public String validateForFanout(JSONObject body) {
        return this == RERANKER ? RerankerMerger.validate(body) : null;
    }

    public void prepareChunkBody(JSONObject body) {
        if (this == RERANKER) {
            RerankerMerger.prepare(body);
        }
    }

    public void finishMerge(JSONObject body, List<SubBatchResult> subs,
                            List<Integer> failedIndices, JSONObject originalRequest) {
        switch (this) {
            case EMBEDDING -> EmbeddingMerger.merge(body, subs, originalRequest);
            case RERANKER -> RerankerMerger.merge(body, subs, failedIndices, originalRequest);
            default -> { }
        }
    }

    public Object failedItem(int index, String reason) {
        return switch (this) {
            case CHAT -> JSONObject.of("index", index, "error", JSONObject.of(
                    "code", "dispatcher_sub_batch_failed", "message", reason));
            case EMBEDDING -> JSONObject.of("index", index, "embedding", null, "error", reason);
            default -> null;
        };
    }

    /** FE aligns companions to the entire prompt batch and streams SSE without JSON buffering. */
    public boolean requiresWholeBody(JSONObject body) {
        if (!PROMPT_BATCH_FIELD.equals(requestArrayField)) {
            return false;
        }
        if (body.get("images") != null || body.get("urls") != null) {
            return true;
        }
        if (requestsStreaming(body)) {
            return true;
        }
        // FE promotes recognized top-level config fields after reading nested config, so an
        // explicitly present top-level value wins even when it is JSON null. Preserve that exact
        // precedence: splitting a list would misalign it, while splitting a scalar could turn an
        // invalid multi-prompt request into several valid single-prompt requests.
        return effectiveAdapterName(body) != null;
    }

    /** Mirrors the raw-request streaming forms accepted by FE before config normalization. */
    private static boolean requestsStreaming(JSONObject body) {
        if (jsonTruthy(body.get("stream"))
                || jsonTruthy(body.get("yield_generator"))
                || jsonTruthy(body.get("is_streaming"))) {
            return true;
        }
        for (String key : List.of("generate_config", "generation_config")) {
            if (body.get(key) instanceof JSONObject gc
                    && (jsonTruthy(gc.get("yield_generator"))
                    || jsonTruthy(gc.get("is_streaming")))) {
                return true;
            }
        }
        return false;
    }

    private static JSONObject effectiveGenerateConfig(JSONObject body) {
        if (body.get("generate_config") instanceof JSONObject gc) {
            return gc;
        }
        return body.get("generation_config") instanceof JSONObject gc ? gc : null;
    }

    /** Mirrors FE's top-level-over-nested config precedence for {@code adapter_name}. */
    private static Object effectiveAdapterName(JSONObject body) {
        if (body.containsKey("adapter_name")) {
            return body.get("adapter_name");
        }
        JSONObject gc = effectiveGenerateConfig(body);
        return gc == null ? null : gc.get("adapter_name");
    }

    /** JSON values use the same truthiness rules as FE's Python request checks. */
    private static boolean jsonTruthy(Object value) {
        if (value == null) {
            return false;
        }
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof Number number) {
            return number.doubleValue() != 0.0;
        }
        if (value instanceof CharSequence chars) {
            return !chars.isEmpty();
        }
        if (value instanceof Collection<?> collection) {
            return !collection.isEmpty();
        }
        if (value instanceof Map<?, ?> map) {
            return !map.isEmpty();
        }
        return true;
    }

}
