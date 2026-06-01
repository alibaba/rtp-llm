package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import lombok.Value;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Per-endpoint spec for the dispatcher batch path. Carries the request / response array
 * field names and the two SPI hooks — {@link FailedItemFactory} for per-item failure
 * placeholders and {@link PostMerger} for cross-chunk aggregation — plus the hardcoded
 * {@link #SPECS} table.
 *
 * <p>Type-for-type mirror of {@code BatchEndpointSpec} on the Jackson side. The two specs
 * stay parallel so the equivalence test can run both handlers against the same input bodies
 * with the same endpoint configuration.
 *
 * <p>Bare {@code POST /} aliases {@code /batch_infer} semantics — rtp_llm FE historically
 * exposes batch generation on the root path and accepts the same {@code prompt_batch} /
 * {@code response_batch} wire shape. Embedding variants
 * ({@code /v1/embeddings/dense|sparse|colbert|similarity}, {@code /v1/reranker},
 * {@code /v1/classifier}) share the embedding response shape but use different request
 * fields — add them here, one row each.
 */
@Value
public class BatchEndpointSpec {
    String path;
    String requestArrayField;
    String responseArrayField;
    FailedItemFactory failedItemFactory;
    /** May be null when an endpoint has no cross-chunk aggregation. */
    PostMerger postMerger;

    /**
     * Cross-chunk aggregation hook; runs after {@link ResponseMerger} has stitched
     * the response array.
     */
    @FunctionalInterface
    public interface PostMerger {
        void apply(JSONObject mergedBody,
                   List<SubBatchResult> subs,
                   List<Integer> failedIndices);
    }

    /**
     * Builds a per-item failure placeholder at the absolute batch index. Returning {@code null}
     * is legal and means "store a JSON null at this position" — fastjson2's {@code JSONArray}
     * preserves a null slot when you {@code add(null)}.
     */
    @FunctionalInterface
    public interface FailedItemFactory {
        Object build(int absoluteIndex, String reason);

        FailedItemFactory NULL = (idx, reason) -> null;

        FailedItemFactory OPENAI_ERROR = (idx, reason) -> {
            JSONObject err = new JSONObject();
            err.put("code", "dispatcher_sub_batch_failed");
            err.put("message", reason);
            JSONObject item = new JSONObject();
            item.put("index", idx);
            item.put("error", err);
            return item;
        };

        FailedItemFactory EMBEDDING_NULL = (idx, reason) -> {
            JSONObject item = new JSONObject();
            item.put("index", idx);
            item.put("embedding", null);
            item.put("error", reason);
            return item;
        };
    }

    /**
     * Hardcoded spec table for the dispatcher's batch endpoints. Mirrors
     * {@code BatchEndpointSpec.SPECS} row-for-row.
     */
    public static final List<BatchEndpointSpec> SPECS = List.of(
            new BatchEndpointSpec("/",
                    "prompt_batch", "response_batch",
                    FailedItemFactory.NULL, null),
            new BatchEndpointSpec("/batch_infer",
                    "prompt_batch", "response_batch",
                    FailedItemFactory.NULL, null),
            new BatchEndpointSpec("/v1/batch/chat/completions",
                    "requests", "responses",
                    FailedItemFactory.OPENAI_ERROR, null),
            new BatchEndpointSpec("/v1/embeddings",
                    "input", "data",
                    FailedItemFactory.EMBEDDING_NULL, EmbeddingMerger.INSTANCE)
    );

    /**
     * Path-lookup table; {@link Collectors#toUnmodifiableMap} throws at class-load if
     * {@link #SPECS} contains duplicate paths so a typo in the table becomes a clear startup
     * failure instead of a stream-collector explosion later.
     */
    public static final Map<String, BatchEndpointSpec> BY_PATH = SPECS.stream()
            .collect(Collectors.toUnmodifiableMap(BatchEndpointSpec::getPath, Function.identity()));
}
