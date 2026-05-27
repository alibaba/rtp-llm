package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Value;
import org.flexlb.dispatcher.FanoutService.SubBatchResult;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Spec for one batch endpoint the dispatcher fans out to. Also the home of the two SPI types
 * a spec carries — {@link PostMerger} (cross-chunk aggregation) and {@link FailedItemFactory}
 * (per-item placeholder shape) — plus the hardcoded {@link #SPECS} table.
 *
 * <p>Bare {@code POST /} aliases {@code /batch_infer} semantics — rtp_llm FE historically
 * exposes batch generation on the root path and accepts the same {@code prompt_batch} /
 * {@code response_batch} wire shape. Embedding variants ({@code /v1/embeddings/dense|sparse|
 * colbert|similarity}, {@code /v1/reranker}, {@code /v1/classifier}) share the embedding
 * response shape but use different request fields — add them here, one row each.
 *
 * <p>The dispatcher is array-only by design. {@link GenericBatchHandler} hard-rejects (400) any
 * request whose {@code requestArrayField} is missing or not a JSON array — single-item callers
 * should hit FE directly.
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
     * Cross-chunk aggregation hook; runs after {@link PartialFailureMerger} has stitched the
     * response array.
     */
    @FunctionalInterface
    public interface PostMerger {
        void apply(ObjectNode mergedBody,
                   List<SubBatchResult> subs,
                   List<Integer> failedIndices,
                   ObjectMapper mapper);
    }

    /** Builds a per-item failure placeholder at the absolute batch index. */
    @FunctionalInterface
    public interface FailedItemFactory {
        JsonNode build(int absoluteIndex, String reason, ObjectMapper mapper);

        FailedItemFactory NULL = (idx, reason, mapper) -> mapper.nullNode();

        FailedItemFactory OPENAI_ERROR = (idx, reason, mapper) -> {
            ObjectNode err = mapper.createObjectNode();
            err.put("code", "dispatcher_sub_batch_failed");
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

    /**
     * Hardcoded spec table for the dispatcher's batch endpoints.
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
                    FailedItemFactory.EMBEDDING_NULL, EmbeddingPostMerger.INSTANCE)
    );

    /**
     * Path-lookup table; {@link Collectors#toUnmodifiableMap} throws at class-load if
     * {@link #SPECS} contains duplicate paths — a typo in the table becomes a clear startup
     * failure instead of a stream-collector explosion later.
     */
    public static final Map<String, BatchEndpointSpec> BY_PATH = SPECS.stream()
            .collect(Collectors.toUnmodifiableMap(BatchEndpointSpec::getPath, Function.identity()));
}
