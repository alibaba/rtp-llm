package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
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
 * <p>Bare {@code POST /} aliases {@code /batch_infer} semantics — rtp_llm FE historically
 * exposes batch generation on the root path and accepts the same {@code prompt_batch} /
 * {@code response_batch} wire shape. Embedding variants
 * ({@code /v1/embeddings/dense|sparse|colbert|similarity}, {@code /v1/reranker},
 * {@code /v1/classifier}) share the embedding response shape but use different request
 * fields — add them here, one row each.
 */
@Value
public class BatchEndpointSpec {

    /**
     * Request array field of the rtp_llm generation endpoints (root {@code /} and
     * {@code /batch_infer}). Load-bearing beyond the {@link #SPECS} rows: {@link
     * #requiresWholeBody} keys off it to scope the sample-aligned companion-field check to
     * exactly these endpoints.
     */
    public static final String PROMPT_BATCH_FIELD = "prompt_batch";

    String path;
    String requestArrayField;
    String responseArrayField;
    FailedItemFactory failedItemFactory;
    /** May be null when an endpoint has no cross-chunk aggregation. */
    PostMerger postMerger;
    /**
     * When true, outbound chunk bodies serialize with
     * {@link com.alibaba.fastjson2.JSONWriter.Feature#WriteNulls} so user-supplied null entries
     * (e.g. {@code "tools": null}) reach FE byte-for-byte. The dispatcher itself never adds
     * nulls to a chunk body — they only come from the input request — so this matters only when
     * FE pydantic distinguishes "field absent" from "field null" (rare in rtp_llm FE: pydantic
     * Optional defaults to None and treats both the same). Costs ~18% on the chunk serialize
     * step (measured by {@code SerializeMicroBench}), which on a 100-chunk fanout request is
     * ~160µs CPU per request. Default false to take that win on the common wire shape; flip
     * per-endpoint above for any endpoint where wire compat matters.
     */
    boolean fanoutWriteNulls;
    /**
     * When true, the array field only counts as a batch if every element is a string.
     * {@code /v1/embeddings} needs this: FE's {@code input} union also admits a single
     * multimodal/chat input expressed as {@code List[ContentPart]} / {@code List[ChatMessage]}
     * — an array of JSON objects that is ONE input and must not be split per element.
     * Endpoints whose batch items are legitimately objects (e.g. {@code requests} on
     * {@code /v1/batch/chat/completions}) keep this false.
     */
    boolean splitRequiresStringItems;

    /**
     * Whether the array field is batch-shaped for this endpoint and may be split into
     * chunks; non-splittable bodies are passthrough-forwarded whole.
     */
    public boolean canSplit(JSONArray arr) {
        if (!splitRequiresStringItems) {
            return true;
        }
        for (Object item : arr) {
            if (!(item instanceof String)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether this body must be forwarded whole instead of split, because it carries a
     * companion field that FE positionally aligns to the request array but the dispatcher does
     * not slice.
     *
     * <p>Only the {@code prompt_batch} endpoints (root {@code /} and {@code /batch_infer}) carry
     * such companions. FE's root {@code /} handler validates top-level {@code images}/{@code
     * urls} (each a {@code list[list]} indexed by prompt) and a list-form {@code
     * generate_config.adapter_name} against the prompt count
     * ({@code request_extractor._get_urls} / {@code _get_adapter}). A split chunk would carry the
     * full-length companion against a shorter prompt slice, so FE would reject every chunk.
     * Forwarding the intact body to one FE keeps such requests correct, at the cost of fanout.
     */
    public boolean requiresWholeBody(JSONObject body) {
        if (!PROMPT_BATCH_FIELD.equals(requestArrayField)) {
            return false;
        }
        if (body.get("images") != null || body.get("urls") != null) {
            return true;
        }
        if (!(body.get("generate_config") instanceof JSONObject gc)) {
            return false;
        }
        // A scalar (String) adapter_name applies to the whole batch and is safe to split; a
        // list-form adapter_name is one entry per prompt and FE rejects a length mismatch.
        Object adapter = gc.get("adapter_name");
        return adapter != null && !(adapter instanceof String);
    }

    /**
     * Cross-chunk aggregation hook; runs after {@link ResponseMerger} has stitched
     * the response array.
     */
    @FunctionalInterface
    public interface PostMerger {
        void apply(JSONObject mergedBody,
                   List<SubBatchResult> subs,
                   List<Integer> failedIndices,
                   BatchEndpointSpec spec);
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
     * Hardcoded spec table for the dispatcher's batch endpoints.
     */
    public static final List<BatchEndpointSpec> SPECS = List.of(
            new BatchEndpointSpec("/",
                    PROMPT_BATCH_FIELD, "response_batch",
                    FailedItemFactory.NULL, null, false, false),
            new BatchEndpointSpec("/batch_infer",
                    PROMPT_BATCH_FIELD, "response_batch",
                    FailedItemFactory.NULL, null, false, false),
            new BatchEndpointSpec("/v1/batch/chat/completions",
                    "requests", "responses",
                    FailedItemFactory.OPENAI_ERROR, null, true, false),
            new BatchEndpointSpec("/v1/embeddings",
                    "input", "data",
                    FailedItemFactory.EMBEDDING_NULL, EmbeddingMerger.INSTANCE, true, true)
    );

    /**
     * Path-lookup table; {@link Collectors#toUnmodifiableMap} throws at class-load if
     * {@link #SPECS} contains duplicate paths so a typo in the table becomes a clear startup
     * failure instead of a stream-collector explosion later.
     */
    public static final Map<String, BatchEndpointSpec> BY_PATH = SPECS.stream()
            .collect(Collectors.toUnmodifiableMap(BatchEndpointSpec::getPath, Function.identity()));
}
