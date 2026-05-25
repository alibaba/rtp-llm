package org.flexlb.dispatcher;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Hard-coded spec table for the dispatcher's batch endpoints.
 *
 * <p><b>Why {@code /} is not listed:</b> {@code POST /} triggers batch mode only when the request
 * body carries {@code prompt_batch} (see {@code request_extractor.py:162-163}), in which case its
 * wire shape is identical to {@code /batch_infer}. Sending {@code prompt: [...]} to {@code /} is a
 * footgun &mdash; {@code frontend_worker.py:425-430} unwraps to {@code batch[0]} on the
 * non-{@code batch_infer} branch, so the engine processes all prompts but returns only the first
 * result. We don't want the dispatcher to normalize that pattern; add {@code /} once V12 settles
 * whether to alias it to {@code /batch_infer} or treat it as passthrough.
 *
 * <p>Embedding variants ({@code /v1/embeddings/dense|sparse|colbert|similarity},
 * {@code /v1/reranker}, {@code /v1/classifier}) share the embedding response shape but use
 * different request fields. Add them after V10, one row each.
 */
@Configuration
public class BatchEndpointRegistry {

    @Bean
    public List<BatchEndpointSpec> batchSpecs() {
        return List.of(
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
    }

    /**
     * Path &rarr; spec lookup table for the router. {@link Collectors#toUnmodifiableMap} throws at
     * startup if two specs share a path, turning a typo in {@link #batchSpecs()} into a clear
     * Spring bean-init failure instead of a downstream stream-collector explosion.
     */
    @Bean
    public Map<String, BatchEndpointSpec> batchSpecsByPath(List<BatchEndpointSpec> batchSpecs) {
        return batchSpecs.stream()
                .collect(Collectors.toUnmodifiableMap(BatchEndpointSpec::getPath, Function.identity()));
    }
}
