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
 * <p><b>Bare {@code POST /} is aliased to {@code /batch_infer} semantics</b> &mdash; rtp_llm FE
 * historically exposes batch generation on the root path and accepts the same
 * {@code prompt_batch} / {@code response_batch} wire shape. The dispatcher registers it under
 * {@code /dispatcher/} so deployments whose FE never wired a separate {@code /batch_infer} route
 * still get fanout. The known footgun applies to <em>callers</em>, not the dispatcher: sending
 * {@code prompt: [...]} (not {@code prompt_batch}) to {@code POST /} unwraps to
 * {@code batch[0]} on FE (see {@code frontend_worker.py:425-430}) and silently drops all but the
 * first result &mdash; clients hitting {@code /dispatcher/} must use {@code prompt_batch}.
 *
 * <p>Embedding variants ({@code /v1/embeddings/dense|sparse|colbert|similarity},
 * {@code /v1/reranker}, {@code /v1/classifier}) share the embedding response shape but use
 * different request fields. Add them after V10, one row each.
 *
 * <p><b>Dispatcher is array-only, by design.</b> {@link GenericBatchHandler} hard-rejects (400)
 * any request whose {@code requestArrayField} is missing or not a JSON array — this is what
 * neutralizes the {@code POST /} footgun above and matches ft_proxy's gating behavior. The one
 * place this is stricter than upstream is {@code /v1/embeddings}: the OpenAI spec accepts
 * {@code input} as either a string OR an array, but the dispatcher rejects the bare-string form.
 * Single-text embedding callers should hit FE directly; the dispatcher exists to fan batches out,
 * not to passthrough single items.
 */
@Configuration
public class BatchEndpointRegistry {

    @Bean
    public List<BatchEndpointSpec> batchSpecs() {
        return List.of(
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
