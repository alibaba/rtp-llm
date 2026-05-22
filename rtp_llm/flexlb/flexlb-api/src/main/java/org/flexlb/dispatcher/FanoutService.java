package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
public class FanoutService {

    private final FeClient feClient;
    private final FePool fePool;
    private final ObjectMapper mapper;
    private final int subBatchSize;

    /**
     * Split prompts into K-sized chunks, POST each to one FE concurrently, and merge in order.
     * A chunk whose FE call errors — or whose FE pick from the pool fails — becomes a
     * {@link SubBatchResult#failed} and never aborts its siblings (ft_proxy semantics). The
     * returned {@link MergedResponse} reports how many chunks succeeded so the caller can 200
     * on partial success and 500 only when all chunks failed.
     */
    public Mono<MergedResponse> dispatch(List<String> prompts, JsonNode generateConfig) {
        List<List<String>> chunks = BatchSplitter.split(prompts, subBatchSize);
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunks.size());
        for (List<String> chunk : chunks) {
            ObjectNode body = mapper.createObjectNode();
            body.set(DispatchProtocol.FIELD_PROMPT_BATCH, mapper.valueToTree(chunk));
            if (generateConfig != null) {
                body.set(DispatchProtocol.FIELD_GENERATE_CONFIG, generateConfig);
            }
            int chunkSize = chunk.size();
            calls.add(Mono.fromCallable(fePool::next)
                    .flatMap(feUrl -> feClient.postBatch(feUrl, body)
                            .map(resp -> SubBatchResult.ok(resp, chunkSize))
                            .onErrorResume(e -> {
                                log.warn("FE sub-batch failed: url={}, size={}", feUrl, chunkSize, e);
                                return Mono.just(SubBatchResult.failed(chunkSize));
                            }))
                    .onErrorResume(e -> {
                        log.warn("FE pick failed for chunk size={}", chunkSize, e);
                        return Mono.just(SubBatchResult.failed(chunkSize));
                    }));
        }
        // mergeSequential dispatches concurrently but collects results in chunk order.
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.parallel())
                .map(subs -> ResponseMerger.merge(subs, mapper));
    }
}
