package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

public class FanoutService {

    private final FeClient feClient;
    private final FePool fePool;
    private final ObjectMapper mapper;
    private final int subBatchSize;

    public FanoutService(FeClient feClient, FePool fePool, ObjectMapper mapper, int subBatchSize) {
        this.feClient = feClient;
        this.fePool = fePool;
        this.mapper = mapper;
        this.subBatchSize = subBatchSize;
    }

    /**
     * Split prompts into K-sized chunks, POST each to one FE concurrently, and merge in order.
     * A chunk whose FE call errors becomes a {@link SubBatchResult#failed} — it never aborts its
     * siblings (ft_proxy semantics). The returned {@link MergedResponse} reports how many chunks
     * succeeded so the caller can 200 on partial success and 500 only when all chunks failed.
     */
    public Mono<MergedResponse> dispatch(List<String> prompts, JsonNode generateConfig) {
        List<List<String>> chunks = BatchSplitter.split(prompts, subBatchSize);
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunks.size());
        for (List<String> chunk : chunks) {
            ObjectNode body = mapper.createObjectNode();
            body.set("prompt_batch", mapper.valueToTree(chunk));
            if (generateConfig != null) {
                body.set("generate_config", generateConfig);
            }
            int chunkSize = chunk.size();
            calls.add(feClient.postBatch(fePool.next(), body)
                    .map(resp -> SubBatchResult.ok(resp, chunkSize))
                    .onErrorResume(e -> Mono.just(SubBatchResult.failed(chunkSize))));
        }
        // mergeSequential dispatches concurrently but collects results in chunk order.
        // publishOn(boundedElastic) moves the synchronous merge (JsonNode walk + ArrayNode addAll)
        // OFF the Reactor-Netty event loop, so a 5-MB N=500 merge can't add tail latency to the
        // co-located Master /schedule (1-5 ms SLA) sharing the same loop.
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.boundedElastic())
                .map(subs -> ResponseMerger.merge(subs, mapper));
    }
}
