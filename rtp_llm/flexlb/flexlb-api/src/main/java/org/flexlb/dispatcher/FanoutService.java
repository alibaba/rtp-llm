package org.flexlb.dispatcher;

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

    /**
     * POST each pre-built chunk body to one FE concurrently and return the per-chunk outcomes in
     * the original order. Each entry of {@code chunkBodies} is already the full FE request — the
     * caller (handler) is responsible for splitting the input array, deep-copying the envelope,
     * and replacing {@code spec.requestArrayField} with the chunk slice before calling here.
     *
     * <p>A chunk whose FE call errors — or whose FE pick from the pool fails — becomes a
     * {@link SubBatchResult#failed} and never aborts its siblings (ft_proxy semantics). The
     * handler is responsible for calling {@link PartialFailureMerger} on the returned list.
     */
    public Mono<List<SubBatchResult>> dispatchChunks(String fePath, List<ObjectNode> chunkBodies,
                                                     BatchEndpointSpec spec) {
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunkBodies.size());
        int start = 0;
        for (ObjectNode body : chunkBodies) {
            int chunkSize = body.get(spec.getRequestArrayField()).size();
            int startIndex = start;
            calls.add(Mono.fromCallable(fePool::next)
                    .flatMap(feUrl -> feClient.post(feUrl, fePath, body)
                            .map(resp -> SubBatchResult.ok(resp, chunkSize, startIndex))
                            .onErrorResume(e -> {
                                log.warn("FE chunk failed: url={}, path={}, size={}", feUrl, fePath, chunkSize, e);
                                return Mono.just(SubBatchResult.failed(chunkSize, startIndex, briefReason(e)));
                            }))
                    .onErrorResume(e -> {
                        log.warn("FE pick failed for chunk size={}", chunkSize, e);
                        return Mono.just(SubBatchResult.failed(chunkSize, startIndex, briefReason(e)));
                    }));
            start += chunkSize;
        }
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.parallel());
    }

    private static String briefReason(Throwable e) {
        String m = e.getClass().getSimpleName();
        return e.getMessage() == null ? m : m + ": " + e.getMessage();
    }
}
