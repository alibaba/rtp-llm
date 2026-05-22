package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
@RequiredArgsConstructor
public class FanoutService {

    /**
     * One sub-batch outcome. Successful chunks carry the FE's response JSON in {@code body};
     * failed chunks carry a textual {@code reason}. {@code startIndex} is the absolute offset
     * of this chunk's first item in the full batch and is used by {@link PartialFailureMerger}
     * when building per-item failure placeholders / failed_indices metadata.
     */
    public record SubBatchResult(boolean success,
                                 int chunkSize,
                                 int startIndex,
                                 JsonNode body,
                                 String reason) {

        public static SubBatchResult ok(JsonNode body, int chunkSize, int startIndex) {
            return new SubBatchResult(true, chunkSize, startIndex, body, null);
        }

        public static SubBatchResult failed(int chunkSize, int startIndex, String reason) {
            return new SubBatchResult(false, chunkSize, startIndex, null, reason);
        }

        public boolean isSuccess() {
            return success;
        }
    }

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
                                Logger.warn("FE chunk failed: url={}, path={}, size={}, err={}",
                                        feUrl, fePath, chunkSize, briefReason(e));
                                return Mono.just(SubBatchResult.failed(chunkSize, startIndex, briefReason(e)));
                            }))
                    .onErrorResume(e -> {
                        Logger.warn("FE pick failed for chunk size={}, err={}",
                                chunkSize, briefReason(e));
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
