package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import lombok.RequiredArgsConstructor;
import org.flexlb.dispatcher.FeClient;
import org.flexlb.dispatcher.FePool;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

/**
 * Per-chunk fanout on the dispatcher batch path. Serializes each chunk via fastjson2's
 * {@link JSON#toJSONBytes(Object, JSONWriter.Feature...)} and parses the FE response bytes back
 * into a {@link JSONObject}. Whether the serialize includes {@link JSONWriter.Feature#WriteNulls}
 * is driven by {@link BatchEndpointSpec#isFanoutWriteNulls()} — see the field's Javadoc for
 * when null preservation matters. Failed chunks become {@link SubBatchResult#failed} and never
 * abort their siblings.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
@RequiredArgsConstructor
public class FanoutService {

    private final FeClient feClient;
    private final FePool fePool;

    public Mono<List<SubBatchResult>> dispatchChunks(String fePath,
                                                              List<JSONObject> chunkBodies,
                                                              BatchEndpointSpec spec) {
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunkBodies.size());
        int start = 0;
        JSONWriter.Feature[] features = spec.isFanoutWriteNulls()
                ? new JSONWriter.Feature[] { JSONWriter.Feature.WriteNulls }
                : new JSONWriter.Feature[0];
        for (JSONObject body : chunkBodies) {
            int chunkSize = body.getJSONArray(spec.getRequestArrayField()).size();
            int startIndex = start;
            byte[] payload = JSON.toJSONBytes(body, features);
            calls.add(Mono.fromCallable(fePool::next)
                    .flatMap(feUrl -> feClient.postBytes(feUrl, fePath, payload)
                            .map(bytes -> SubBatchResult.ok(
                                    JSON.parseObject(bytes), chunkSize, startIndex))
                            .onErrorResume(e -> {
                                Logger.warn("FE chunk failed: url={}, path={}, size={}, err={}",
                                        feUrl, fePath, chunkSize, briefReason(e));
                                return Mono.just(SubBatchResult.failed(
                                        chunkSize, startIndex, briefReason(e)));
                            }))
                    .onErrorResume(e -> {
                        Logger.warn("FE pick failed for chunk size={}, err={}",
                                chunkSize, briefReason(e));
                        return Mono.just(SubBatchResult.failed(
                                chunkSize, startIndex, briefReason(e)));
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
