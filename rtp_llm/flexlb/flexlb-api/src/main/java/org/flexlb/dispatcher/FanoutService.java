package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import lombok.RequiredArgsConstructor;
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

    private static final JSONWriter.Feature[] WRITE_NULLS = { JSONWriter.Feature.WriteNulls };
    private static final JSONWriter.Feature[] NO_FEATURES = {};
    /**
     * Failure WARNs are capped at one per interval with a suppressed-count rider: during an
     * FE outage the fanout path fails per chunk, and at production QPS an uncapped WARN
     * stream is tens of thousands of log lines per second — enough to cost real throughput.
     */
    private static final long FAILURE_WARN_INTERVAL_NANOS = java.util.concurrent.TimeUnit.SECONDS.toNanos(1);

    private final FeClient feClient;
    private final FePool fePool;
    private final java.util.concurrent.atomic.AtomicLong lastFailureWarnNanos =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong suppressedFailureWarns =
            new java.util.concurrent.atomic.AtomicLong();

    public Mono<List<SubBatchResult>> dispatchChunks(String fePath,
                                                     List<JSONObject> chunkBodies,
                                                     BatchEndpointSpec spec) {
        List<Mono<SubBatchResult>> calls = new ArrayList<>(chunkBodies.size());
        int start = 0;
        JSONWriter.Feature[] features = spec.isFanoutWriteNulls() ? WRITE_NULLS : NO_FEATURES;
        for (JSONObject body : chunkBodies) {
            int chunkSize = body.getJSONArray(spec.getRequestArrayField()).size();
            int startIndex = start;
            byte[] payload = JSON.toJSONBytes(body, features);
            calls.add(Mono.fromCallable(fePool::next)
                    .flatMap(feUrl -> feClient.postBytes(feUrl, fePath, payload)
                            .map(bytes -> SubBatchResult.ok(
                                    JSON.parseObject(bytes), chunkSize, startIndex))
                            .onErrorResume(e -> {
                                String reason = DispatcherResponses.briefReason(e);
                                warnRateLimited("FE chunk failed: url={}, path={}, size={}, err={}, suppressed={}",
                                        feUrl, fePath, chunkSize, reason);
                                return Mono.just(SubBatchResult.failed(chunkSize, startIndex, reason));
                            }))
                    .onErrorResume(e -> {
                        String reason = DispatcherResponses.briefReason(e);
                        warnRateLimited("FE pick failed for chunk size={}, err={}, suppressed={}",
                                chunkSize, reason);
                        return Mono.just(SubBatchResult.failed(chunkSize, startIndex, reason));
                    }));
            start += chunkSize;
        }
        return Flux.mergeSequential(Flux.fromIterable(calls))
                .collectList()
                .publishOn(Schedulers.parallel());
    }

    /**
     * Emits at most one failure WARN per {@link #FAILURE_WARN_INTERVAL_NANOS}; calls landing
     * inside the window are counted and reported via the emitting call's trailing
     * {@code suppressed=} placeholder, so outage magnitude stays visible without the volume.
     */
    private void warnRateLimited(String format, Object... argsWithoutSuppressed) {
        long now = System.nanoTime();
        long last = lastFailureWarnNanos.get();
        if (now - last >= FAILURE_WARN_INTERVAL_NANOS && lastFailureWarnNanos.compareAndSet(last, now)) {
            Object[] args = new Object[argsWithoutSuppressed.length + 1];
            System.arraycopy(argsWithoutSuppressed, 0, args, 0, argsWithoutSuppressed.length);
            args[argsWithoutSuppressed.length] = suppressedFailureWarns.getAndSet(0);
            Logger.warn(format, args);
        } else {
            suppressedFailureWarns.incrementAndGet();
        }
    }
}
