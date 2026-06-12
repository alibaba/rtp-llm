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
     * Caps how many sub-calls are in flight (and how many serialized payloads / parsed
     * responses are resident) at once, so a pathologically large fanout cannot balloon
     * heap. The common {@code count:5} split never reaches it; it only bites huge batches.
     */
    private static final int FANOUT_MAX_CONCURRENCY = 64;
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
        JSONWriter.Feature[] features = spec.isFanoutWriteNulls() ? WRITE_NULLS : NO_FEATURES;
        String arrayField = spec.getRequestArrayField();
        List<ChunkPlan> plans = new ArrayList<>(chunkBodies.size());
        int start = 0;
        for (JSONObject body : chunkBodies) {
            int chunkSize = body.getJSONArray(arrayField).size();
            plans.add(new ChunkPlan(body, start, chunkSize));
            start += chunkSize;
        }
        return Flux.fromIterable(plans)
                .flatMapSequential(plan -> dispatchOne(fePath, plan, features), FANOUT_MAX_CONCURRENCY)
                .collectList()
                .publishOn(Schedulers.parallel());
    }

    /**
     * Serializes the chunk lazily at subscription (so payloads are not all held at once) and
     * picks the FE in declaration order, keeping round-robin assignment deterministic. The FE
     * response is parsed on a {@link Schedulers#parallel()} worker rather than the Netty event
     * loop, so a large embedding response cannot stall the I/O thread serving other connections.
     */
    private Mono<SubBatchResult> dispatchOne(String fePath, ChunkPlan plan, JSONWriter.Feature[] features) {
        return Mono.fromCallable(() -> new Pick(fePool.next(), JSON.toJSONBytes(plan.body(), features)))
                .flatMap(pick -> feClient.postBytes(pick.feUrl(), fePath, pick.payload())
                        .publishOn(Schedulers.parallel())
                        .map(bytes -> SubBatchResult.ok(
                                JSON.parseObject(bytes), plan.chunkSize(), plan.startIndex()))
                        .onErrorResume(e -> {
                            String reason = DispatcherResponses.briefReason(e);
                            warnRateLimited("FE chunk failed: url={}, path={}, size={}, err={}, suppressed={}",
                                    pick.feUrl(), fePath, plan.chunkSize(), reason);
                            return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(), reason));
                        }))
                .onErrorResume(e -> {
                    String reason = DispatcherResponses.briefReason(e);
                    warnRateLimited("FE pick failed for chunk size={}, err={}, suppressed={}",
                            plan.chunkSize(), reason);
                    return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(), reason));
                });
    }

    /** A chunk's request body plus its absolute offset and item count in the batch. */
    private record ChunkPlan(JSONObject body, int startIndex, int chunkSize) {
    }

    /** A picked FE URL with the chunk already serialized for it. */
    private record Pick(String feUrl, byte[] payload) {
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
