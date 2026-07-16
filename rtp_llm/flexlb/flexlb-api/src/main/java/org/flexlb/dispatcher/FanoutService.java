package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import lombok.RequiredArgsConstructor;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

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
     * Caps how many sub-calls are in flight — and thus how many serialized request payloads exist
     * at once — bounding concurrent I/O and outbound buffer pressure. The merge still collects all
     * parsed responses before assembling, so peak heap scales with total batch size, not with this
     * cap. The common {@code count:5} split never reaches it; it only bites huge batches.
     */
    private static final int FANOUT_MAX_CONCURRENCY = 64;

    private final FeClient feClient;
    private final FePool fePool;
    private final DispatcherMetricsReporter metricsReporter;
    /** During an FE outage the fanout path fails per chunk; cap the WARN stream at 1/s. */
    private final RateLimitedWarn failureWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

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
                .flatMapSequential(plan -> dispatchOne(fePath, plan, features, spec), FANOUT_MAX_CONCURRENCY)
                .collectList()
                .publishOn(Schedulers.parallel());
    }

    /**
     * Serializes the chunk at subscription and picks the FE in declaration order, keeping
     * round-robin assignment deterministic. Note {@code flatMapSequential} subscribes eagerly up
     * to {@link #FANOUT_MAX_CONCURRENCY}, so within that cap all serialized payloads coexist;
     * only beyond it does serialization stagger. The FE response is parsed on a
     * {@link Schedulers#parallel()} worker rather than the Netty event loop, so a large
     * embedding response cannot stall the I/O thread serving other connections.
     *
     * <p>Threading note: the per-chunk {@code JSON.toJSONBytes} serialize (and the inbound parse +
     * chunk-body build in {@link BatchHandler}) deliberately stay on the event loop — byte-array
     * work over a shallow-copied envelope, cheaper than a scheduler hand-off. Only the FE-response
     * parse and downstream merge are offloaded.
     */
    private Mono<SubBatchResult> dispatchOne(String fePath, ChunkPlan plan, JSONWriter.Feature[] features,
                                             BatchEndpointSpec spec) {
        return Mono.fromCallable(() -> new Pick(fePool.next(), JSON.toJSONBytes(plan.body(), features)))
                .flatMap(pick -> {
                    long start = System.currentTimeMillis();
                    return feClient.postBytes(pick.feUrl(), fePath, pick.payload())
                            .publishOn(Schedulers.parallel())
                            .map(bytes -> {
                                // Parse before reporting: a 200 with a non-JSON body must count
                                // once as failed, not once as ok and again as failed.
                                JSONObject parsed = JSON.parseObject(bytes);
                                SubBatchResult result = SubBatchResult.ok(parsed, plan.chunkSize(), plan.startIndex());
                                // A 200 whose response array is absent or the wrong length is merged
                                // as a failure, so meter it as one too — using the merge's own
                                // authority so the metric can't drift from the merge outcome.
                                String reason = ResponseMerger.wellFormed(result, spec)
                                        ? DispatcherMetricsReporter.CHUNK_OK
                                        : DispatcherMetricsReporter.CHUNK_MALFORMED;
                                metricsReporter.reportChunk(reason, System.currentTimeMillis() - start);
                                return result;
                            })
                            // A 200 with an empty body completes the Mono empty; without a
                            // placeholder the chunk would silently vanish from collectList and
                            // the merged response array would shift indices.
                            .switchIfEmpty(Mono.fromSupplier(() -> {
                                metricsReporter.reportChunk(DispatcherMetricsReporter.CHUNK_TRANSPORT,
                                        System.currentTimeMillis() - start);
                                failureWarn.warn("FE chunk returned empty body: url={}, path={}, size={}",
                                        pick.feUrl(), fePath, plan.chunkSize());
                                return SubBatchResult.failed(plan.chunkSize(), plan.startIndex(),
                                        "empty FE response body");
                            }))
                            .onErrorResume(e -> {
                                String reason = DispatcherResponses.briefReason(e);
                                int feStatus = DispatcherResponses.httpStatusOf(e);
                                metricsReporter.reportChunk(reasonCategory(feStatus),
                                        System.currentTimeMillis() - start);
                                failureWarn.warn("FE chunk failed: url={}, path={}, size={}, err={}",
                                        pick.feUrl(), fePath, plan.chunkSize(), reason);
                                return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(),
                                        reason, feStatus));
                            });
                })
                .onErrorResume(e -> {
                    String reason = DispatcherResponses.briefReason(e);
                    metricsReporter.reportChunk(DispatcherMetricsReporter.CHUNK_PICK_FAILED, 0);
                    failureWarn.warn("FE pick failed for chunk size={}, err={}",
                            plan.chunkSize(), reason);
                    return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(), reason));
                });
    }

    /** Bounded failure-reason category for the {@code reason} metric tag (keeps cardinality low). */
    private static String reasonCategory(int feStatus) {
        if (feStatus >= 400 && feStatus < 500) {
            return DispatcherMetricsReporter.CHUNK_HTTP_4XX;
        }
        if (feStatus >= 500) {
            return DispatcherMetricsReporter.CHUNK_HTTP_5XX;
        }
        return DispatcherMetricsReporter.CHUNK_TRANSPORT;
    }

    /** A chunk's request body plus its absolute offset and item count in the batch. */
    private record ChunkPlan(JSONObject body, int startIndex, int chunkSize) {
    }

    /** A picked FE URL with the chunk already serialized for it. */
    private record Pick(String feUrl, byte[] payload) {
    }
}
