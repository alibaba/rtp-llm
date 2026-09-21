package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import com.google.common.util.concurrent.RateLimiter;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FanoutService {

    /** Independent per-batch ceilings for outbound requests and retained FE responses. */
    static final long MAX_AGGREGATE_BYTES = 128L * 1024 * 1024;

    private static final int FANOUT_CONCURRENCY = (int) (MAX_AGGREGATE_BYTES / FeClient.MAX_RESPONSE_BYTES);

    private final FeClient feClient;
    private final DispatcherMetricsReporter metricsReporter;
    /** During an FE outage the fanout path fails per chunk; cap the WARN stream at 1/s. */
    private final RateLimiter failureWarn = RateLimiter.create(1);

    public FanoutService(FeClient feClient, DispatcherMetricsReporter metricsReporter) {
        this.feClient = feClient;
        this.metricsReporter = metricsReporter;
    }

    public Mono<List<SubBatchResult>> dispatchChunks(String fePath,
                                                     List<JSONObject> chunkBodies,
                                                     List<String> preAssignedFeUrls,
                                                     BatchEndpointSpec spec,
                                                     HttpHeaders inboundHeaders,
                                                     String rawQuery) {
        String arrayField = spec.getRequestArrayField();
        List<ChunkPlan> plans = new ArrayList<>(chunkBodies.size());
        int start = 0;
        for (int i = 0; i < chunkBodies.size(); i++) {
            JSONObject body = chunkBodies.get(i);
            int chunkSize = body.getJSONArray(arrayField).size();
            String preAssignedFe = i < preAssignedFeUrls.size() ? preAssignedFeUrls.get(i) : null;
            plans.add(new ChunkPlan(body, start, chunkSize, preAssignedFe));
            start += chunkSize;
        }
        return Mono.defer(() -> {
            AtomicByteBudget responseBudget = new AtomicByteBudget(MAX_AGGREGATE_BYTES);
            AtomicByteBudget requestBudget = new AtomicByteBudget(MAX_AGGREGATE_BYTES);
            return Flux.fromIterable(plans)
                    .flatMapSequential(plan -> dispatchOne(fePath, plan, spec, inboundHeaders,
                                    rawQuery, responseBudget, requestBudget),
                            FANOUT_CONCURRENCY)
                    .collectList();
        });
    }

    /** Serializes lazily so at most the concurrency limit of outbound bodies exists at once. */
    private Mono<SubBatchResult> dispatchOne(String fePath, ChunkPlan plan,
                                             BatchEndpointSpec spec, HttpHeaders inboundHeaders, String rawQuery,
                                             AtomicByteBudget responseBudget,
                                             AtomicByteBudget requestBudget) {
        if (plan.feUrl() == null || plan.feUrl().isBlank()) {
            metricsReporter.reportChunk(DispatcherMetricsReporter.CHUNK_NO_FE, 0);
            if (failureWarn.tryAcquire()) {
                Logger.warn("chunk has no master FE assignment: size={}", plan.chunkSize());
            }
            return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(), 0));
        }
        AtomicByteBudget.Reservation responseReservation = responseBudget.newReservation();
        return Mono.fromCallable(() -> {
                    byte[] payload = BatchBodyParser.serialize(plan.body());
                    if (!requestBudget.tryReserve(payload.length)) {
                        throw new AggregateRequestTooLargeException(requestBudget.limit());
                    }
                    return payload;
                })
                .subscribeOn(Schedulers.parallel())
                .flatMap(payload -> {
                    long start = System.currentTimeMillis();
                    return feClient.postBytes(plan.feUrl(), fePath, payload, inboundHeaders,
                                    rawQuery, responseReservation)
                            .publishOn(Schedulers.parallel())
                            .map(bytes -> {
                                JSONObject parsed = BatchBodyParser.parseObject(bytes);
                                SubBatchResult result = SubBatchResult.ok(parsed, plan.chunkSize(), plan.startIndex());
                                // A malformed 2xx counts as a failure in both metrics and merging.
                                String reason = ResponseMerger.wellFormed(result, spec)
                                        ? DispatcherMetricsReporter.CHUNK_OK
                                        : DispatcherMetricsReporter.CHUNK_MALFORMED;
                                metricsReporter.reportChunk(reason, System.currentTimeMillis() - start);
                                return result;
                            })
                            .onErrorResume(e -> failedChunk(plan, fePath, start, e));
                });
    }

    private Mono<SubBatchResult> failedChunk(ChunkPlan plan, String path, long start, Throwable error) {
        if (error instanceof AggregateResponseTooLargeException) {
            return Mono.error(error);
        }
        String reason = error.toString();
        int status = error instanceof WebClientResponseException http ? http.getRawStatusCode() : 0;
        metricsReporter.reportChunk(reasonCategory(status), System.currentTimeMillis() - start);
        if (failureWarn.tryAcquire()) {
            Logger.warn("FE chunk failed: url={}, path={}, size={}, err={}",
                    plan.feUrl(), path, plan.chunkSize(), reason);
        }
        return Mono.just(SubBatchResult.failed(plan.chunkSize(), plan.startIndex(), status));
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

    private record ChunkPlan(JSONObject body, int startIndex, int chunkSize, String feUrl) {
    }
}
