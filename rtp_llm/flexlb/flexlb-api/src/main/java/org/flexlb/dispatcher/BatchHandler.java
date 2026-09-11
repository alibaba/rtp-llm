package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Dispatcher batch handler. Reads each batch request body as raw bytes, parses with
 * fastjson2, splits the request array per {@link SubBatchSpec}, builds per-chunk bodies,
 * stamps any pre-assigned BE targets, fans out via {@link FanoutService}, and merges with
 * {@link ResponseMerger}.
 *
 * <p>Status mapping: 400 on a non-JSON-object body, passthrough disposition for registered
 * paths whose body is not a splittable batch, 200 on full or partial success, and on total
 * failure the chunks' shared FE 4xx when they agree on one — 500 otherwise.
 *
 * <p>Single-element batches still fan out as one chunk so partial-failure semantics stay
 * uniform; router-level rejection of non-batch traffic happens upstream.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class BatchHandler {

    private static final int MAX_LOG_SCALAR_CHARS = 256;

    private final FanoutService fanoutService;
    private final SubBatchSpec subBatch;
    private final String splitPolicy;
    private final BatchScheduleClient batchScheduleClient;
    private final PassthroughClient passthroughClient;
    private final DispatcherMetricsReporter metricsReporter;
    private final boolean preAssignBe;
    private final FlexlbConfig loadBalanceConfig;
    private final FeAllocationMode feAllocationMode;
    private final int maxChunkCount;
    private final long maxAggregateRequestBytes;
    private final Scheduler cpuScheduler;

    @Autowired
    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleClient batchScheduleClient,
                        PassthroughClient passthroughClient,
                        DispatcherMetricsReporter metricsReporter,
                        ConfigService configService,
                        @Qualifier("dispatcherCpuScheduler") Scheduler cpuScheduler) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                configService.loadBalanceConfig(),
                cfg.getMaxAggregateRequestBytes(), cpuScheduler);
    }

    private BatchHandler(FanoutService fanoutService,
                         DispatchConfig cfg,
                         BatchScheduleClient batchScheduleClient,
                         PassthroughClient passthroughClient,
                         DispatcherMetricsReporter metricsReporter,
                         FlexlbConfig loadBalanceConfig,
                         long maxAggregateRequestBytes,
                         Scheduler cpuScheduler) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                loadBalanceConfig.getBatchScheduleMaxCount(), maxAggregateRequestBytes,
                loadBalanceConfig, cpuScheduler);
    }

    /** Package-private convenience for focused tests; mirrors the production default. */
    BatchHandler(FanoutService fanoutService,
                 DispatchConfig cfg,
                 BatchScheduleClient batchScheduleClient,
                 PassthroughClient passthroughClient,
                 DispatcherMetricsReporter metricsReporter) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                1000, cfg.getMaxAggregateRequestBytes(), null, Schedulers.immediate());
    }

    BatchHandler(FanoutService fanoutService,
                 DispatchConfig cfg,
                 BatchScheduleClient batchScheduleClient,
                 PassthroughClient passthroughClient,
                 DispatcherMetricsReporter metricsReporter,
                 int maxChunkCount) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                maxChunkCount, cfg.getMaxAggregateRequestBytes(), null, Schedulers.immediate());
    }

    BatchHandler(FanoutService fanoutService,
                 DispatchConfig cfg,
                 BatchScheduleClient batchScheduleClient,
                 PassthroughClient passthroughClient,
                 DispatcherMetricsReporter metricsReporter,
                 int maxChunkCount,
                 long maxAggregateRequestBytes) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                maxChunkCount, maxAggregateRequestBytes, null, Schedulers.immediate());
    }

    BatchHandler(FanoutService fanoutService,
                 DispatchConfig cfg,
                 BatchScheduleClient batchScheduleClient,
                 PassthroughClient passthroughClient,
                 DispatcherMetricsReporter metricsReporter,
                 int maxChunkCount,
                 long maxAggregateRequestBytes,
                 FlexlbConfig loadBalanceConfig) {
        this(fanoutService, cfg, batchScheduleClient, passthroughClient, metricsReporter,
                maxChunkCount, maxAggregateRequestBytes, loadBalanceConfig,
                Schedulers.immediate());
    }

    BatchHandler(FanoutService fanoutService,
                 DispatchConfig cfg,
                 BatchScheduleClient batchScheduleClient,
                 PassthroughClient passthroughClient,
                 DispatcherMetricsReporter metricsReporter,
                 int maxChunkCount,
                 long maxAggregateRequestBytes,
                 FlexlbConfig loadBalanceConfig,
                 Scheduler cpuScheduler) {
        this.fanoutService = fanoutService;
        this.subBatch = cfg.getSubBatchSpec();
        this.splitPolicy = subBatch.mode().name().toLowerCase() + ":" + subBatch.value();
        this.batchScheduleClient = batchScheduleClient;
        this.passthroughClient = passthroughClient;
        this.metricsReporter = metricsReporter;
        this.preAssignBe = cfg.isPreAssignBe();
        this.loadBalanceConfig = loadBalanceConfig;
        this.feAllocationMode = cfg.getFeAllocation() == null
                ? FeAllocationMode.MASTER
                : FeAllocationMode.parse(cfg.getFeAllocation());
        if (maxChunkCount < 1) {
            throw new IllegalArgumentException("maxChunkCount must be >= 1, got " + maxChunkCount);
        }
        if (maxAggregateRequestBytes <= 0) {
            throw new IllegalArgumentException("maxAggregateRequestBytes must be > 0, got "
                    + maxAggregateRequestBytes);
        }
        this.maxChunkCount = maxChunkCount;
        this.maxAggregateRequestBytes = maxAggregateRequestBytes;
        this.cpuScheduler = cpuScheduler;
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        AtomicBoolean delegatedToPassthrough = new AtomicBoolean(false);
        AtomicBoolean pvFinalized = new AtomicBoolean(false);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0])
                .flatMap(bytes -> Mono.fromCallable(
                                () -> handleBody(request, spec, bytes, pv, delegatedToPassthrough))
                        .subscribeOn(cpuScheduler)
                        .flatMap(response -> response))
                .onErrorResume(e -> {
            String errMsg = DispatcherResponses.briefReason(e);
            Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
            pv.setError(errMsg);
            if (e instanceof DataBufferLimitException) {
                // Body over spring.codec.max-in-memory-size is a deterministic client error;
                // a 500 would invite pointless retries and pollute the server error rate.
                return DispatcherResponses.error(413, "request_body_too_large",
                        "batch body exceeds the server limit; see MAX_IN_MEMORY_SIZE");
            }
            if (e instanceof AggregateResponseTooLargeException) {
                return DispatcherResponses.error(413, "batch_response_too_large",
                        "aggregate sub-batch response exceeds the dispatcher limit");
            }
            if (e instanceof AggregateRequestTooLargeException) {
                return DispatcherResponses.error(413, "batch_request_too_large",
                        "aggregate sub-batch request exceeds the dispatcher limit");
            }
            // Stable, non-revealing text: the exception message can carry the FE address or
            // upstream response detail, which must not cross the client boundary. The full
            // reason is in the WARN above and in pv.log.
            return DispatcherResponses.error(500, "dispatch_failed", "batch dispatch failed");
        }).doOnNext(resp -> {
            pv.setHttpStatus(resp.rawStatusCode());
            if (!delegatedToPassthrough.get() && pvFinalized.compareAndSet(false, true)) {
                // Emit before handing the response downstream. A doFinally-only record can race
                // the caller observing Mono.block()/onNext after CPU work was offloaded.
                finalizePvRecord(pv, SignalType.ON_COMPLETE);
            }
        })
          .doFinally(signal -> {
              // Cancellation before any response has no onNext; this is also a defensive fallback
              // for an unexpected terminal error while preserving exactly-once accounting.
              if (!delegatedToPassthrough.get()
                      && pvFinalized.compareAndSet(false, true)) {
                  finalizePvRecord(pv, signal);
              }
          });
    }

    private Mono<ServerResponse> handleBody(ServerRequest request, BatchEndpointSpec spec,
                                            byte[] bytes, DispatchPvLogData pv,
                                            AtomicBoolean delegatedToPassthrough) {
            JSONObject body = BatchBodyParser.parseObject(bytes);
            if (body == null) {
                return badRequest("expected a JSON object body");
            }
            populateRequestLogFields(pv, body);
            // Enforce the reserved routing field at the registered HTTP boundary, before the
            // split-vs-passthrough disposition. A companion-field request is forwarded whole,
            // but must not use that path to make an FE dial a caller-selected backend.
            String generateConfigError = BatchChunkAssembler.validateGenerateConfig(body);
            if (generateConfigError != null) {
                return badRequest(generateConfigError);
            }
            JSONArray arr = BatchBodyParser.findArrayField(body, spec.getRequestArrayField());
            if (!spec.isSplittableBatch(body, arr)) {
                // Registered path, but this body is not a splittable batch (absent array field,
                // non-batch-shaped array, or a whole-body companion field — see
                // BatchEndpointSpec#isSplittableBatch). Forward verbatim to one FE per the
                // registry contract. PassthroughClient emits its own pv record.
                delegatedToPassthrough.set(true);
                return passthroughClient.forward(request, bytes);
            }
            String validationError = spec.validateForFanout(body);
            if (validationError != null) {
                return badRequest(validationError);
            }
            if (arr.isEmpty()) {
                JSONObject emptyEnvelope = new JSONObject();
                emptyEnvelope.put(spec.getResponseArrayField(), new JSONArray());
                if (spec.getPostMerger() != null) {
                    spec.getPostMerger().apply(emptyEnvelope, List.of(), List.of(), spec, body);
                }
                return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(emptyEnvelope));
            }
            pv.setTotalItems(arr.size());
            int chunkCount = BatchChunkAssembler.chunkCount(arr.size(), subBatch);
            pv.setChunkCount(chunkCount);
            if (chunkCount > maxChunkCount) {
                return DispatcherResponses.error(413, "too_many_sub_batches",
                        "batch produces " + chunkCount + " sub-batches; maximum is "
                                + maxChunkCount + " (BATCH_SCHEDULE_MAX_COUNT)");
            }
            boolean trafficPolicyActive = hasActiveTrafficPolicy();
            boolean atomicBatchAllowed = !trafficPolicyActive;
            PreparedBatch prepared = prepareBatch(
                    body, arr, chunkCount, spec, pv, atomicBatchAllowed);
            return Mono.defer(() -> {
                        boolean assignBe = shouldPreAssignBe(spec, trafficPolicyActive);
                        boolean assignFe = feAllocationMode == FeAllocationMode.MASTER;
                        return resolveTargets(prepared.chunkBodies().size(), assignBe, assignFe)
                                .publishOn(cpuScheduler)
                                .flatMap(targets -> {
                                    if (assignBe) {
                                        BatchChunkAssembler.stampPreAssignedBe(
                                                prepared.chunkBodies(), targets);
                                    }
                                    List<String> preAssignedFeUrls = assignFe
                                            ? preAssignedFeUrls(targets) : List.of();
                                    long fanoutStart = System.currentTimeMillis();
                                    return fanoutService.dispatchChunks(
                                                    spec.getPath(), prepared.chunkBodies(),
                                                    preAssignedFeUrls, spec,
                                                    request.headers().asHttpHeaders(),
                                                    request.uri().getRawQuery())
                                            .doOnNext(subs -> metricsReporter.reportFanoutRt(
                                                    System.currentTimeMillis() - fanoutStart,
                                                    feAllocationMode.configValue()))
                                            .publishOn(cpuScheduler)
                                            .map(subs -> ResponseMerger.merge(subs, spec, body))
                                            .flatMap(merged -> {
                                                pv.setFailedChunks(
                                                        merged.failedReasons().size());
                                                if (merged.allFailed()
                                                        || (spec.isFailOnPartialFailure()
                                                        && merged.hasFailures())) {
                                                    return errorResponse(merged);
                                                }
                                                return DispatcherResponses.jsonBytes(
                                                        200, BatchBodyParser.serialize(
                                                                merged.body()));
                                            });
                                });
                    });
    }

    private void finalizePvRecord(DispatchPvLogData pv, SignalType signal) {
        int status = pv.getHttpStatus();
        String error = pv.getError();
        if (signal == SignalType.CANCEL && status == 0) {
            status = 499;
            error = error != null ? error : "client cancelled";
        }
        pv.finish(status, error);
        pv.emit();
        metricsReporter.reportRequest("batch", pv.getPath(), status, pv.getCostMs());
        if (pv.getChunkCount() > 0) {
            metricsReporter.reportBatchShape(pv.getPath(), pv.getTotalItems(), pv.getChunkCount());
        }
    }

    private PreparedBatch prepareBatch(JSONObject body, JSONArray arr, int chunkCount,
                                       BatchEndpointSpec spec, DispatchPvLogData pv,
                                       boolean atomicBatchAllowed) {
        long projectedBytes = BatchChunkAssembler.projectedOutboundBytes(
                body, arr, chunkCount, spec, atomicBatchAllowed);
        if (projectedBytes > maxAggregateRequestBytes) {
            throw new AggregateRequestTooLargeException(maxAggregateRequestBytes);
        }
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, subBatch);
        recordChunkShape(pv, chunks);
        List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                body, chunks, spec.getRequestArrayField(), atomicBatchAllowed);
        spec.prepareChunkBodies(body, chunkBodies);
        return new PreparedBatch(chunkBodies);
    }

    private boolean shouldPreAssignBe(BatchEndpointSpec spec, boolean trafficPolicyActive) {
        if (!preAssignBe || !spec.isPreAssignable()) {
            return false;
        }
        return !trafficPolicyActive;
    }

    private boolean hasActiveTrafficPolicy() {
        TrafficPolicyConfig policy = loadBalanceConfig == null
                ? null : loadBalanceConfig.getTrafficPolicy();
        return policy != null && policy.hasActiveRoutingRules();
    }

    private record PreparedBatch(List<JSONObject> chunkBodies) {}

    private void populateRequestLogFields(DispatchPvLogData pv, JSONObject body) {
        pv.setSplitPolicy(splitPolicy);
        pv.setModel(scalarForLog(body.get("model")));
        Object requestId = body.get("__request_id__");
        if (requestId == null) {
            requestId = body.get("request_id");
        }
        pv.setCallerRequestId(scalarForLog(requestId));
    }

    private static void recordChunkShape(DispatchPvLogData pv, List<JSONArray> chunks) {
        if (chunks.isEmpty()) {
            return;
        }
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (JSONArray chunk : chunks) {
            min = Math.min(min, chunk.size());
            max = Math.max(max, chunk.size());
        }
        pv.setMinChunkItems(min);
        pv.setMaxChunkItems(max);
    }

    /** Keeps request-controlled observability fields scalar and bounded. */
    private static String scalarForLog(Object value) {
        if (!(value instanceof String || value instanceof Number || value instanceof Boolean)) {
            return null;
        }
        String text = String.valueOf(value);
        if (text.length() <= MAX_LOG_SCALAR_CHARS) {
            return text;
        }
        return text.substring(0, MAX_LOG_SCALAR_CHARS);
    }

    /**
     * Master mode's per-chunk FE assignment, index-aligned to {@code targets} (and thus to chunks).
     * A null entry — or an index past a short target list — means "no master FE for this chunk";
     * {@link FanoutService} fails such a chunk visibly rather than changing allocation source.
     */
    private static List<String> preAssignedFeUrls(List<BatchScheduleTarget> targets) {
        List<String> feUrls = new ArrayList<>(targets.size());
        for (BatchScheduleTarget target : targets) {
            feUrls.add(target.getFeUrl());
        }
        return feUrls;
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_batch_request", message);
    }

    private Mono<ServerResponse> errorResponse(ResponseMerger.MergedResponse merged) {
        JSONObject body = new JSONObject();
        body.put("error", merged.allFailed()
                ? "all_sub_batches_failed" : "sub_batch_failed");
        // Item units, matching the success-path _partial_failure block. For the ordinary
        // all-failed path failed_count == total_count; fail-closed endpoints can have fewer
        // failed items than total items. total_chunks is sub-batch units.
        int failedItems = merged.failedIndices().size();
        body.put("failed_count", failedItems);
        body.put("total_count", merged.totalItems());
        body.put("total_chunks", merged.totalChunks());
        JSONArray reasons = new JSONArray();
        merged.failedReasons().stream().distinct().forEach(reasons::add);
        body.put("failed_reasons", reasons);
        return DispatcherResponses.jsonBytes(merged.errorStatus(), BatchBodyParser.serialize(body));
    }

    /**
     * Resolves only the allocation dimensions the request will consume. Master FE mode always asks
     * for {@code assign_fe}; BE selection is requested only when this endpoint can consume the
     * stamped role address. Local FE mode with BE pre-assignment disabled needs no master call at
     * all. This keeps both global cursors free of invisible, discarded advances.
     */
    private Mono<List<BatchScheduleTarget>> resolveTargets(
            int chunkCount, boolean assignBe, boolean assignFe) {
        if (!assignBe && !assignFe) {
            return Mono.just(List.of());
        }
        long start = System.currentTimeMillis();
        return batchScheduleClient.requestTargets(chunkCount, assignBe, assignFe)
                .doOnNext(targets -> metricsReporter.reportPreassignRt(
                        System.currentTimeMillis() - start, !targets.isEmpty(), assignBe, assignFe));
    }
}
