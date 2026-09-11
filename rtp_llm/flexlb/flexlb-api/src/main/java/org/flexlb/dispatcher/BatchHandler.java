package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.exception.BatchScheduleTransportException;
import org.flexlb.service.BatchScheduleCoordinator;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Scheduler;

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
    private final BatchScheduleCoordinator batchScheduleCoordinator;
    private final PassthroughClient passthroughClient;
    private final DispatcherMetricsReporter metricsReporter;
    private final boolean preAssignBe;
    private final FlexlbConfig loadBalanceConfig;
    private final FeAllocationMode feAllocationMode;
    private final int maxChunkCount;
    private final long maxAggregateRequestBytes;
    private final Scheduler cpuScheduler;

    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleCoordinator batchScheduleCoordinator,
                        PassthroughClient passthroughClient,
                        DispatcherMetricsReporter metricsReporter,
                        ConfigService configService,
                        @Qualifier("dispatcherCpuScheduler") Scheduler cpuScheduler) {
        this.fanoutService = fanoutService;
        this.subBatch = cfg.getSubBatchSpec();
        this.splitPolicy = subBatch.mode().name().toLowerCase() + ":" + subBatch.value();
        this.batchScheduleCoordinator = batchScheduleCoordinator;
        this.passthroughClient = passthroughClient;
        this.metricsReporter = metricsReporter;
        this.preAssignBe = cfg.isPreAssignBe();
        this.loadBalanceConfig = configService.loadBalanceConfig();
        this.feAllocationMode = FeAllocationMode.parse(cfg.getFeAllocation());
        this.maxChunkCount = loadBalanceConfig.getRouter().getBatchScheduleMaxCount();
        this.maxAggregateRequestBytes = cfg.getMaxAggregateRequestBytes();
        this.cpuScheduler = cpuScheduler;
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        AtomicBoolean delegatedToPassthrough = new AtomicBoolean(false);
        AtomicBoolean pvFinalized = new AtomicBoolean(false);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0])
                .flatMap(bytes -> Mono.defer(
                                () -> handleBody(request, spec, bytes, pv, delegatedToPassthrough))
                        .subscribeOn(cpuScheduler))
                .onErrorResume(e -> {
                    String errMsg = DispatcherResponses.briefReason(e);
                    Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
                    pv.setError(errMsg);
                    if (e instanceof BatchScheduleTransportException) {
                        return DispatcherResponses.error(503, "batch_schedule_failed", "batch target allocation failed");
                    }
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
                })
                .doOnNext(resp -> {
                    pv.setHttpStatus(resp.rawStatusCode());
                    if (!delegatedToPassthrough.get() && pvFinalized.compareAndSet(false, true)) {
                        // Emit before handing the response downstream. A doFinally-only record can race
                        // the caller observing Mono.block()/onNext after CPU work was offloaded.
                        finalizePvRecord(pv, SignalType.ON_COMPLETE);
                    }
                })
                .doFinally(signal -> {
                    // Cancellation before a response has no onNext; emit exactly once.
                    if (!delegatedToPassthrough.get() && pvFinalized.compareAndSet(false, true)) {
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
            spec.finishMerge(emptyEnvelope, List.of(), List.of(), body);
            return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(emptyEnvelope));
        }
        pv.setTotalItems(arr.size());
        int chunkCount = BatchChunkAssembler.chunkCount(arr.size(), subBatch);
        pv.setChunkCount(chunkCount);
        if (chunkCount > maxChunkCount) {
            return DispatcherResponses.error(413, "too_many_sub_batches",
                    "batch produces " + chunkCount + " sub-batches; maximum is "
                            + maxChunkCount + " (router.batchScheduleMaxCount)");
        }
        boolean atomicBatchAllowed = !hasActiveTrafficPolicy();
        List<JSONObject> chunkBodies = prepareBatch(
                body, arr, chunkCount, spec, pv, atomicBatchAllowed);
        boolean assignBe = preAssignBe && spec.isPreAssignable() && atomicBatchAllowed;
        boolean assignFe = feAllocationMode == FeAllocationMode.MASTER;
        return resolveTargets(chunkBodies.size(), assignBe, assignFe)
                .publishOn(cpuScheduler)
                .flatMap(allocation -> {
                    if (!allocation.isSuccess()) {
                        int status = allocation.getCode() == StrategyErrorType.INVALID_REQUEST.getErrorCode()
                                ? 400 : 503;
                        return DispatcherResponses.error(status, "batch_schedule_failed",
                                allocation.getErrorMessage());
                    }
                    List<BatchScheduleTarget> targets = allocation.getServerStatus();
                    if (assignBe) {
                        BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
                    }
                    List<String> preAssignedFeUrls = assignFe
                            ? preAssignedFeUrls(targets) : List.of();
                    long fanoutStart = System.currentTimeMillis();
                    return fanoutService.dispatchChunks(
                                    spec.getPath(), chunkBodies,
                                    preAssignedFeUrls, spec,
                                    request.headers().asHttpHeaders(),
                                    request.uri().getRawQuery())
                            .doOnNext(subs -> metricsReporter.reportFanoutRt(
                                    System.currentTimeMillis() - fanoutStart,
                                    feAllocationMode.configValue()))
                            .publishOn(cpuScheduler)
                            .map(subs -> ResponseMerger.merge(subs, spec, body))
                            .flatMap(merged -> {
                                pv.setFailedChunks(merged.failedReasons().size());
                                if (merged.allFailed()
                                        || (spec.isFailOnPartialFailure()
                                        && merged.hasFailures())) {
                                    return errorResponse(merged);
                                }
                                return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(merged.body()));
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

    private List<JSONObject> prepareBatch(JSONObject body, JSONArray arr, int chunkCount,
                                       BatchEndpointSpec spec, DispatchPvLogData pv,
                                       boolean atomicBatchAllowed) {
        long projectedBytes = BatchChunkAssembler.projectedChunkBytes(
                body, arr, chunkCount, spec, atomicBatchAllowed, List.of())
                + 1024L * chunkCount; // Preserve the allowance for routing fields added after allocation.
        if (projectedBytes > maxAggregateRequestBytes) {
            throw new AggregateRequestTooLargeException(maxAggregateRequestBytes);
        }
        List<JSONArray> chunks = BatchChunkAssembler.split(arr, subBatch);
        recordChunkShape(pv, chunks);
        return BatchChunkAssembler.buildChunkBodies(body, chunks, spec, atomicBatchAllowed);
    }

    private boolean hasActiveTrafficPolicy() {
        TrafficPolicyConfig policy = loadBalanceConfig.getRouter().getGroupSelector();
        return policy != null && (!policy.getRules().isEmpty() || !policy.getDefaultTargets().isEmpty());
    }

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

    /** FE assignments retain the same index order as the allocated targets and chunks. */
    private static List<String> preAssignedFeUrls(List<BatchScheduleTarget> targets) {
        return targets.stream().map(BatchScheduleTarget::getFeUrl).toList();
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
    private Mono<BatchScheduleResponse> resolveTargets(
            int chunkCount, boolean assignBe, boolean assignFe) {
        if (!assignBe && !assignFe) {
            return Mono.just(BatchScheduleResponse.success(List.of()));
        }
        BatchScheduleRequest request = new BatchScheduleRequest();
        request.setBatchCount(chunkCount);
        request.setAssignBe(assignBe);
        request.setAssignFe(assignFe);
        long start = System.currentTimeMillis();
        return batchScheduleCoordinator.schedule(request)
                .doOnNext(response -> metricsReporter.reportPreassignRt(
                        System.currentTimeMillis() - start, response.isSuccess(), assignBe, assignFe))
                .doOnError(error -> metricsReporter.reportPreassignRt(
                        System.currentTimeMillis() - start, false, assignBe, assignFe));
    }
}
