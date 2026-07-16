package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;

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

    private final FanoutService fanoutService;
    private final SubBatchSpec subBatch;
    private final BatchScheduleClient batchScheduleClient;
    private final PassthroughClient passthroughClient;
    private final DispatcherMetricsReporter metricsReporter;
    private final boolean preAssignBe;

    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleClient batchScheduleClient,
                        PassthroughClient passthroughClient,
                        DispatcherMetricsReporter metricsReporter) {
        this.fanoutService = fanoutService;
        this.subBatch = cfg.getSubBatchSpec();
        this.batchScheduleClient = batchScheduleClient;
        this.passthroughClient = passthroughClient;
        this.metricsReporter = metricsReporter;
        this.preAssignBe = cfg.isPreAssignBe();
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        AtomicBoolean delegatedToPassthrough = new AtomicBoolean(false);
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0]).flatMap(bytes -> {
            JSONObject body = BatchBodyParser.parseObject(bytes);
            if (body == null) {
                return badRequest("expected a JSON object body");
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
            if (arr.isEmpty()) {
                JSONObject emptyEnvelope = new JSONObject();
                emptyEnvelope.put(spec.getResponseArrayField(), new JSONArray());
                return DispatcherResponses.jsonBytes(200, BatchBodyParser.serialize(emptyEnvelope));
            }
            pv.setTotalItems(arr.size());
            List<JSONArray> chunks = BatchChunkAssembler.split(arr, subBatch);
            pv.setChunkCount(chunks.size());
            List<JSONObject> chunkBodies = BatchChunkAssembler.buildChunkBodies(
                    body, chunks, spec.getRequestArrayField());
            return resolvePreAssignedTargets(spec, chunks.size())
                    .flatMap(targets -> {
                        BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
                        long fanoutStart = System.currentTimeMillis();
                        return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                                .doOnNext(subs -> metricsReporter.reportFanoutRt(
                                        System.currentTimeMillis() - fanoutStart))
                                .map(subs -> ResponseMerger.merge(subs, spec))
                                .flatMap(merged -> {
                                    pv.setFailedChunks(merged.failedReasons().size());
                                    if (merged.allFailed()) {
                                        return errorResponse(merged);
                                    }
                                    return DispatcherResponses.jsonBytes(
                                            200, BatchBodyParser.serialize(merged.body()));
                                });
                    });
        }).onErrorResume(e -> {
            String errMsg = DispatcherResponses.briefReason(e);
            Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
            pv.setError(errMsg);
            if (e instanceof DataBufferLimitException) {
                // Body over spring.codec.max-in-memory-size is a deterministic client error;
                // a 500 would invite pointless retries and pollute the server error rate.
                return DispatcherResponses.error(413, "request_body_too_large",
                        "batch body exceeds the server limit; see MAX_IN_MEMORY_SIZE");
            }
            return DispatcherResponses.error(500, "dispatch_failed", String.valueOf(e.getMessage()));
        }).doOnNext(resp -> pv.setHttpStatus(resp.rawStatusCode()))
          .doFinally(signal -> {
              if (!delegatedToPassthrough.get()) {
                  finalizePvRecord(pv, signal);
              }
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

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_batch_request", message);
    }

    private Mono<ServerResponse> errorResponse(ResponseMerger.MergedResponse merged) {
        JSONObject body = new JSONObject();
        body.put("error", "all_sub_batches_failed");
        // Item units, matching the success-path _partial_failure block; every item failed so
        // failed_count == total_count. total_chunks is sub-batch units.
        int failedItems = merged.failedIndices().size();
        body.put("failed_count", failedItems);
        body.put("total_count", failedItems);
        body.put("total_chunks", merged.totalChunks());
        JSONArray reasons = new JSONArray();
        merged.failedReasons().stream().distinct().forEach(reasons::add);
        body.put("failed_reasons", reasons);
        return DispatcherResponses.jsonBytes(merged.errorStatus(), BatchBodyParser.serialize(body));
    }

    /**
     * Resolves pre-assigned BE targets when both the config toggle and the endpoint support it;
     * otherwise short-circuits to an empty list without touching master (no {@code /batch_schedule}
     * round-trip, no RR cursor advance, no pre-assign metric for a path FE would ignore).
     */
    private Mono<List<BatchScheduleTarget>> resolvePreAssignedTargets(BatchEndpointSpec spec,
                                                                      int chunkCount) {
        if (!preAssignBe || !spec.isPreAssignable()) {
            return Mono.just(List.of());
        }
        long start = System.currentTimeMillis();
        return batchScheduleClient.requestTargets(chunkCount)
                .doOnNext(targets -> metricsReporter.reportPreassignRt(
                        System.currentTimeMillis() - start, !targets.isEmpty()));
    }
}
