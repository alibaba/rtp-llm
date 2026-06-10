package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;

import java.util.List;

/**
 * Dispatcher batch handler. Reads each batch request body as raw bytes, parses with
 * fastjson2, splits the request array per {@link SubBatchSpec}, builds per-chunk bodies,
 * stamps any pre-assigned BE targets, fans out via {@link FanoutService}, and merges with
 * {@link ResponseMerger}.
 *
 * <p>Status mapping: 400 on missing/non-array field, 500 only when every sub-batch failed,
 * 200 on full or partial success.
 *
 * <p>Single-element batches still fan out as one chunk so partial-failure semantics stay
 * uniform; router-level rejection of non-batch traffic happens upstream.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class BatchHandler {

    /** Per-request access log, shared with {@code /schedule} / {@code /batch_schedule} → pv.log. */
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final FanoutService fanoutService;
    private final SubBatchSpec subBatch;
    private final BatchScheduleClient batchScheduleClient;
    private final boolean preAssignBe;

    public BatchHandler(FanoutService fanoutService,
                        DispatchConfig cfg,
                        BatchScheduleClient batchScheduleClient) {
        this.fanoutService = fanoutService;
        this.subBatch = cfg.getSubBatchSpec();
        this.batchScheduleClient = batchScheduleClient;
        this.preAssignBe = cfg.isPreAssignBe();
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        return request.bodyToMono(byte[].class).defaultIfEmpty(new byte[0]).flatMap(bytes -> {
            JSONObject body = BatchBodyParser.parseObject(bytes);
            if (body == null) {
                return badRequest("expected a JSON object body");
            }
            JSONArray arr = BatchBodyParser.findArrayField(body, spec.getRequestArrayField());
            if (arr == null) {
                return badRequest("missing or non-array field: " + spec.getRequestArrayField());
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
            return resolvePreAssignedTargets(chunks.size())
                    .flatMap(targets -> {
                        BatchChunkAssembler.stampPreAssignedBe(chunkBodies, targets);
                        return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                                .map(subs -> ResponseMerger.merge(subs, spec))
                                .flatMap(merged -> {
                                    pv.setFailedChunks(merged.failedIndices().size());
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
            return DispatcherResponses.error(500, "dispatch_failed", String.valueOf(e.getMessage()));
        }).doOnNext(resp -> pv.setHttpStatus(resp.statusCode().value()))
          .doFinally(signal -> finalizePvRecord(pv, signal));
    }

    private void finalizePvRecord(DispatchPvLogData pv, SignalType signal) {
        int status = pv.getHttpStatus();
        String error = pv.getError();
        if (signal == SignalType.CANCEL && status == 0) {
            status = 499;
            error = error != null ? error : "client cancelled";
        }
        pv.finish(status, error);
        pv.emit(pvLogger);
    }

    private Mono<ServerResponse> badRequest(String message) {
        return DispatcherResponses.error(400, "invalid_batch_request", message);
    }

    private Mono<ServerResponse> errorResponse(ResponseMerger.MergedResponse merged) {
        JSONObject body = new JSONObject();
        body.put("error", "all_sub_batches_failed");
        body.put("failed_count", merged.failedIndices().size());
        body.put("total_chunks", merged.totalChunks());
        JSONArray reasons = new JSONArray();
        merged.failedReasons().stream().distinct().forEach(reasons::add);
        body.put("failed_reasons", reasons);
        return DispatcherResponses.jsonBytes(500, BatchBodyParser.serialize(body));
    }

    private Mono<List<BatchScheduleTarget>> resolvePreAssignedTargets(int chunkCount) {
        return preAssignBe ? batchScheduleClient.requestTargets(chunkCount) : Mono.just(List.of());
    }
}
