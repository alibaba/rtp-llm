package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * One handler for every batch endpoint. Parses the body, splits {@code spec.requestArrayField}
 * into chunks of at most {@code subBatchSize}, fans out one chunk per FE call, and merges with
 * {@link PartialFailureMerger}. Returns 200 on partial success, 500 only on
 * {@link PartialFailureMerger.MergedResponse#allFailed()}, and 400 when the body is not a JSON object or the
 * requested array field is missing/non-array (a request that landed on a batch path but doesn't
 * shape like a batch is a client bug, not something to silently relay to FE).
 *
 * <p>Single-element batches fanout as one chunk — same merge path, same status mapping —
 * rather than being forwarded verbatim. The router is responsible for sending non-batch traffic
 * to passthrough; by the time we're inside this handler the request body has already been
 * consumed by {@link ServerRequest#bodyToMono}, so re-forwarding through {@code WebClient}
 * would lose the body.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class GenericBatchHandler {

    /** Per-request access log, shared with {@code /schedule} / {@code /batch_schedule} → pv.log. */
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final FanoutService fanoutService;
    private final ObjectMapper mapper;
    private final SubBatchSpec subBatch;
    private final BatchScheduleClient batchScheduleClient;
    private final boolean preAssignBe;

    public GenericBatchHandler(FanoutService fanoutService,
                               ObjectMapper mapper,
                               DispatchConfig cfg,
                               BatchScheduleClient batchScheduleClient) {
        this.fanoutService = fanoutService;
        this.mapper = mapper;
        this.subBatch = cfg.subBatchSpec();
        this.batchScheduleClient = batchScheduleClient;
        this.preAssignBe = cfg.isPreAssignBe();
    }

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
        DispatchPvLogData pv = DispatchPvLogData.batch(spec.getPath(), System.currentTimeMillis());
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            if (!(body instanceof ObjectNode obj)) {
                return badRequest("expected a JSON object body");
            }
            JsonNode arr = obj.get(spec.getRequestArrayField());
            if (arr == null || !arr.isArray()) {
                return badRequest("missing or non-array field: " + spec.getRequestArrayField());
            }
            if (arr.isEmpty()) {
                ObjectNode emptyEnvelope = mapper.createObjectNode();
                emptyEnvelope.set(spec.getResponseArrayField(), mapper.createArrayNode());
                return json(200, emptyEnvelope);
            }
            pv.setTotalItems(arr.size());
            List<ArrayNode> chunks = BatchSplitter.split((ArrayNode) arr, subBatch, mapper);
            pv.setChunkCount(chunks.size());
            List<ObjectNode> chunkBodies = BatchChunkBuilder.buildChunkBodies(
                    obj, chunks, spec.getRequestArrayField());
            return resolvePreAssignedTargets(chunks.size())
                    .flatMap(targets -> {
                        BatchChunkBuilder.stampPreAssignedBe(chunkBodies, targets);
                        return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                                .map(subs -> PartialFailureMerger.merge(subs, spec, mapper))
                                .flatMap(merged -> {
                                    pv.setFailedChunks(merged.failedIndices().size());
                                    return merged.allFailed() ? errorResponse(merged) : json(200, merged.body());
                                });
                    });
        }).onErrorResume(e -> {
            String errMsg = e.getClass().getSimpleName() + ": " + e.getMessage();
            Logger.warn("dispatcher request failed: spec={}, err={}", spec.getPath(), errMsg);
            pv.setError(errMsg);
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return json(500, err);
        }).doOnNext(resp -> pv.setHttpStatus(resp.statusCode().value()))
          .doFinally(signal -> finalizePvRecord(pv, signal));
    }

    /**
     * Finalize on every terminal signal (COMPLETE / ERROR / CANCEL) so a client-disconnect
     * before {@code writeTo} subscribes still produces a pv.log line. Mirrors
     * {@code HttpLoadBalanceServer}'s {@code doFinally(signal -> finalizeXxx(...))} pattern;
     * cancel events arrive with httpStatus=0 and are stamped 499 so operators can discriminate
     * client-cancel from server-side errors.
     */
    private void finalizePvRecord(DispatchPvLogData pv, reactor.core.publisher.SignalType signal) {
        int status = pv.getHttpStatus();
        String error = pv.getError();
        if (signal == reactor.core.publisher.SignalType.CANCEL && status == 0) {
            status = 499;
            error = error != null ? error : "client cancelled";
        }
        pv.finish(status, error);
        pv.emit(pvLogger);
    }

    /** Wraps {@link ServerResponse#status(int)} + JSON content type + body in one call. */
    private static Mono<ServerResponse> json(int status, Object body) {
        return ServerResponse.status(status).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }

    private Mono<ServerResponse> badRequest(String message) {
        ObjectNode err = mapper.createObjectNode();
        err.put("error", "invalid_batch_request");
        err.put("message", message);
        return json(400, err);
    }

    private Mono<ServerResponse> errorResponse(PartialFailureMerger.MergedResponse merged) {
        ObjectNode body = mapper.createObjectNode();
        body.put("error", "all_sub_batches_failed");
        body.put("failed_count", merged.failedIndices().size());
        body.put("total_chunks", merged.totalChunks());
        ArrayNode reasons = body.putArray("failed_reasons");
        merged.failedReasons().stream().distinct().forEach(reasons::add);
        return json(500, body);
    }

    /**
     * Resolves N pre-assigned BE targets via {@link BatchScheduleClient} when
     * {@link #preAssignBe} is on, returning {@link List#of()} otherwise. The client itself
     * collapses every failure path to an empty list, so the caller never has to handle
     * errors here — an empty list simply means "no stamping happens this round".
     */
    private Mono<List<BatchScheduleTarget>> resolvePreAssignedTargets(int chunkCount) {
        if (!preAssignBe || batchScheduleClient == null) {
            return Mono.just(List.of());
        }
        return batchScheduleClient.requestTargets(chunkCount);
    }

}
