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

import java.util.ArrayList;
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
            List<ArrayNode> chunks = splitChunks((ArrayNode) arr);
            pv.setChunkCount(chunks.size());
            List<ObjectNode> chunkBodies = new ArrayList<>(chunks.size());
            for (ArrayNode chunk : chunks) {
                ObjectNode copy = obj.deepCopy();
                copy.set(spec.getRequestArrayField(), chunk);
                injectForceBatch(copy);
                chunkBodies.add(copy);
            }
            return resolvePreAssignedTargets(chunks.size())
                    .flatMap(targets -> {
                        stampPreAssignedBe(chunkBodies, targets);
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

    /**
     * Stamps {@code generate_config.force_batch=true} into each sub-batch body. The legacy
     * ft_proxy convention is that the dispatch layer — not the user — tags batch traffic with
     * {@code force_batch} so the per-chunk FE's {@code FIFOScheduler} groups the chunk's prompts
     * into a single scheduling slot instead of interleaving them with whatever else is queued.
     * Without this stamp, splitting a batch into chunks would be observationally equivalent to
     * the user issuing N independent requests, defeating the whole point of going through a
     * batch endpoint.
     *
     * <p>A user-supplied {@code force_batch} is honored verbatim: an explicit {@code false} is a
     * legitimate opt-out (e.g. for measuring scheduler interleaving) and must not be overwritten.
     * Any other fields under {@code generate_config} (temperature, max_new_tokens, …) are
     * preserved by the surrounding deep-copy.
     */
    private static void injectForceBatch(ObjectNode chunkBody) {
        JsonNode gcNode = chunkBody.get("generate_config");
        ObjectNode gc;
        if (gcNode instanceof ObjectNode existing) {
            gc = existing;
        } else {
            gc = chunkBody.putObject("generate_config");
        }
        if (!gc.has("force_batch")) {
            gc.put("force_batch", true);
        }
    }

    private List<ArrayNode> splitChunks(ArrayNode arr) {
        return switch (subBatch.mode()) {
            case SIZE -> BatchSplitter.splitArray(arr, subBatch.value(), mapper);
            case COUNT -> BatchSplitter.splitByCount(arr, subBatch.value(), mapper);
        };
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

    /**
     * Appends each chunk's pre-resolved BE target into {@code generate_config.role_addrs} —
     * the same field FE's existing
     * {@code rtp_llm.server.backend_rpc_server_visitor.route_ips} skips master on when set
     * (PD-disagg's prefill→decode handoff uses the same mechanism). No FE-side change
     * required for this stamping to take effect.
     *
     * <p>Wire shape per addr matches Python {@code rtp_llm.config.generate_config.RoleAddr}
     * exactly: {@code {role, ip, http_port, grpc_port}}. Note {@code ip} (not
     * {@code server_ip} as in {@link BatchScheduleTarget}'s wire shape) — the rename is
     * deliberate to align with the FE-side schema.
     *
     * <p>Tolerates a short target list (callers degrade to no-stamp if
     * {@link BatchScheduleClient} returns empty). User-supplied {@code role_addrs} entries
     * are preserved and the dispatcher's resolved target is appended after them.
     */
    private void stampPreAssignedBe(List<ObjectNode> chunkBodies, List<BatchScheduleTarget> targets) {
        if (targets.isEmpty()) {
            return;
        }
        int max = Math.min(chunkBodies.size(), targets.size());
        for (int i = 0; i < max; i++) {
            BatchScheduleTarget target = targets.get(i);
            ObjectNode chunkBody = chunkBodies.get(i);
            ObjectNode gc = chunkBody.get("generate_config") instanceof ObjectNode existing
                    ? existing
                    : chunkBody.putObject("generate_config");
            ArrayNode roleAddrs = gc.get("role_addrs") instanceof ArrayNode existingAddrs
                    ? existingAddrs
                    : gc.putArray("role_addrs");
            ObjectNode addr = roleAddrs.addObject();
            addr.put("role", target.getRole().name());
            addr.put("ip", target.getServerIp());
            addr.put("http_port", target.getHttpPort());
            addr.put("grpc_port", target.getGrpcPort());
        }
    }
}
