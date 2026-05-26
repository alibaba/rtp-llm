package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

/**
 * One handler for every batch endpoint. Parses the body, splits {@code spec.requestArrayField}
 * into chunks of at most {@code subBatchSize}, fans out one chunk per FE call, and merges with
 * {@link PartialFailureMerger}. Returns 200 on partial success, 500 only on
 * {@link MergedResponse#allFailed()}, and 400 when the body is not a JSON object or the
 * requested array field is missing/non-array (a request that landed on a batch path but doesn't
 * shape like a batch is a client bug, not something to silently relay to FE).
 *
 * <p>Single-element batches fanout as one chunk — same merge path, same status mapping —
 * rather than being forwarded verbatim. The router is responsible for sending non-batch traffic
 * to passthrough; by the time we're inside this handler the request body has already been
 * consumed by {@link ServerRequest#bodyToMono}, so re-forwarding through {@code WebClient}
 * would lose the body.
 */
@RequiredArgsConstructor
public class GenericBatchHandler {

    /** Per-request access log, shared with {@code /schedule} / {@code /batch_schedule} → pv.log. */
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final FanoutService fanoutService;
    private final ObjectMapper mapper;
    private final SubBatchSpec subBatch;

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
                return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(emptyEnvelope);
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
            return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                    .map(subs -> PartialFailureMerger.merge(subs, spec, mapper))
                    .flatMap(merged -> {
                        pv.setFailedChunks(merged.failedIndices().size());
                        return merged.allFailed()
                                ? errorResponse(merged)
                                : ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body());
                    });
        }).onErrorResume(e -> {
            Logger.warn("dispatcher request failed: spec={}, err={}",
                    spec.getPath(), e.getClass().getSimpleName() + ": " + e.getMessage());
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
        }).doOnSuccess(resp -> {
            // doOnSuccess sees the ServerResponse whether it was built by the happy path,
            // badRequest(), errorResponse(), or onErrorResume — single emission point covers
            // every outcome. doOnError is unreachable because onErrorResume always recovers.
            pv.finish(resp.statusCode().value(), null);
            emitPv(pv);
        });
    }

    private static void emitPv(DispatchPvLogData pv) {
        try {
            String jsonLog = JsonUtils.toStringOrEmpty(pv);
            if (pv.isSuccess()) {
                pvLogger.info(jsonLog);
            } else {
                pvLogger.error(jsonLog);
            }
        } catch (Exception ex) {
            Logger.error("Failed to serialize dispatcher batch PV log data", ex);
        }
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
        return ServerResponse.badRequest().contentType(MediaType.APPLICATION_JSON).bodyValue(err);
    }

    private Mono<ServerResponse> errorResponse(MergedResponse merged) {
        ObjectNode body = mapper.createObjectNode();
        body.put("error", "all_sub_batches_failed");
        body.put("failed_count", merged.failedIndices().size());
        body.put("total_chunks", merged.totalChunks());
        return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }
}
