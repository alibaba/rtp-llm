package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
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
@Slf4j
@RequiredArgsConstructor
public class GenericBatchHandler {

    private final FanoutService fanoutService;
    private final ObjectMapper mapper;
    private final int subBatchSize;

    public Mono<ServerResponse> handle(ServerRequest request, BatchEndpointSpec spec) {
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
            List<ArrayNode> chunks = BatchSplitter.splitArray((ArrayNode) arr, subBatchSize, mapper);
            List<ObjectNode> chunkBodies = new ArrayList<>(chunks.size());
            for (ArrayNode chunk : chunks) {
                ObjectNode copy = obj.deepCopy();
                copy.set(spec.getRequestArrayField(), chunk);
                chunkBodies.add(copy);
            }
            return fanoutService.dispatchChunks(spec.getPath(), chunkBodies, spec)
                    .map(subs -> PartialFailureMerger.merge(subs, spec, mapper))
                    .flatMap(merged -> merged.allFailed()
                            ? errorResponse(merged)
                            : ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body()));
        }).onErrorResume(e -> {
            log.warn("dispatcher request failed: spec={}", spec.getPath(), e);
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
        });
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
