package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
public class DispatchHandler {

    private final FanoutService fanoutService;
    private final PassthroughClient passthroughClient;
    private final ObjectMapper mapper;

    public Mono<ServerResponse> handleBatch(ServerRequest request) {
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            List<String> prompts = new ArrayList<>();
            JsonNode arr = body.get(DispatchProtocol.FIELD_PROMPT_BATCH);
            if (arr != null && arr.isArray()) {
                arr.forEach(n -> prompts.add(n.asText()));
            }
            JsonNode generateConfig = body.get(DispatchProtocol.FIELD_GENERATE_CONFIG);
            return fanoutService.dispatch(prompts, generateConfig).flatMap(merged -> {
                if (merged.allFailed()) {
                    log.warn("batch dispatch failed: all {} sub-batches failed", merged.totalChunks());
                    return errorResponse("all_sub_batches_failed",
                            "all " + merged.totalChunks() + " sub-batches failed");
                }
                return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body());
            });
        }).onErrorResume(e -> {
            log.warn("batch dispatch failed before fanout", e);
            return errorResponse("dispatch_failed", String.valueOf(e.getMessage()));
        });
    }

    private Mono<ServerResponse> errorResponse(String error, String message) {
        ObjectNode body = mapper.createObjectNode();
        body.put("error", error);
        body.put("message", message);
        return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }

    public Mono<ServerResponse> handlePassthrough(ServerRequest request) {
        return passthroughClient.forward(request);
    }
}
