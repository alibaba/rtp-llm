package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

public class DispatchHandler {

    private final FanoutService fanoutService;
    private final PassthroughClient passthroughClient;
    private final ObjectMapper mapper;

    public DispatchHandler(FanoutService fanoutService, PassthroughClient passthroughClient, ObjectMapper mapper) {
        this.fanoutService = fanoutService;
        this.passthroughClient = passthroughClient;
        this.mapper = mapper;
    }

    public Mono<ServerResponse> handleBatch(ServerRequest request) {
        return request.bodyToMono(JsonNode.class).flatMap(body -> {
            List<String> prompts = new ArrayList<>();
            JsonNode arr = body.get("prompt_batch");
            if (arr != null && arr.isArray()) {
                arr.forEach(n -> prompts.add(n.asText()));
            }
            JsonNode generateConfig = body.get("generate_config");
            return fanoutService.dispatch(prompts, generateConfig).flatMap(merged -> {
                if (merged.allFailed()) {
                    ObjectNode err = mapper.createObjectNode();
                    err.put("error", "all_sub_batches_failed");
                    return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
                }
                return ServerResponse.ok().contentType(MediaType.APPLICATION_JSON).bodyValue(merged.body());
            });
        }).onErrorResume(e -> {
            // Fanout itself never errors (each chunk soft-fails); this guards genuine pre-flight
            // failures only — e.g. an unparseable request body.
            ObjectNode err = mapper.createObjectNode();
            err.put("error", "dispatch_failed");
            err.put("message", String.valueOf(e.getMessage()));
            return ServerResponse.status(500).contentType(MediaType.APPLICATION_JSON).bodyValue(err);
        });
    }

    public Mono<ServerResponse> handlePassthrough(ServerRequest request) {
        return passthroughClient.forward(request);
    }
}
