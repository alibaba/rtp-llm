package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

final class DispatcherResponses {

    private DispatcherResponses() {}

    static Mono<ServerResponse> jsonBytes(int status, byte[] body) {
        return ServerResponse.status(status).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }

    static Mono<ServerResponse> error(int status, String code, String message) {
        return jsonBytes(status, BatchBodyParser.serialize(JSONObject.of("error", code, "message", message)));
    }

}
