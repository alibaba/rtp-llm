package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

/** Shared response/error helpers for the dispatcher handlers. */
final class DispatcherResponses {

    private DispatcherResponses() {}

    static Mono<ServerResponse> jsonBytes(int status, byte[] body) {
        return ServerResponse.status(status).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }

    static Mono<ServerResponse> error(int status, String code, String message) {
        return jsonBytes(status, BatchBodyParser.serialize(JSONObject.of("error", code, "message", message)));
    }

    /** FE response status when the failure is a {@link WebClientResponseException}, else 0. */
    static int httpStatusOf(Throwable e) {
        return e instanceof WebClientResponseException w ? w.getRawStatusCode() : 0;
    }
}
