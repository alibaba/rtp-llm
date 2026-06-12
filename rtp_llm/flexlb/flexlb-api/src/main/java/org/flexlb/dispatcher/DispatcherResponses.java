package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

/**
 * Shared response/error helpers for the dispatcher handlers. Every dispatcher path answers
 * errors with the same {@code {error: <code>, message: <msg>}} envelope so callers parse one
 * shape regardless of which handler produced it.
 */
final class DispatcherResponses {

    private DispatcherResponses() {}

    static Mono<ServerResponse> jsonBytes(int status, byte[] body) {
        return ServerResponse.status(status).contentType(MediaType.APPLICATION_JSON).bodyValue(body);
    }

    static Mono<ServerResponse> error(int status, String code, String message) {
        JSONObject err = new JSONObject();
        err.put("error", code);
        err.put("message", message);
        return jsonBytes(status, BatchBodyParser.serialize(err));
    }

    /** {@code SimpleName: message} one-liner for WARN logs and failure reasons. */
    static String briefReason(Throwable e) {
        String m = e.getClass().getSimpleName();
        return e.getMessage() == null ? m : m + ": " + e.getMessage();
    }
}
