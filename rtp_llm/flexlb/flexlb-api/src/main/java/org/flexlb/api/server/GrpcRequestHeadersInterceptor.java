package org.flexlb.api.server;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

/**
 * Captures the incoming request metadata for local QoS normalization and forwarding to the Master.
 */
@Component
class GrpcRequestHeadersInterceptor implements ServerInterceptor {
    private static final Metadata.Key<String> QOS_HEADER_KEY =
            Metadata.Key.of(PriorityNormalizer.QOS_HEADER_NAME, Metadata.ASCII_STRING_MARSHALLER);
    private static final Context.Key<Metadata> REQUEST_HEADERS =
            Context.key("flexlbRequestHeaders");

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
        // The next handler takes ownership of the original Metadata instance.
        Metadata requestHeaders = new Metadata();
        requestHeaders.merge(headers);
        Context ctx = Context.current().withValue(REQUEST_HEADERS, requestHeaders);
        return Contexts.interceptCall(ctx, call, headers, next);
    }

    static String qosLevel() {
        Metadata headers = REQUEST_HEADERS.get();
        return headers == null ? null : headers.get(QOS_HEADER_KEY);
    }

    static Metadata copyRequestHeaders() {
        Metadata headers = REQUEST_HEADERS.get();
        Metadata copy = new Metadata();
        if (headers != null) {
            copy.merge(headers);
        }
        return copy;
    }
}
