package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

/**
 * gRPC server interceptor that extracts the Auto-TPM QoS level carried in the
 * {@code x-dashscope-inner-qos-level} metadata header.
 *
 * <p>The raw header value is propagated via gRPC {@link Context} so that
 * {@code FlexlbServiceImpl.buildContext} can normalize it together with the
 * proto {@code priority} field (proto valid value &gt; metadata header &gt;
 * default). The header channel is a fallback only — current callers set the
 * proto field.
 */
@Component
public class GrpcQosHeaderInterceptor implements ServerInterceptor {

    private static final Metadata.Key<String> QOS_HEADER_KEY =
            Metadata.Key.of(PriorityNormalizer.QOS_HEADER_NAME, Metadata.ASCII_STRING_MARSHALLER);

    /**
     * Context key carrying the raw QoS header value (may be null when the
     * header is absent or the call bypassed the interceptor).
     */
    public static final Context.Key<String> QOS_LEVEL_KEY = Context.key("qosLevel");

    /**
     * Convenience method to retrieve the raw QoS header value from the active
     * context. Returns {@code null} when unset.
     */
    public static String get() {
        return QOS_LEVEL_KEY.get();
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers,
            ServerCallHandler<ReqT, RespT> next) {
        String qosLevel = headers.get(QOS_HEADER_KEY);
        if (qosLevel == null) {
            return next.startCall(call, headers);
        }
        Context ctx = Context.current().withValue(QOS_LEVEL_KEY, qosLevel);
        return Contexts.interceptCall(ctx, call, headers, next);
    }
}
