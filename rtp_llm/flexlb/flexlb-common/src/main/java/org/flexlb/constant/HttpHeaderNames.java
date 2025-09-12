package org.flexlb.constant;

public class HttpHeaderNames {

    public static final String CONTENT_LENGTH = "Content-Length";

    public static final String EAGLE_EYE_TRACE_ID = "EagleEye-TraceId";
    public static final String EAGLE_EYE_RPC_ID = "EagleEye-RpcId";  
    public static final String EAGLE_EYE_USER_DATA = "EagleEye-UserData";
    
    // Standard tracing headers for open source compatibility
    public static final String X_TRACE_ID = "X-Trace-Id";
    public static final String X_SPAN_ID = "X-Span-Id";
    public static final String X_REQUEST_ID = "X-Request-Id";

    public static final String TRACE_PARENT = "traceparent";

    public static final String TRACE_STATE = "tracestate";

}
