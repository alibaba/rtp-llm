package org.flexlb.dao;

import com.taobao.eagleeye.RpcContext_inner;
import lombok.Data;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.trace.NoopSpanImpl;
import org.flexlb.trace.WhaleSpan;

/**
 * 请求推理上下文
 */
@Data
@Slf4j
@ToString
public class RequestContext {

    private static final WhaleSpan NOOP_SPAN = new NoopSpanImpl();

    //======================== Basic ========================//

    private String engineFramework;

    private String requestId = "";

    private String dashScopeRequestId;

    protected String model = "";

    protected String physicalServiceId = "";

    private boolean privateRequest;

    //======================== Route ========================//

    private ServiceRoute serviceRoute;

    //===================== trace and log ===================//

    private RpcContext_inner eagleTraceCtx;

    private WhaleSpan span = NOOP_SPAN;
    /**
     * the traceId passed from the upstream application
     */
    private String originTraceId;

    private Integer resBufferLength;

    private Long reqContentLength;

    private String traceUserData;

    private String rpcId;

    private String otlpTraceParent;

    private String otlpTraceState;
}
