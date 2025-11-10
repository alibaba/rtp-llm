package org.flexlb.dao;

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

    private Integer resBufferLength;

    private Long reqContentLength;

    //======================== Route ========================//

    private ServiceRoute serviceRoute;

    //===================== trace and log ===================//

    private WhaleSpan span = NOOP_SPAN;

    private String otlpTraceParent;

    private String otlpTraceState;
}
