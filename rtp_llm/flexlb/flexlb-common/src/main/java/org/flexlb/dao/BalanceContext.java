package org.flexlb.dao;

import lombok.Data;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.QueuedRequest;
import org.flexlb.dao.pv.PvLogData;
import org.flexlb.trace.NoopSpanImpl;
import org.flexlb.trace.WhaleSpan;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
public class BalanceContext {

    private static final WhaleSpan NOOP_SPAN = new NoopSpanImpl();

    //======================== Basic ========================//

    private WhaleMasterConfig config;

    private MasterRequest masterRequest;

    private QueuedRequest queuedRequest;

    private MasterResponse masterResponse;

    //======================== Meters =====================//

    private long startTime = System.nanoTime() / 1000;

    private long interRequestId;

    private boolean success = true;

    //===================== trace and log ===================//

    private PvLogData pvLogData;

    private WhaleSpan span = NOOP_SPAN;

    private String otlpTraceParent;

    private String otlpTraceState;

}
