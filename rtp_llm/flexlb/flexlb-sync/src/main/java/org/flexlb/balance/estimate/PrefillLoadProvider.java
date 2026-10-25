package org.flexlb.balance.estimate;

import org.flexlb.balance.endpoint.PrefillEndpoint;

/**
 * 提供一台 Prefill 的负载视图，供选 Prefill 时预估 TTFT。
 * <p>
 * 不同调度链路记录排队和在途请求的方式不同，各自实现这个接口。
 */
public interface PrefillLoadProvider {

    /**
     * 返回这台 Prefill 当前的负载视图。
     */
    PrefillTtftEstimate.LoadView loadView(PrefillEndpoint endpoint);
}
