package org.flexlb.balance.estimate;

import org.flexlb.balance.endpoint.PrefillEndpoint;

/**
 * 从 PrefillEndpoint 自带的队列和在途记录读取负载视图，BATCH 和 NON_BATCH 使用。
 */
public final class EndpointPrefillLoadProvider implements PrefillLoadProvider {

    @Override
    public PrefillTtftEstimate.LoadView loadView(PrefillEndpoint endpoint) {
        return endpoint.captureLoadView();
    }
}
