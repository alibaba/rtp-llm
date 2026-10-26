package org.flexlb.balance.schedulev2;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;

/**
 * 请求发布后或最近一次状态更新后，允许多久没有该请求的引擎状态。
 * <p>
 * 时长由 requestLifecycle.request.statusUpdateTimeoutMs 独立配置，单位为毫秒。
 * 超过这个时间未再收到匹配的活跃或完成状态，V2 归还本地 Prefill 在途记录和 Decode reserve。
 * 本地回收不代表引擎未收到请求或请求已完成。
 */
public final class RequestStatusTimeout {

    private RequestStatusTimeout() {
    }

    public static long timeoutMs(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        return config.getRequestLifecycle().getRequest().getStatusUpdateTimeoutMs();
    }

}
