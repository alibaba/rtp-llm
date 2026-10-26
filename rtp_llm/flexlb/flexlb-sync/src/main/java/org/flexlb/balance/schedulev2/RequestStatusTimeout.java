package org.flexlb.balance.schedulev2;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;

/**
 * 请求状态超时配置，单位为毫秒，独立于引擎状态轮询周期。
 */
public final class RequestStatusTimeout {

    private RequestStatusTimeout() {
    }

    public static long timeoutMs(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        return config.getRequestLifecycle().getRequest().getStatusUpdateTimeoutMs();
    }
}
