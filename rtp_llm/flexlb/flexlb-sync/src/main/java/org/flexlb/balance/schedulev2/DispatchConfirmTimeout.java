package org.flexlb.balance.schedulev2;

import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.config.ConfigService;

/**
 * 请求发布后，引擎要在多长时间内上报这个请求。
 * <p>
 * 时长由 requestLifecycle.request.statusUpdateTimeoutMs 配置，单位为毫秒，和老的调度一致。
 * 超过这个时间引擎还没上报，就认为调用方没把请求发给引擎，或者请求在两次上报之间就结束了，
 * V2 为它记的 Prefill 在途记录和 Decode reserve 会被归还。
 */
public final class DispatchConfirmTimeout {

    private DispatchConfirmTimeout() {
    }

    public static long timeoutMs(ConfigService configService) {
        return RequestStatusTimeout.timeoutMs(configService);
    }

    /**
     * 请求是否已经发布、并且发布后过了 timeoutMs。
     */
    public static boolean expired(Request request, long nowMs, long timeoutMs) {
        long publishedAtMs = request.publishedAtMs();
        return publishedAtMs > 0L && nowMs - publishedAtMs >= timeoutMs;
    }
}
