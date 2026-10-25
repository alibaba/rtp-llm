package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;

/**
 * 主队列给请求选好的 Prefill。
 *
 * @param endpoint         选中的 Prefill。同一地址重新上线后是另一个对象，dispatch 前据此判断机器是否还是选中的那台
 * @param serverStatus     返回给调用方的 Prefill 地址等信息
 * @param hitCacheTokens   在这台 Prefill 上命中缓存的 token 数
 * @param predictedMs      预估的 Prefill 耗时
 * @param commitTelemetry  dispatch 时上报的选路指标
 */
public record PrefillChoice(PrefillEndpoint endpoint,
                            ServerStatus serverStatus,
                            long hitCacheTokens,
                            long predictedMs,
                            Runnable commitTelemetry) {

    /**
     * 从选择结果取出需要的信息，并关闭选择结果持有的机器 hold，排队期间不阻止这台机器下线。
     */
    public static PrefillChoice from(SelectedRole selected, long seqLen) {
        WorkerEndpoint.EndpointHold hold = selected.takeEndpointHold();
        try {
            ServerStatus serverStatus = selected.serverStatus();
            DebugInfo debugInfo = serverStatus.getDebugInfo();
            long hitCache = debugInfo == null ? 0L : debugInfo.getHitCacheLen();
            long hitCacheTokens = Math.max(0L, Math.min(seqLen, hitCache));
            return new PrefillChoice((PrefillEndpoint) hold.endpoint(), serverStatus, hitCacheTokens,
                    selected.prefillWorkMs(), selected.takeCommitTelemetry());
        } finally {
            hold.close();
        }
    }

    /**
     * 还没算进引擎 KV 的部分：输入 token 数减去命中缓存的 token 数。
     */
    public long kvTokens(long seqLen) {
        return Math.max(0L, seqLen - hitCacheTokens);
    }
}
