package org.flexlb.balance.schedulev2.prefill;

import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.schedulev2.Request;

/**
 * 已从 Prefill 队列 dispatch 出去、还没在引擎上结束的请求。
 * <p>
 * dispatch 时创建，phase 为 COMMITTED（引擎还没上报这个请求）；引擎上报后改成 ENGINE_QUEUED 或 ENGINE_RUNNING。
 * 引擎上报过、后来又不在引擎的任务列表里时，说明 Prefill 已结束，记录被移除；请求失败或取消时也会移除。
 * 请求发布后超过 {@link org.flexlb.balance.schedulev2.DispatchConfirmTimeout} 引擎还没上报过，记录也会被移除。
 */
public final class DispatchedRequest {

    private final Request request;
    /**
     * 这个请求在引擎上要新占的 KV token 数（输入 token 数减去命中缓存的部分）。
     */
    private final long kvTokens;
    private final long predictedMs;
    /**
     * 只由 Prefill 队列循环线程修改。
     */
    private volatile WorkView.Phase phase = WorkView.Phase.COMMITTED;

    public DispatchedRequest(Request request, long kvTokens, long predictedMs) {
        this.request = request;
        this.kvTokens = kvTokens;
        this.predictedMs = predictedMs;
    }

    public Request request() {
        return request;
    }

    public long kvTokens() {
        return kvTokens;
    }

    public long predictedMs() {
        return predictedMs;
    }

    public WorkView.Phase phase() {
        return phase;
    }

    public void setPhase(WorkView.Phase phase) {
        this.phase = phase;
    }

    /**
     * 引擎是否已经上报过这个请求。
     */
    public boolean seenByEngine() {
        return phase != WorkView.Phase.COMMITTED;
    }
}
