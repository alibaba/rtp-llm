package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.estimate.WorkView;

/**
 * 预占提交成功后返回给调用方的结果。
 *
 * <p>workerKeepAlive：调用方持有期间 worker 不会被清理，交付完调用 {@link #close()} 释放，只能释放一次。
 * precedingWork：提交时这个请求前面还排着的工作，用来估算等待时间。</p>
 */
public final class CommitResult implements AutoCloseable {
    public EndpointGenerationLifecycle.HandoffPermit workerKeepAlive;
    private final WorkView precedingWork;

    public CommitResult(EndpointGenerationLifecycle.HandoffPermit workerKeepAlive, WorkView precedingWork) {
        this.workerKeepAlive = workerKeepAlive;
        this.precedingWork = precedingWork;
    }

    /**
     * 提交时这个请求前面还排着的工作。
     */
    public WorkView precedingWork() {
        return precedingWork;
    }

    @Override
    public synchronized void close() {
        EndpointGenerationLifecycle.HandoffPermit exact = workerKeepAlive;
        if (exact == null) {
            return;
        }
        workerKeepAlive = null;
        exact.close();
    }
}
