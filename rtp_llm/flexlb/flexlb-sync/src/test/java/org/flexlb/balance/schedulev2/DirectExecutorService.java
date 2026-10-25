package org.flexlb.balance.schedulev2;

import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * 在调用线程执行任务，便于逐轮验证调度规则；线程并行和关闭竞争另用真实线程池测试。
 */
public final class DirectExecutorService extends AbstractExecutorService {

    private volatile boolean closed;

    @Override
    public void shutdown() {
        closed = true;
    }

    @Override
    public List<Runnable> shutdownNow() {
        shutdown();
        return List.of();
    }

    @Override
    public boolean isShutdown() {
        return closed;
    }

    @Override
    public boolean isTerminated() {
        return closed;
    }

    @Override
    public boolean awaitTermination(long timeout, TimeUnit unit) {
        return closed;
    }

    @Override
    public void execute(Runnable command) {
        if (closed) {
            throw new RejectedExecutionException("executor closed");
        }
        command.run();
    }
}
