package org.flexlb.mockengine;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Cluster-owned execution keeps simulated engine methods off shared Netty I/O threads. */
final class MockRpcExecutors implements AutoCloseable {
    private final ThreadPoolExecutor calls;

    MockRpcExecutors(int threads, int queuedCalls) {
        AtomicInteger index = new AtomicInteger();
        calls = new ThreadPoolExecutor(threads, threads, 0, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(queuedCalls), task -> {
                    Thread thread = new Thread(task, "mock-rpc-" + index.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                }, new ThreadPoolExecutor.AbortPolicy());
    }

    Executor executor() {
        return calls;
    }

    @Override
    public void close() {
        calls.shutdownNow();
    }
}
