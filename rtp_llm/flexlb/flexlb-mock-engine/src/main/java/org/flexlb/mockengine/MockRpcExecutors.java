package org.flexlb.mockengine;

import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallExecutorSupplier;
import org.flexlb.engine.grpc.RpcServiceGrpc;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Cluster-owned bounded execution; cache snapshots cannot occupy status or Netty threads. */
final class MockRpcExecutors implements ServerCallExecutorSupplier, AutoCloseable {
    private final ThreadPoolExecutor calls;
    private final ThreadPoolExecutor cache;

    MockRpcExecutors(int threads, int queuedCalls) {
        calls = pool(threads, queuedCalls, "mock-rpc");
        cache = pool(threads, queuedCalls, "mock-cache-rpc");
    }

    private static ThreadPoolExecutor pool(int threads, int capacity, String name) {
        AtomicInteger index = new AtomicInteger();
        return new ThreadPoolExecutor(threads, threads, 0, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(capacity), task -> {
                    Thread thread = new Thread(task, name + "-" + index.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                }, new ThreadPoolExecutor.AbortPolicy());
    }

    @Override
    public <ReqT, RespT> Executor getExecutor(ServerCall<ReqT, RespT> call, Metadata headers) {
        return executor(call.getMethodDescriptor().getFullMethodName());
    }

    Executor executor(String method) {
        return method.equals(RpcServiceGrpc.getGetCacheStatusMethod().getFullMethodName()) ? cache : calls;
    }

    @Override
    public void close() {
        calls.shutdownNow();
        cache.shutdownNow();
    }
}
