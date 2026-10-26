package org.flexlb.engine.grpc.config;

import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;

/** gRPC callbacks transfer message ownership and must execute, including during shutdown. */
final class GrpcCallbackCallerRunsPolicy implements RejectedExecutionHandler {
    @Override
    public void rejectedExecution(Runnable task, ThreadPoolExecutor executor) {
        // JDK CallerRunsPolicy silently drops tasks after shutdown. A racing
        // terminal callback must still close its message and settle its future.
        task.run();
    }
}
