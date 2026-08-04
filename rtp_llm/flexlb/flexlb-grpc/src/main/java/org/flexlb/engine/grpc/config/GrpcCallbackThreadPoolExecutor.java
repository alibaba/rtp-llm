package org.flexlb.engine.grpc.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.concurrent.RejectedTaskCountProvider;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

@Slf4j
public class GrpcCallbackThreadPoolExecutor extends ThreadPoolExecutor implements RejectedTaskCountProvider {

    private static final long REJECTION_LOG_INTERVAL_MILLIS = TimeUnit.MINUTES.toMillis(1);

    private final LongAdder rejectedTaskCount = new LongAdder();
    private final LongAdder suppressedRejectionLogCount = new LongAdder();
    private final AtomicLong lastRejectionLogTimeMillis = new AtomicLong();

    public GrpcCallbackThreadPoolExecutor(
            int corePoolSize,
            int maximumPoolSize,
            long keepAliveTime,
            TimeUnit unit,
            BlockingQueue<Runnable> workQueue,
            ThreadFactory threadFactory) {
        super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory);
    }

    @Override
    public void execute(Runnable command) {
        try {
            super.execute(command);
        } catch (RejectedExecutionException exception) {
            if (!isShutdown()) {
                rejectedTaskCount.increment();
                logRejection(exception);
            }
            throw exception;
        }
    }

    private void logRejection(RejectedExecutionException exception) {
        long currentTimeMillis = System.currentTimeMillis();
        long previousLogTimeMillis = lastRejectionLogTimeMillis.get();
        if (previousLogTimeMillis != 0
                && currentTimeMillis - previousLogTimeMillis < REJECTION_LOG_INTERVAL_MILLIS) {
            suppressedRejectionLogCount.increment();
            return;
        }
        if (!lastRejectionLogTimeMillis.compareAndSet(previousLogTimeMillis, currentTimeMillis)) {
            suppressedRejectionLogCount.increment();
            return;
        }
        log.error("gRPC callback executor rejected task: activeCount={}, poolSize={}, maximumPoolSize={}, "
                        + "largestPoolSize={}, completedTaskCount={}, taskCount={}, suppressedRejectionCount={}",
                getActiveCount(), getPoolSize(), getMaximumPoolSize(), getLargestPoolSize(),
                getCompletedTaskCount(), getTaskCount(), suppressedRejectionLogCount.sumThenReset(), exception);
    }

    public long getRejectedTaskCount() {
        return rejectedTaskCount.sum();
    }
}
