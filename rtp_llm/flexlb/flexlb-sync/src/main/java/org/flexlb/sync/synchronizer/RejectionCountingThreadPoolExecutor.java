package org.flexlb.sync.synchronizer;

import org.flexlb.concurrent.RejectedTaskCountProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

public class RejectionCountingThreadPoolExecutor extends ThreadPoolExecutor
        implements RejectedTaskCountProvider {

    private static final Logger logger = LoggerFactory.getLogger(RejectionCountingThreadPoolExecutor.class);
    private static final long REJECTION_LOG_INTERVAL_MILLIS = TimeUnit.MINUTES.toMillis(1);

    private final LongAdder rejectedTaskCount = new LongAdder();
    private final LongAdder suppressedRejectionLogCount = new LongAdder();
    private final AtomicLong lastRejectionLogTimeMillis = new AtomicLong();

    public RejectionCountingThreadPoolExecutor(
            int corePoolSize,
            int maximumPoolSize,
            long keepAliveTime,
            TimeUnit unit,
            BlockingQueue<Runnable> workQueue,
            ThreadFactory threadFactory,
            RejectedExecutionHandler rejectionHandler) {
        super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory, rejectionHandler);
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
        logger.error("Synchronization executor rejected task: activeCount={}, poolSize={}, maximumPoolSize={}, "
                        + "largestPoolSize={}, completedTaskCount={}, taskCount={}, suppressedRejectionCount={}",
                getActiveCount(), getPoolSize(), getMaximumPoolSize(), getLargestPoolSize(),
                getCompletedTaskCount(), getTaskCount(), suppressedRejectionLogCount.sumThenReset(), exception);
    }

    @Override
    public long getRejectedTaskCount() {
        return rejectedTaskCount.sum();
    }
}
