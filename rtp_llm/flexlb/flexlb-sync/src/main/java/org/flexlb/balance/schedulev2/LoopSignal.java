package org.flexlb.balance.schedulev2;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * 唤醒一个循环线程。
 * <p>
 * 有新请求或名额被归还时调用 {@link #wake()}；循环线程没事可做时调用 {@link #await(long)}。
 * 在 await 之前到达的 wake 不会丢：pending 标记会让下一次 await 立刻返回。
 */
public final class LoopSignal {

    private final AtomicBoolean pending = new AtomicBoolean();
    private volatile Thread waiter;

    public void wake() {
        pending.set(true);
        Thread thread = waiter;
        if (thread != null) {
            LockSupport.unpark(thread);
        }
    }

    /**
     * 等到被唤醒，最多等 timeoutMs 毫秒。只允许循环线程自己调用。
     */
    public void await(long timeoutMs) {
        waiter = Thread.currentThread();
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (!pending.getAndSet(false)) {
            long remainingNanos = deadlineNanos - System.nanoTime();
            if (remainingNanos <= 0L || Thread.currentThread().isInterrupted()) {
                return;
            }
            LockSupport.parkNanos(this, remainingNanos);
        }
    }
}
