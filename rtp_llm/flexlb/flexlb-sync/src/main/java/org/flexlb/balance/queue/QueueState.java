package org.flexlb.balance.queue;

import org.flexlb.balance.endpoint.prefill.PrefillActiveIndex;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 一台 Prefill 的排队队列共用的状态：队列锁、排队中的请求、两个版本号和停止标志。
 * <p>
 * 入队、出队、投递、停止都在队列锁内修改这些状态，并通过同一个条件变量唤醒调度线程。
 * 版本号让调度线程和缓存的负载视图判断队列或调度输入有没有变过。
 */
public final class QueueState {

    private final ReentrantLock lock = new ReentrantLock();
    /**
     * 队列有新请求、请求被移走、容量释放或调度输入变化时，都通过它唤醒调度线程。
     * 等待条件的检查和唤醒都在队列锁内完成，所以调度线程检查完容量、还没开始等待时发生的释放也不会漏掉。
     */
    private final Condition changed = lock.newCondition();
    private final PrefillActiveIndex activeIndex;
    /**
     * 队列内容的版本号：入队、移除、投递和停止清空时加一。
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /**
     * 调度输入的版本号：worker 状态、耗时预测、投递容量阻塞变化时加一。
     * 只在队列锁内修改；读负载视图缓存时不加锁，所以用 volatile。
     */
    private volatile long schedulingInputVersion;
    private volatile boolean stopped;

    public QueueState(PrefillActiveIndex activeIndex) {
        this.activeIndex = activeIndex;
    }

    /**
     * 队列锁本身，交给共用这把锁的 PrefillState。
     */
    public ReentrantLock reentrantLock() {
        return lock;
    }

    public void lock() {
        lock.lock();
    }

    public void lockInterruptibly() throws InterruptedException {
        lock.lockInterruptibly();
    }

    public void unlock() {
        lock.unlock();
    }

    public boolean isLockedByCurrentThread() {
        return lock.isHeldByCurrentThread();
    }

    /**
     * 排队中的请求，按调度顺序排列。只在队列锁内读写。
     */
    public PrefillActiveIndex activeIndex() {
        return activeIndex;
    }

    public long queueVersion() {
        return queueVersion.get();
    }

    public void incrementQueueVersion() {
        queueVersion.incrementAndGet();
    }

    public long schedulingInputVersion() {
        return schedulingInputVersion;
    }

    /**
     * 记下调度输入变了，但不唤醒线程。缓存的负载视图和正在等待的调度决定会因此失效。
     */
    public void incrementSchedulingInputVersionUnderLock() {
        if (!lock.isHeldByCurrentThread()) {
            throw new IllegalStateException("scheduling input update requires the queue lock");
        }
        schedulingInputVersion++;
    }

    public boolean isStopped() {
        return stopped;
    }

    public void markStopped() {
        stopped = true;
    }

    /**
     * 等待队列状态变化。调用方持有队列锁。
     */
    public void awaitChange() throws InterruptedException {
        changed.await();
    }

    /**
     * 最多等待 {@code remainingMs} 毫秒，至少等 1 毫秒。调用方持有队列锁。
     */
    public void awaitChangeFor(long remainingMs) throws InterruptedException {
        changed.awaitNanos(TimeUnit.MILLISECONDS.toNanos(Math.max(1L, remainingMs)));
    }

    /**
     * 唤醒调度线程。调用方持有队列锁。
     */
    public void signalChange() {
        changed.signal();
    }

    /**
     * 唤醒所有等待者，停止时使用。调用方持有队列锁。
     */
    public void signalAllChanges() {
        changed.signalAll();
    }
}
