package org.flexlb.diagnostics;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleConsumer;

/** Constant-size task timing: no queue walks or per-task tags. */
public final class ObservedExecutor extends ThreadPoolExecutor {
    private static final Logger LOG = LoggerFactory.getLogger("syncLogger");
    private static final DoubleConsumer NOOP = value -> { };
    private static final java.util.Set<ObservedExecutor> LIVE = ConcurrentHashMap.newKeySet();
    public static void forEachLive(java.util.function.Consumer<ObservedExecutor> consumer) { LIVE.forEach(consumer); }
    public String metricName() { return name; }
    private final String name;
    private final ConcurrentHashMap<Thread, ActiveSlot> running = new ConcurrentHashMap<>();
    private final AtomicLong rejections = new AtomicLong(), callerRuns = new AtomicLong(), lastLogNs = new AtomicLong();
    private volatile DoubleConsumer queueMs = NOOP, runMs = NOOP;
    private volatile boolean bound;
    private static final class ActiveSlot { volatile long startNs; }

    public ObservedExecutor(int core, int max, long keepAlive, TimeUnit unit,
                            BlockingQueue<Runnable> queue, ThreadFactory factory,
                            RejectedExecutionHandler policy, String name) {
        super(core, max, keepAlive, unit, queue, factory);
        this.name = name;
        LIVE.add(this);
        setRejectedExecutionHandler((task, pool) -> {
            rejections.incrementAndGet();
            boolean fallsBack = policy instanceof CallerRunsPolicy && !pool.isShutdown();
            if (fallsBack) { callerRuns.incrementAndGet(); }
            long now = System.nanoTime(), last = lastLogNs.get();
            if (now - last >= TimeUnit.SECONDS.toNanos(10) && lastLogNs.compareAndSet(last, now)) {
                LOG.warn("event=executor_rejected pool={} reason={} fallback={} queue_size={} queue_capacity={} active={} max_threads={} oldest_pending_ms={} oldest_running_ms={} rejections={} heap_used_bytes={}",
                        name, pool.isShutdown() ? "shutdown" : "capacity", fallsBack,
                        pool.getQueue().size(), pool.getQueue().size() + pool.getQueue().remainingCapacity(),
                        pool.getActiveCount(), pool.getMaximumPoolSize(), oldestPendingMs(), oldestRunningMs(),
                        rejections.get(), Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
            }
            policy.rejectedExecution(task, pool);
        });
    }

    @Override public void execute(Runnable command) { super.execute(new TimedTask(command)); }

    @Override public java.util.List<Runnable> shutdownNow() {
        java.util.List<Runnable> pending=super.shutdownNow();
        java.util.List<Runnable> unwrapped=new java.util.ArrayList<>(pending.size());
        for(Runnable r:pending)unwrapped.add(r instanceof ObservedExecutor.TimedTask t?t.delegate:r);
        return unwrapped;
    }
    @Override public boolean remove(Runnable command) {
        for(Runnable r:getQueue())if(r instanceof ObservedExecutor.TimedTask t && t.delegate==command)return super.remove(r);
        return super.remove(command);
    }
    @Override public void purge(){
        getQueue().removeIf(r->r instanceof ObservedExecutor.TimedTask t && t.delegate instanceof Future<?> f && f.isCancelled());
        super.purge();
    }

    private final class TimedTask implements Runnable {
        final Runnable delegate;
        final long enqueuedNs = System.nanoTime();
        TimedTask(Runnable delegate) { this.delegate = delegate; }
        @Override public void run() {
            long start = System.nanoTime();
            try { queueMs.accept((start - enqueuedNs) / 1e6); } catch (RuntimeException ignored) { }
            try { delegate.run(); }
            finally { try { runMs.accept((System.nanoTime() - start) / 1e6); } catch (RuntimeException ignored) { } }
        }
    }

    @Override protected void beforeExecute(Thread thread, Runnable task) {
        super.beforeExecute(thread, task);
        ActiveSlot slot = running.get(thread);
        if (slot == null) { slot = new ActiveSlot(); running.put(thread, slot); }
        slot.startNs = System.nanoTime();
    }
    @Override protected void afterExecute(Runnable task, Throwable failure) {
        ActiveSlot slot = running.get(Thread.currentThread());
        if (slot != null) { slot.startNs = 0; }
        super.afterExecute(task, failure);
    }
    @Override protected void terminated() { LIVE.remove(this); running.clear(); super.terminated(); }
    public double oldestPendingMs() {
        Runnable head = getQueue().peek();
        return head instanceof ObservedExecutor.TimedTask timed ? (System.nanoTime() - timed.enqueuedNs) / 1e6 : 0;
    }
    public double oldestRunningMs() {
        running.keySet().removeIf(thread -> !thread.isAlive());
        long now = System.nanoTime(), age = 0;
        for (ActiveSlot slot : running.values()) { long started = slot.startNs; if (started > 0) { age = Math.max(age, now - started); } }
        return age / 1e6;
    }
    public long rejectionCount() { return rejections.get(); }
    public long callerRunCount() { return callerRuns.get(); }
    public void bind(FlexMonitor monitor) {
        if (bound) { return; }
        synchronized (this) {
            if (bound) { return; }
            monitor.register("app.executor.queue.wait.ms", FlexMetricType.TIMER);
            monitor.register("app.executor.task.run.ms", FlexMetricType.TIMER);
            FlexMetricTags tags = FlexMetricTags.of("threadPool", name);
            queueMs = monitor.bind("app.executor.queue.wait.ms", tags);
            runMs = monitor.bind("app.executor.task.run.ms", tags);
            bound = true;
        }
    }
}
