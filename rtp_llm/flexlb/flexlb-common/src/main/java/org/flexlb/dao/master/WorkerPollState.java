package org.flexlb.dao.master;

import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/** Generation-local transport health and a bounded, single-writer status mailbox. */
public final class WorkerPollState {
    private final AtomicBoolean retired = new AtomicBoolean();
    private final AtomicBoolean applying = new AtomicBoolean();
    private final AtomicReference<Application> pending = new AtomicReference<>();
    private final AtomicLong lastRpcSuccessUs = new AtomicLong();
    private final AtomicLong lastAppliedUs = new AtomicLong();
    private volatile long rpcDeadlineUs;
    private volatile boolean managed;
    private volatile long staleAfterUs = 10_000_000L;

    private record Application(long receivedUs, Runnable action) { }

    public void configure(long staleAfterMs) { staleAfterUs = staleAfterMs * 1000L; managed = true; }
    public boolean isManaged() { return managed; }
    public boolean retire() {
        boolean changed = retired.compareAndSet(false, true);
        pending.set(null);
        return changed;
    }
    public boolean isRetired() { return retired.get(); }
    public void rpcStarted(long nowUs, long timeoutMs) { rpcDeadlineUs = nowUs + timeoutMs * 1000L; }
    public void rpcFinished() { rpcDeadlineUs = 0; }
    public void rpcSucceeded(long receivedUs) { lastRpcSuccessUs.accumulateAndGet(receivedUs, Math::max); }
    public void applied(long receivedUs) { lastAppliedUs.accumulateAndGet(receivedUs, Math::max); }
    public long lastRpcSuccessUs() { return lastRpcSuccessUs.get(); }
    public long lastAppliedUs() { return lastAppliedUs.get(); }
    public boolean isSchedulingFresh(long nowUs) {
        long applied = lastAppliedUs.get();
        return !isRetired() && applied > 0 && nowUs - applied <= staleAfterUs;
    }
    public boolean isExpired(long nowUs, long fallbackUs, long timeoutUs) {
        long last = Math.max(lastRpcSuccessUs.get(), fallbackUs);
        return !isRetired() && nowUs - last > timeoutUs && nowUs > rpcDeadlineUs;
    }

    public void offer(long receivedUs, Runnable action, Executor executor) {
        if (isRetired()) return;
        Application next = new Application(receivedUs, action);
        pending.accumulateAndGet(next, (old, value) -> old == null || value.receivedUs >= old.receivedUs ? value : old);
        schedule(executor);
    }

    private void schedule(Executor executor) {
        if (isRetired() || !applying.compareAndSet(false, true)) return;
        try {
            executor.execute(() -> drain(executor));
        } catch (RejectedExecutionException rejected) {
            // Retain one latest observation; the next successful RPC retries dispatch.
            applying.set(false);
        }
    }

    private void drain(Executor executor) {
        try {
            for (int n = 0; n < 16 && !isRetired(); n++) {
                Application application = pending.getAndSet(null);
                if (application == null) break;
                application.action.run();
            }
        } finally {
            applying.set(false);
            if (!isRetired() && pending.get() != null) schedule(executor);
        }
    }
}
