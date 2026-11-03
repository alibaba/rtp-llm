package org.flexlb.mock.engine.rtp;

import java.util.List;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.Objects;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.engine.Scheduler;

/** Ordinary FIFO admission; Forward batches are formed from individual requests. */
public final class RtpScheduler extends RtpSchedulerBase {
    private final Set<Entry> loading = new LinkedHashSet<>();

    public static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache) -> config.groupScheduling()
            ? new RtpGroupScheduler(lock, config, cache) : new RtpScheduler(lock, config, cache);
    }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache) { super(lock, config, cache); }

    @Override protected boolean remoteLookup() { return prefill(); }

    @Override protected void cacheLoading(Entry entry) {
        waiting.remove(entry);
        entry.phase = Phase.CACHE_LOADING;
        loading.add(entry);
    }

    @Override public void maintain() {
        super.maintain();
        for (var entry : List.copyOf(loading)) {
            if (!live(entry)) continue;
            if (cacheReady(entry)) {
                loading.remove(entry);
                entry.phase = Phase.QUEUED;
                waiting.addLast(entry);
            }
        }
    }

    @Override protected void removeWaiting(Entry entry) {
        super.removeWaiting(entry);
        loading.remove(entry);
    }

    @Override protected void selectPrefill(List<Entry> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
