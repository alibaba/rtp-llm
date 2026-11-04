package org.flexlb.mock.engine.rtp;

import java.util.List;
import java.util.Objects;
import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.Scheduler;

/** Ordinary FIFO admission; Forward batches are formed from individual requests. */
public final class RtpScheduler extends RtpSchedulerBase {

    public static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache, finished) -> config.groupScheduling()
            ? new RtpGroupScheduler(lock, config, cache, finished) : new RtpScheduler(lock, config, cache, finished);
    }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache) { this(lock, config, cache, new org.flexlb.mock.engine.TaskHistory()); }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache, org.flexlb.mock.engine.TaskHistory finished) { super(lock, config, cache, finished); }

    @Override protected void selectPrefill(List<Entry> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
