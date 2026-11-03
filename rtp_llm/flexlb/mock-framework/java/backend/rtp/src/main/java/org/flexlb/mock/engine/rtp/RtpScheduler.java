package org.flexlb.mock.engine.rtp;

import java.util.List;
import java.util.Objects;
import org.flexlb.mock.cache.KvCache;
import org.flexlb.mock.engine.TaskHistory;
import org.flexlb.mock.engine.Scheduler;

/**
 * Ordinary FIFO admission; Forward batches are formed from individual requests.
 */
public final class RtpScheduler extends RtpSchedulerBase {
    public static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache, history) -> config.groupScheduling()
            ? new RtpGroupScheduler(lock, config, cache, history) : new RtpScheduler(lock, config, cache, history);
    }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache) { this(lock, config, cache, new TaskHistory()); }
    RtpScheduler(Object lock, RtpConfig config, KvCache cache, TaskHistory history) { super(lock, config, cache, history); }

    @Override protected void selectPrefill(List<Entry> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
