package org.flexlb.mock.engine.scheduler;

import java.util.List;
import java.util.Objects;
import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.config.RtpConfig;

/** Ordinary FIFO admission; Forward batches are formed from individual requests. */
public final class RtpScheduler extends RtpSchedulerBase {

    public static java.util.function.Supplier<List<io.prometheus.client.Collector.MetricFamilySamples>> metricsFor(Scheduler scheduler) {
        return scheduler instanceof RtpSchedulerBase rtp ? rtp.metrics()::collect : null;
    }

    public static Scheduler.Factory factory(RtpConfig config) {
        Objects.requireNonNull(config);
        return (lock, cache, finished) -> config.groupScheduling()
            ? new RtpGroupScheduler(lock, config, cache, finished) : new RtpScheduler(lock, config, cache, finished);
    }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache) { this(lock, config, cache, new org.flexlb.mock.engine.scheduler.TaskHistory()); }

    RtpScheduler(Object lock, RtpConfig config, KvCache cache, org.flexlb.mock.engine.scheduler.TaskHistory finished) { super(lock, config, cache, finished); }

    @Override protected void selectPrefill(List<Entry> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
