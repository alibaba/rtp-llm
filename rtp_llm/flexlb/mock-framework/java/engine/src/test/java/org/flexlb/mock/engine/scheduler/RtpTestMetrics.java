package org.flexlb.mock.engine.scheduler;

import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.config.RtpConfig;

/** Captures the concrete RTP metrics for tests that assemble an admission component directly. */
public final class RtpTestMetrics implements Scheduler.Factory {
    private final Scheduler.Factory factory;
    private Object lock;
    private RtpMetrics metrics;

    public RtpTestMetrics(RtpConfig config) {
        factory = RtpScheduler.factory(config);
    }

    @Override public Scheduler create(Object lock, KvCache cache, TaskHistory finished) {
        var scheduler = (RtpSchedulerBase) factory.create(lock, cache, finished);
        this.lock = lock;
        metrics = scheduler.metrics();
        return scheduler;
    }

    public String scrape() {
        synchronized (lock) {
            return metrics.scrape();
        }
    }
}
