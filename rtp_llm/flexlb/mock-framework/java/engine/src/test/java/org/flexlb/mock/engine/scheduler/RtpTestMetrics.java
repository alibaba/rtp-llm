package org.flexlb.mock.engine.scheduler;

import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;

/** Captures the concrete RTP metrics for tests that assemble an admission component directly. */
public final class RtpTestMetrics implements EngineScheduler.Factory {
    private final EngineScheduler.Factory factory;
    private Object lock;
    private RtpMetrics metrics;

    public RtpTestMetrics(RtpConfig config) {
        factory = AbstractRtpScheduler.factory(config);
    }

    @Override public EngineScheduler create(Object lock, KvCacheManager cache, TaskHistory finished) {
        var scheduler = (AbstractRtpScheduler) factory.create(lock, cache, finished);
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
