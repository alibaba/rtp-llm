package org.flexlb.mock.engine.scheduler;

import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;

/**
 * Assembles an RTP scheduler and its metrics for tests that run without a Mock Engine.
 */
public final class RtpTestMetrics implements EngineScheduler.Factory {
    private final RtpConfig config;
    private AbstractRtpScheduler scheduler;
    private RtpMetrics metrics;

    public RtpTestMetrics(RtpConfig config) {
        this.config = config;
    }

    @Override public AbstractRtpScheduler create(Object lock, KvCacheManager cache, TaskHistory finished) {
        metrics = createMetrics(lock, cache);
        scheduler = AbstractRtpScheduler.create(lock, config, cache, finished, metrics);
        return scheduler;
    }

    public String scrape() { return metrics.scrape(); }

    static RtpFifoScheduler fifo(Object lock, RtpConfig config, KvCacheManager cache) {
        return fifo(lock, config, cache, new TaskHistory());
    }

    static RtpFifoScheduler fifo(Object lock, RtpConfig config, KvCacheManager cache, TaskHistory finished) {
        var fixture = new RtpTestMetrics(config);
        var scheduler = new RtpFifoScheduler(lock, config, cache, finished, fixture.createMetrics(lock, cache));
        fixture.scheduler = scheduler;
        return scheduler;
    }

    static RtpPrefillGroupScheduler grouped(Object lock, RtpConfig config, KvCacheManager cache) {
        var fixture = new RtpTestMetrics(config);
        var scheduler = new RtpPrefillGroupScheduler(lock, config, cache, new TaskHistory(), fixture.createMetrics(lock, cache));
        fixture.scheduler = scheduler;
        return scheduler;
    }

    private RtpMetrics createMetrics(Object lock, KvCacheManager cache) {
        return new RtpMetrics(config, lock, () -> scheduler.queues(), cache);
    }
}
