package org.flexlb.mock.engine.scheduler;

import java.util.List;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;

/**
 * Ordinary FIFO admission; Forward batches are formed from individual requests.
 */
public final class RtpFifoScheduler extends AbstractRtpScheduler {

    RtpFifoScheduler(Object lock, RtpConfig config, KvCacheManager cache, TaskHistory finished, RtpMetrics metrics) {
        super(lock, config, cache, finished, metrics);
    }

    @Override protected void selectPrefill(List<GenerateStream> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
