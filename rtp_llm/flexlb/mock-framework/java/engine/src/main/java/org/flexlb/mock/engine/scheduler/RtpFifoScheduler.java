package org.flexlb.mock.engine.scheduler;

import java.util.List;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;

/** Ordinary FIFO admission; Forward batches are formed from individual requests. */
public final class RtpFifoScheduler extends AbstractRtpScheduler {

    RtpFifoScheduler(Object lock, RtpConfig config, KvCacheManager cache) { this(lock, config, cache, new TaskHistory()); }

    RtpFifoScheduler(Object lock, RtpConfig config, KvCacheManager cache, TaskHistory finished) { super(lock, config, cache, finished); }

    @Override protected void selectPrefill(List<GenerateStream> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
