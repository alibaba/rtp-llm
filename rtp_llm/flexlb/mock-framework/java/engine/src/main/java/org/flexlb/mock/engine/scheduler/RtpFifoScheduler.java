package org.flexlb.mock.engine.scheduler;

import java.util.List;
import org.flexlb.mock.engine.cache.KvCache;
import org.flexlb.mock.engine.config.RtpConfig;

/** Ordinary FIFO admission; Forward batches are formed from individual requests. */
public final class RtpFifoScheduler extends AbstractRtpScheduler {

    RtpFifoScheduler(Object lock, RtpConfig config, KvCache cache) { this(lock, config, cache, new TaskHistory()); }

    RtpFifoScheduler(Object lock, RtpConfig config, KvCache cache, TaskHistory finished) { super(lock, config, cache, finished); }

    @Override protected void selectPrefill(List<Entry> selected) {
        admitPrefill(waiting, selected, config.maxUncachedBatchTokens());
    }
}
