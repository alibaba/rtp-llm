package org.flexlb.sync.runner;

import java.util.concurrent.atomic.AtomicLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SyncSummaryLogger {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final long SUMMARY_INTERVAL = 50;

    private final AtomicLong syncRoundCounter = new AtomicLong(0);

    public void recordSyncRound(int total, int alive, int failed) {
        long round = syncRoundCounter.incrementAndGet();
        if (round % SUMMARY_INTERVAL == 0) {
            logger.info("Sync summary [round={}]: {} workers total, {} alive, {} failed in last {} rounds",
                round, total, alive, failed, SUMMARY_INTERVAL);
        }
    }
}
