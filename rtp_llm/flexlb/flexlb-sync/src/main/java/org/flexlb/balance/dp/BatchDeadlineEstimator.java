package org.flexlb.balance.dp;

import org.flexlb.dao.master.TaskInfo;

class BatchDeadlineEstimator {

    static long computeDeadlineMicros(long nowMicros, long seqLen, long cacheMatchedTokens,
                                      long avgQueueTimeMs, long ttftSloMs, long safeMarginMs,
                                      long minIntervalMs, long maxIntervalMs) {
        long computeLen = Math.max(0, seqLen - cacheMatchedTokens);
        long tPrefillMs = TaskInfo.estimatePrefillTimeMs(computeLen, 0);
        long ttftEstimateMs = tPrefillMs + avgQueueTimeMs;
        long slackMs = ttftSloMs - ttftEstimateMs - safeMarginMs;
        long intervalMs = Math.max(minIntervalMs, Math.min(maxIntervalMs, slackMs));
        return nowMicros + intervalMs * 1000L;
    }

    private BatchDeadlineEstimator() {}
}
