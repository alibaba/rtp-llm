package org.flexlb.cache.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Worker cache update result
 *
 * @author FlexLB
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkerCacheUpdateResult {
    public enum Outcome {
        APPLIED,
        STALE_GENERATION,
        INVALID_INPUT,
        FAILED
    }

    private Outcome outcome;
    private String engineIpPort;
    private long cacheBlockCount;
    private long availableKvCache;
    private long totalKvCache;
    private long cacheVersion;
    private String errorMessage;

    public boolean isSuccess() {
        return outcome == Outcome.APPLIED;
    }

    public boolean isStaleGeneration() {
        return outcome == Outcome.STALE_GENERATION;
    }
}
