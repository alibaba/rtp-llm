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
    private boolean success;
    private String engineIpPort;
    private long cacheBlockCount;
    private long availableKvCache;
    private long totalKvCache;
    private long cacheVersion;
    private String errorMessage;
}