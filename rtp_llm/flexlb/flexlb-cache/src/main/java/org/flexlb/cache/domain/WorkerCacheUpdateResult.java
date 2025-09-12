package org.flexlb.cache.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Worker缓存更新结果
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
    private long timestamp;
    private String errorMessage;
    
    @Override
    public String toString() {
        return String.format("WorkerCacheUpdate[ip=%s, success=%s, blocks=%d, available=%d/%d, version=%d]",
            engineIpPort, success, cacheBlockCount, availableKvCache, totalKvCache, cacheVersion);
    }
}