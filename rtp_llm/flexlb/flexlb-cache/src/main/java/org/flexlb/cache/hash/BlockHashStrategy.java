package org.flexlb.cache.hash;

import java.util.List;

public interface BlockHashStrategy {

    List<Long> calculate(int[] inputIds, long blockSize, int lookaheadTokens);

    default List<Long> cacheablePrefix(
            List<Long> blockCacheKeys, int inputTokenCount, long blockSize, int lookaheadTokens) {
        return blockCacheKeys;
    }
}
