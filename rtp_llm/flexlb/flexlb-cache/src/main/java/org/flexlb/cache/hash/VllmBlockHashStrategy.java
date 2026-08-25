package org.flexlb.cache.hash;

import org.flexlb.util.BlockCacheKeyCalculator;

import java.util.List;

public final class VllmBlockHashStrategy implements BlockHashStrategy {

    @Override
    public List<Long> calculate(int[] inputIds, long blockSize, int lookaheadTokens) {
        return BlockCacheKeyCalculator.calculate(inputIds, blockSize, lookaheadTokens);
    }
}
