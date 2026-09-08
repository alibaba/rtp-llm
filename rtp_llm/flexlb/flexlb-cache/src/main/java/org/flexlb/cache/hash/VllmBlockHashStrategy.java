package org.flexlb.cache.hash;

import org.flexlb.dao.loadbalance.TokenIds;
import org.flexlb.util.BlockCacheKeyCalculator;

import java.util.List;

public final class VllmBlockHashStrategy implements BlockHashStrategy {

    @Override
    public List<Long> calculate(TokenIds inputIds, long blockSize, int lookaheadTokens) {
        return BlockCacheKeyCalculator.calculate(inputIds, blockSize, lookaheadTokens);
    }
}
