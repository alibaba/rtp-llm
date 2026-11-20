package org.flexlb.cache.hash;

import org.flexlb.util.BlockCacheKeyCalculator;
import org.flexlb.util.TokenIds;

import java.util.List;

/**
 * Builds vLLM-compatible block hashes from tokens and lookahead inputs.
 */
public final class VllmBlockHashStrategy implements BlockHashStrategy {

    @Override
    public List<Long> calculate(TokenIds inputIds, long blockSize, int lookaheadTokens) {
        return BlockCacheKeyCalculator.calculate(inputIds, blockSize, lookaheadTokens);
    }
}
