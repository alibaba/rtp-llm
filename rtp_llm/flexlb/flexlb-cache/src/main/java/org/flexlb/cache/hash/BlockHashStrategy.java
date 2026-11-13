package org.flexlb.cache.hash;

import org.flexlb.util.TokenIds;

import java.util.List;

/**
 * Computes cache keys for complete token blocks using the selected engine's hashing contract.
 */
public interface BlockHashStrategy {

    List<Long> calculate(TokenIds inputIds, long blockSize, int lookaheadTokens);
}
