package org.flexlb.cache.domain;

/**
 * Block hash configuration resolved from the active engine workers.
 */
public record BlockHashConfig(long blockSize, int lookaheadTokens) {

    public BlockHashConfig {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("block_size must be greater than 0");
        }
        if (lookaheadTokens < 0) {
            throw new IllegalArgumentException("block_hash_lookahead_tokens must not be negative");
        }
    }
}
