package org.flexlb.cache.domain;

/**
 * Worker-status-owned block shape parameters used to calculate request cache keys.
 *
 * <p>{@code blockSize} is read from {@code CacheStatus.blockSize} and {@code lookaheadTokens}
 * from {@code WorkerStatus.blockHashLookaheadTokens}. Both values are supplied by alive engine
 * workers through their status; they are not configured in {@code FLEXLB_CONFIG}.
 *
 * <p>This model is intentionally separate from {@link org.flexlb.config.BlockHashConfig}, which
 * configures the algorithm type and vLLM hash seed. It is refreshed from worker status by
 * {@code WorkerBlockHashConfigResolver}.
 *
 * @param blockSize tokens in each engine cache block, reported by worker status
 * @param lookaheadTokens tokens used to extend the engine's block hash calculation, reported by
 *                        worker status
 */
public record WorkerBlockHashConfig(long blockSize, int lookaheadTokens) {

    public WorkerBlockHashConfig {
        if (blockSize <= 0) {
            throw new IllegalArgumentException("block_size must be greater than 0");
        }
        if (lookaheadTokens < 0) {
            throw new IllegalArgumentException("block_hash_lookahead_tokens must not be negative");
        }
    }
}
