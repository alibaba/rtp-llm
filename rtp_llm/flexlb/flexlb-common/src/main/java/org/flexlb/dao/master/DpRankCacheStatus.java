package org.flexlb.dao.master;

import java.util.Set;

/**
 * Per-DP-rank slice of {@link CacheStatus}, carried in the {@code dp_cache[]}
 * field of {@code CacheStatusPB} and populated by DP0 when {@code dp_size > 1}.
 *
 * <p>The outer {@link CacheStatus#getCachedKeys()} is the union across all DP
 * ranks (preserves legacy any-rank match semantics). This per-rank view is
 * tracked as a secondary index for monitoring and V2 rank-aware routing —
 * V1 routing does NOT consume it.
 */
public record DpRankCacheStatus(int dpRank,
                                String ip,
                                int grpcPort,
                                long availableKvCache,
                                long totalKvCache,
                                long blockSize,
                                Set<Long> cachedKeys) {
}
