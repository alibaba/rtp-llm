package org.flexlb.cache.domain;

import java.util.Collections;
import java.util.Map;

/**
 * Cache matches and the block size used to produce them.
 *
 * <p>{@code blockSize} is the only valid unit for converting matched block counts to token counts.
 */
public record CacheMatchResult(
        Map<String, Integer> matches,
        CacheMatchSource source,
        long queryTimeUs,
        long blockSize) {

    public static CacheMatchResult empty(CacheMatchSource source) {
        return new CacheMatchResult(Collections.emptyMap(), source, 0, 0);
    }

    public static CacheMatchResult failed(CacheMatchSource source, long queryTimeUs) {
        return new CacheMatchResult(Collections.emptyMap(), source, queryTimeUs, 0);
    }
}
