package org.flexlb.cache.service;

import java.util.Collections;
import java.util.Map;

public record CacheMatchResult(
        Map<String, Integer> matches,
        CacheMatchSource source,
        long queryTimeUs) {

    public static CacheMatchResult empty(CacheMatchSource source) {
        return new CacheMatchResult(Collections.emptyMap(), source, 0);
    }
}
