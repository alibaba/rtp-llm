package org.flexlb.cache.domain;

/** Prefix-cache match value keyed by an exact {@link EngineGeneration}. */
public record CacheMatch(int prefixMatchLength) {

    public CacheMatch {
        if (prefixMatchLength < 0) {
            throw new IllegalArgumentException(
                    "prefixMatchLength must be non-negative");
        }
    }
}
