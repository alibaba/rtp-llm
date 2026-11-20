package org.flexlb.worker.cache;

/**
 * KV-cache attention layout reported by an engine worker.
 */
public enum KvCacheGroupMode {
    UNSPECIFIED,
    FULL_ATTENTION_ONLY,
    WITH_MAMBA
}
