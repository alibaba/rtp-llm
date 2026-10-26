package org.flexlb.config;

public final class LocalSyncCacheMatchingConfig implements CacheMatchingConfig {

    @Override
    public CacheMatchMode getType() {
        return CacheMatchMode.LOCAL_SYNC;
    }
}
