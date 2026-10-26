package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Uses the local standby index while keeping KVCM available for runtime switching.
 */
public final class LocalStandbyCacheMatchingConfig extends KvcmCacheMatchingConfig {

    @Override
    @JsonIgnore
    public CacheMatchMode getType() {
        return CacheMatchMode.LOCAL_STANDBY;
    }
}
