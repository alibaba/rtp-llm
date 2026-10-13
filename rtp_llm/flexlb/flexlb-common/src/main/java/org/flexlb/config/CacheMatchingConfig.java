package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = LocalSyncCacheMatchingConfig.class, name = "LOCAL_SYNC"),
        @JsonSubTypes.Type(value = KvcmCacheMatchingConfig.class, name = "KVCM")
})
public sealed interface CacheMatchingConfig
        permits LocalSyncCacheMatchingConfig, KvcmCacheMatchingConfig {
}
