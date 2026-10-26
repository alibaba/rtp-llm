package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({@JsonSubTypes.Type(value = LocalSyncCacheMatchingConfig.class, name = "LOCAL_SYNC"),
        @JsonSubTypes.Type(value = KvcmCacheMatchingConfig.class, name = "KVCM"),
        @JsonSubTypes.Type(value = LocalStandbyCacheMatchingConfig.class, name = "LOCAL_STANDBY")})
public sealed interface CacheMatchingConfig permits LocalSyncCacheMatchingConfig, KvcmCacheMatchingConfig {

    @JsonIgnore
    CacheMatchMode getType();
}
