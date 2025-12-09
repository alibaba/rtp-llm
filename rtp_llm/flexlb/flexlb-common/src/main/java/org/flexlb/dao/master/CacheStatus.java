package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Set;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CacheStatus {

    @JsonProperty("message")
    private String message;
    @JsonProperty("available_kv_cache")
    private long availableKvCache;  // available kv cache tokens
    @JsonProperty("total_kv_cache")
    private long totalKvCache;      // total kv cache tokens
    @JsonProperty("block_size")
    private long blockSize;
    @JsonProperty("version")
    private long version = -1;
    @JsonProperty("cached_keys")
    private Set<Long> cachedKeys;
    @JsonProperty("cache_key_size")
    private long cacheKeySize;
}