package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class DebugInfo {
    @JsonProperty("running_batch_size")
    private long runningBatchSize;

    @JsonProperty("queue_size")
    private long queueSize;

    @JsonProperty("waiting_time_ms")
    private long waitingTimeMs;

    @JsonProperty("available_kv_cache_len")
    private long availableKvCacheLen;

    @JsonProperty("estimate_ttft_ms")
    private long estimateTtftMs;

    @JsonProperty("estimate_tpot_ms")
    private long estimateTpotMs;

    @JsonProperty("hit_cache_len")
    private long hitCacheLen;
}
