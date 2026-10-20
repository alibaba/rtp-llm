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

    /** Return an independent copy, or null when the source is null. */
    public static DebugInfo copyOf(DebugInfo source) {
        if (source == null) {
            return null;
        }
        DebugInfo copy = new DebugInfo();
        copy.runningBatchSize = source.runningBatchSize;
        copy.queueSize = source.queueSize;
        copy.waitingTimeMs = source.waitingTimeMs;
        copy.availableKvCacheLen = source.availableKvCacheLen;
        copy.estimateTtftMs = source.estimateTtftMs;
        copy.estimateTpotMs = source.estimateTpotMs;
        copy.hitCacheLen = source.hitCacheLen;
        return copy;
    }
}
