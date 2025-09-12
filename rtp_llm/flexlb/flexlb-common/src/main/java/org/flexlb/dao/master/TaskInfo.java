package org.flexlb.dao.master;


import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;


@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TaskInfo {
    @JsonProperty("inter_request_id")
    private long interRequestId;
    @JsonProperty("prefix_length")
    private long prefixLength;    // cache hit len
    @JsonProperty("prefill_time")
    private long prefillTime;
    @JsonProperty("input_length")
    private long inputLength;
    @JsonProperty("waiting_time")
    private long waitingTime;
    @JsonProperty("iterate_count")
    private long iterateCount;
    @JsonProperty("end_time_ms")
    private long endTimeMs;
    @JsonProperty("dp_rank")
    private long dpRank;
    @JsonProperty("enqueue_time_ms")
    private long enqueueTimeMs;

    public long estimatePrefillTime() {
        return estimatePrefillTimeMs(inputLength, prefixLength);
    }

    static public long estimatePrefillTimeMs(long tokens, long hitCacheTokens) {
        return (long) (tokens * 1.0 - hitCacheTokens * 0.7);
    }
}
