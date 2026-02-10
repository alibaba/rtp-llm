package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
public class Request {
    @ToString.Exclude
    @JsonProperty("block_cache_keys")
    private List<Long> blockCacheKeys;

    @JsonProperty("seq_len")
    private long seqLen;

    @JsonProperty("request_id")
    private String requestId;

    @JsonProperty("generate_timeout")
    private long generateTimeout = 3600 * 1000;

    @JsonProperty("request_time_ms")
    private long requestTimeMs;
}
