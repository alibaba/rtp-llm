package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class SelectWorkersRequest {

    @JsonProperty("role")
    private String role;

    @JsonProperty("count")
    private int count;

    @JsonProperty("request_id")
    private long requestId;

    @JsonProperty("request_time_ms")
    private long requestTimeMs;
}
