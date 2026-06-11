package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class OptimizerTraceQueryRequest {

    @JsonProperty("trace_id")
    private String traceId;

    @JsonProperty("instance_id")
    private String instanceId;

    @JsonProperty("block_keys")
    private List<Long> blockKeys;
}
