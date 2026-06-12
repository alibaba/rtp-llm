package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class OptimizerGetInstanceRequest {

    @JsonProperty("trace_id")
    private String traceId;

    @JsonProperty("instance_id")
    private String instanceId;
}
