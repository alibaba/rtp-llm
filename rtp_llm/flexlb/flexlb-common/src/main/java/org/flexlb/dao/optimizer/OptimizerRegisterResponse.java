package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerRegisterResponse {

    private CommonResponseHeader header;

    @JsonProperty("estimated_capacity_blocks")
    private List<Long> estimatedCapacityBlocks;

    @JsonProperty("size_full_only")
    private long sizeFullOnly;

    @JsonProperty("size_full_linear")
    private long sizeFullLinear;
}
