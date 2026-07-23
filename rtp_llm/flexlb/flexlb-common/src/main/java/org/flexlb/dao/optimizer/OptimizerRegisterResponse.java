package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerRegisterResponse {

    private CommonResponseHeader header;

    @JsonProperty("capacity_blocks")
    private List<Long> capacityBlocks;

    @JsonProperty("avg_bytes_per_block")
    private long avgBytesPerBlock;
}
