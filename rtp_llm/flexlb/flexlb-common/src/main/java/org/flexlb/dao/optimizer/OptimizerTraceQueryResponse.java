package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerTraceQueryResponse {

    private CommonResponseHeader header;

    @JsonProperty("total_blocks")
    private long totalBlocks;

    @JsonProperty("capacity_results")
    private List<CapacityResult> capacityResults;

    @JsonProperty("theoretical_result")
    private TheoreticalResult theoreticalResult;

    @JsonProperty("input_token_len")
    private long inputTokenLen;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class CapacityResult {

        @JsonProperty("capacity_gb")
        private double capacityGb;

        @JsonProperty("cache_hit_count")
        private long cacheHitCount;

        @JsonProperty("hit_rate")
        private double hitRate;

        @JsonProperty("current_unique_keys")
        private long currentUniqueKeys;
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TheoreticalResult {

        @JsonProperty("max_hit_count")
        private long maxHitCount;

        @JsonProperty("current_unique_keys")
        private long currentUniqueKeys;

        @JsonProperty("hit_rate")
        private double hitRate;
    }
}
