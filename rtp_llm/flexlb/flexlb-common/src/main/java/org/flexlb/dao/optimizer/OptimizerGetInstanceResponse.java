package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerGetInstanceResponse {

    private CommonResponseHeader header;

    @JsonProperty("instance_group")
    private String instanceGroup;

    @JsonProperty("instance_id")
    private String instanceId;

    @JsonProperty("block_size")
    private int blockSize;

    @JsonProperty("location_spec_infos")
    private List<LocationSpecInfo> locationSpecInfos;

    @JsonProperty("location_spec_groups")
    private List<LocationSpecGroup> locationSpecGroups;

    @JsonProperty("linear_step")
    private int linearStep;

    @JsonProperty("full_group_name")
    private String fullGroupName;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LocationSpecInfo {
        private String name;
        private int size;
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LocationSpecGroup {
        private String name;

        @JsonProperty("spec_names")
        private List<String> specNames;
    }
}
