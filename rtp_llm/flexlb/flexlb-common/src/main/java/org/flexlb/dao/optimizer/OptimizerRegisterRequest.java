package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
public class OptimizerRegisterRequest {

    @JsonProperty("trace_id")
    private String traceId;

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
    public static class LocationSpecInfo {
        private String name;
        private int size;

        public LocationSpecInfo() {}

        public LocationSpecInfo(String name, int size) {
            this.name = name;
            this.size = size;
        }
    }

    @Data
    public static class LocationSpecGroup {
        private String name;

        @JsonProperty("spec_names")
        private List<String> specNames;

        public LocationSpecGroup() {}

        public LocationSpecGroup(String name, List<String> specNames) {
            this.name = name;
            this.specNames = specNames;
        }
    }
}
