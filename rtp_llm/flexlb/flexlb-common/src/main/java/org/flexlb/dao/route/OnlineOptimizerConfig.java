package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.optimizer.OptimizerInstanceParams;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;

import java.util.ArrayList;
import java.util.List;

/**
 * Online Optimizer client and instance registration configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class OnlineOptimizerConfig {

    public static final String DEFAULT_PATH = "/api/optimizer";
    public static final int DEFAULT_REGISTER_TIMEOUT_MS = 5000;

    @JsonProperty("enabled")
    private boolean enabled;

    @JsonProperty("address")
    private String address;

    @JsonProperty("path")
    private String path = DEFAULT_PATH;

    @JsonProperty("discovery")
    private DiscoveryConfig discovery;

    @JsonProperty("instance_group")
    private String instanceGroup;

    @JsonProperty("instance_id")
    private String instanceId;

    @JsonProperty("register_timeout_ms")
    private int registerTimeoutMs = DEFAULT_REGISTER_TIMEOUT_MS;

    @JsonProperty("block_size")
    private int blockSize;

    @JsonProperty("linear_step")
    private int linearStep;

    @JsonProperty("location_spec_infos")
    private List<LocationSpecInfo> locationSpecInfos = new ArrayList<>();

    @JsonProperty("location_spec_groups")
    private List<LocationSpecGroup> locationSpecGroups = new ArrayList<>();

    @JsonProperty("optimizer_state_info")
    private OptimizerStateInfo optimizerStateInfo;

    public Endpoint toEndpoint() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("http");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }

    public OptimizerInstanceParams toInstanceParams() {
        List<OptimizerRegisterRequest.LocationSpecInfo> registerSpecInfos = locationSpecInfos.stream()
                .map(spec -> new OptimizerRegisterRequest.LocationSpecInfo(spec.getName(), spec.getSize()))
                .toList();
        List<OptimizerRegisterRequest.LocationSpecGroup> registerSpecGroups = locationSpecGroups.stream()
                .map(group -> new OptimizerRegisterRequest.LocationSpecGroup(
                        group.getName(), List.copyOf(group.getSpecNames())))
                .toList();
        org.flexlb.dao.optimizer.OptimizerStateInfo registerStateInfo =
                new org.flexlb.dao.optimizer.OptimizerStateInfo(
                        optimizerStateInfo.getFullLocationSpecGroupName(),
                        optimizerStateInfo.getLinearLocationSpecGroupName());
        return OptimizerInstanceParams.builder()
                .instanceGroup(instanceGroup)
                .blockSize(blockSize)
                .locationSpecInfos(registerSpecInfos)
                .locationSpecGroups(registerSpecGroups)
                .linearStep(linearStep)
                .optimizerStateInfo(registerStateInfo)
                .build();
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LocationSpecInfo {

        @JsonProperty("name")
        private String name;

        @JsonProperty("size")
        private long size;
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class LocationSpecGroup {

        @JsonProperty("name")
        private String name;

        @JsonProperty("spec_names")
        private List<String> specNames = new ArrayList<>();
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OptimizerStateInfo {

        @JsonProperty("full_location_spec_group_name")
        private String fullLocationSpecGroupName;

        @JsonProperty("linear_location_spec_group_name")
        private String linearLocationSpecGroupName;
    }
}
