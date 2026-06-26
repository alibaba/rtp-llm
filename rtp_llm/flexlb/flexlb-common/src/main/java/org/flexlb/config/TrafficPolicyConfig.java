package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class TrafficPolicyConfig {

    @JsonProperty("enabled")
    private boolean enabled = true;

    @JsonProperty("default_group")
    private String defaultGroup;

    @JsonProperty("default_target_groups")
    private List<TrafficTargetGroup> defaultTargetGroups = new ArrayList<>();

    @JsonProperty("rules")
    private List<TrafficPolicyRule> rules = new ArrayList<>();

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TrafficPolicyRule {

        @JsonProperty("name")
        private String name;

        @JsonProperty("target_group")
        private String targetGroup;

        @JsonProperty("target_groups")
        private List<TrafficTargetGroup> targetGroups = new ArrayList<>();

        @JsonProperty("api_keys")
        private List<String> apiKeys = new ArrayList<>();

        @JsonProperty("min_seq_len")
        private Long minSeqLen;

        @JsonProperty("max_seq_len")
        private Long maxSeqLen;
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TrafficTargetGroup {

        @JsonProperty("group")
        private String group;

        @JsonProperty("weight")
        private long weight = 1;
    }
}
