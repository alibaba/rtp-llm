package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.loadbalance.Request;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.zip.CRC32;

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

    public Optional<String> resolveTargetGroup(Request request) {
        if (!enabled || request == null) {
            return Optional.empty();
        }

        if (rules != null) {
            for (TrafficPolicyRule rule : rules) {
                if (rule == null || !rule.matches(request)) {
                    continue;
                }
                Optional<String> group = rule.resolveTargetGroup(request);
                if (group.isPresent()) {
                    return group;
                }
            }
        }

        return resolveGroup(defaultGroup, defaultTargetGroups, request, "default");
    }

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

        boolean matches(Request request) {
            if (!hasMatcher()) {
                return false;
            }

            if (apiKeys != null && !apiKeys.isEmpty() && !apiKeys.contains(request.getApiKey())) {
                return false;
            }

            long seqLen = request.getSeqLen();
            if (minSeqLen != null && seqLen < minSeqLen) {
                return false;
            }
            if (maxSeqLen != null && seqLen > maxSeqLen) {
                return false;
            }

            return true;
        }

        private Optional<String> resolveTargetGroup(Request request) {
            return resolveGroup(targetGroup, targetGroups, request, name);
        }

        private boolean hasMatcher() {
            return (apiKeys != null && !apiKeys.isEmpty()) || minSeqLen != null || maxSeqLen != null;
        }
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TrafficTargetGroup {

        @JsonProperty("group")
        private String group;

        @JsonProperty("weight")
        private long weight = 1;
    }

    private static Optional<String> resolveGroup(String targetGroup,
                                                 List<TrafficTargetGroup> targetGroups,
                                                 Request request,
                                                 String salt) {
        if (StringUtils.isNotBlank(targetGroup)) {
            return Optional.of(targetGroup);
        }
        return chooseWeightedGroup(targetGroups, request, salt);
    }

    private static Optional<String> chooseWeightedGroup(List<TrafficTargetGroup> targetGroups,
                                                        Request request,
                                                        String salt) {
        if (targetGroups == null || targetGroups.isEmpty()) {
            return Optional.empty();
        }

        long totalWeight = targetGroups.stream()
                .filter(targetGroup -> targetGroup != null && StringUtils.isNotBlank(targetGroup.getGroup()))
                .mapToLong(targetGroup -> Math.max(0, targetGroup.getWeight()))
                .sum();
        if (totalWeight <= 0) {
            return Optional.empty();
        }

        long bucket = hashRequest(request, salt) % totalWeight;
        long cumulativeWeight = 0;
        for (TrafficTargetGroup targetGroup : targetGroups) {
            if (targetGroup == null || StringUtils.isBlank(targetGroup.getGroup()) || targetGroup.getWeight() <= 0) {
                continue;
            }
            cumulativeWeight += targetGroup.getWeight();
            if (bucket < cumulativeWeight) {
                return Optional.of(targetGroup.getGroup());
            }
        }

        return Optional.empty();
    }

    private static long hashRequest(Request request, String salt) {
        CRC32 crc32 = new CRC32();
        String key = request.getRequestId() + "|" + request.getApiKey() + "|" + request.getSeqLen() + "|" + salt;
        crc32.update(key.getBytes(StandardCharsets.UTF_8));
        return crc32.getValue();
    }
}
