package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.loadbalance.Request;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.zip.CRC32;

/** First-match request-to-worker-group selector. */
@Getter
@Setter
public final class TrafficPolicyConfig {

    private List<Target> defaultTargets = new ArrayList<>();
    private List<Rule> rules = new ArrayList<>();

    public Optional<String> resolveTargetGroup(Request request) {
        if (request == null) {
            return Optional.empty();
        }
        for (Rule rule : rules) {
            if (rule.matches(request)) {
                return chooseWeightedTarget(rule.getTargets(), request, rule.getName());
            }
        }
        return chooseWeightedTarget(defaultTargets, request, "default");
    }

    static void validate(TrafficPolicyConfig config) {
        if (config.defaultTargets == null || config.rules == null) {
            throw new ConfigValidationException("router.groupSelector",
                    "defaultTargets and rules must not be null");
        }
        validateTargets(config.defaultTargets, "router.groupSelector.defaultTargets", true);
        Set<String> names = new HashSet<>();
        for (int index = 0; index < config.rules.size(); index++) {
            Rule rule = config.rules.get(index);
            String path = "router.groupSelector.rules[" + index + "]";
            if (rule == null || rule.name == null || rule.name.isBlank()) {
                throw new ConfigValidationException(path + ".name", "must not be blank");
            }
            if (!names.add(rule.name)) {
                throw new ConfigValidationException(path + ".name", "must be unique");
            }
            if (rule.match == null || !rule.match.hasConstraint()) {
                throw new ConfigValidationException(path + ".match",
                        "must contain at least one constraint");
            }
            rule.match.validate(path + ".match");
            validateTargets(rule.targets, path + ".targets", false);
        }
    }

    private static void validateTargets(List<Target> targets, String path, boolean optional) {
        if (targets == null || (!optional && targets.isEmpty())) {
            throw new ConfigValidationException(path, "must contain at least one target");
        }
        Set<String> groups = new HashSet<>();
        long total = 0;
        for (int index = 0; index < targets.size(); index++) {
            Target target = targets.get(index);
            if (target == null || target.group == null || target.group.isBlank()) {
                throw new ConfigValidationException(path + "[" + index + "].group",
                        "must not be blank");
            }
            if (!groups.add(target.group)) {
                throw new ConfigValidationException(path + "[" + index + "].group",
                        "must be unique");
            }
            if (target.weight <= 0) {
                throw new ConfigValidationException(path + "[" + index + "].weight",
                        "must be greater than zero");
            }
            try {
                total = Math.addExact(total, target.weight);
            } catch (ArithmeticException overflow) {
                throw new ConfigValidationException(path,
                        "total weight exceeds the supported range", overflow);
            }
        }
        if (!targets.isEmpty() && total <= 0) {
            throw new ConfigValidationException(path, "total weight must be positive");
        }
    }

    private static Optional<String> chooseWeightedTarget(List<Target> targets,
                                                          Request request,
                                                          String salt) {
        if (targets == null || targets.isEmpty()) {
            return Optional.empty();
        }
        long totalWeight = targets.stream().mapToLong(Target::getWeight).sum();
        long bucket = hashRequest(request, salt) % totalWeight;
        long cumulative = 0;
        for (Target target : targets) {
            cumulative += target.weight;
            if (bucket < cumulative) {
                return Optional.of(target.group);
            }
        }
        throw new IllegalStateException("validated weighted target selection fell through");
    }

    private static long hashRequest(Request request, String salt) {
        CRC32 crc32 = new CRC32();
        String key = request.getRequestId() + "|" + request.getApiKey() + "|"
                + request.getSeqLen() + "|" + salt;
        crc32.update(key.getBytes(StandardCharsets.UTF_8));
        return crc32.getValue();
    }

    @Getter
    @Setter
    public static final class Rule {
        private String name;
        private Match match;
        private List<Target> targets = new ArrayList<>();

        private boolean matches(Request request) {
            return match != null && match.matches(request);
        }
    }

    @Getter
    @Setter
    public static final class Match {
        private List<String> apiKeys = new ArrayList<>();
        private InputTokens inputTokens;

        private boolean hasConstraint() {
            return (apiKeys != null && !apiKeys.isEmpty()) || inputTokens != null;
        }

        private void validate(String path) {
            if (apiKeys == null) {
                throw new ConfigValidationException(path + ".apiKeys", "must not be null");
            }
            Set<String> uniqueApiKeys = new HashSet<>();
            for (int index = 0; index < apiKeys.size(); index++) {
                String apiKey = apiKeys.get(index);
                if (apiKey == null || apiKey.isBlank()) {
                    throw new ConfigValidationException(
                            path + ".apiKeys[" + index + "]", "must not be blank");
                }
                if (!uniqueApiKeys.add(apiKey)) {
                    throw new ConfigValidationException(
                            path + ".apiKeys[" + index + "]", "must be unique");
                }
            }
            if (inputTokens != null) {
                inputTokens.validate(path + ".inputTokens");
            }
        }

        private boolean matches(Request request) {
            if (!apiKeys.isEmpty() && !apiKeys.contains(request.getApiKey())) {
                return false;
            }
            return inputTokens == null || inputTokens.matches(request.getSeqLen());
        }
    }

    @Getter
    @Setter
    public static final class InputTokens {
        private Long min;
        private Long max;

        private void validate(String path) {
            if (min == null && max == null) {
                throw new ConfigValidationException(path, "min or max is required");
            }
            if (min != null && min < 0) {
                throw new ConfigValidationException(path + ".min", "must be non-negative");
            }
            if (max != null && max < 0) {
                throw new ConfigValidationException(path + ".max", "must be non-negative");
            }
            if (min != null && max != null && min > max) {
                throw new ConfigValidationException(path, "min must not exceed max");
            }
        }

        private boolean matches(long value) {
            return (min == null || value >= min) && (max == null || value <= max);
        }
    }

    @Getter
    @Setter
    public static final class Target {
        private String group;
        private long weight = 1;
    }
}
