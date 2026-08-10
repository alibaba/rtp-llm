package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Getter
@Slf4j
@Component
public class ConfigService {

    private static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    private static final String PREFILL_TIME_FORMULA_ENV = "PREFILL_TIME_FORMULA";
    private static final String TRAFFIC_POLICY_CONFIG_ENV = "TRAFFIC_POLICY_CONFIG";
    private static final String TRAFFIC_POLICY_CONFIG_FILE_ENV = "TRAFFIC_POLICY_CONFIG_FILE";

    /**
     * Critical config fields whose parse failures must abort startup (fail-fast)
     * instead of silently falling back to defaults.
     *
     * <p>F4 (P0-4): includes {@code autoTpmEnabled} / {@code flexlbBatchQueueMaxSize}
     * (env parse failure aborts via the critical mechanism) and the two SLO spec
     * strings {@code autoTpmSloLengthBuckets} / {@code autoTpmPrioritySloMultipliers}
     * whose assignment never fails — those get a strict format pre-validation at
     * startup instead (see {@link #validateSloPolicySpecs}).
     */
    private static final Set<String> CRITICAL_CONFIG_FIELDS = Set.of(
            "defaultScheduleMode",
            "flexlbBatchAlgorithm",
            "flexlbBatchMaxCapacity",
            "flexlbBatchMaxInflight",
            "flexlbBatchFixedMaxInflightBatches",
            "flexlbBatchSloMaxInflightBatches",
            "costFormula",
            "prefillPredictorType",
            "autoTpmEnabled",
            "flexlbBatchQueueMaxSize",
            "autoTpmSloLengthBuckets",
            "autoTpmPrioritySloMultipliers");

    /** Env prefixes owned by FlexLB config; scanned for unmatched names (F3/P0-3). */
    private static final List<String> SCANNED_ENV_PREFIXES = List.of("FLEXLB_", "AUTO_TPM_", "COST_", "WORKER_");

    /** Deprecated env vars that already have dedicated warnings; excluded from the unmatched scan. */
    private static final Set<String> DEPRECATED_ENV_VARS = Set.of("FLEXLB_BATCH_ENABLED", "ENABLE_QUEUEING");

    private final FlexlbConfig flexlbConfig;

    public ConfigService() {
        this(System.getenv());
    }

    ConfigService(Map<String, String> environment) {
        String lbConfigStr = environment.get(FLEXLB_CONFIG_ENV);
        log.warn("FLEXLB_CONFIG = {}", lbConfigStr);
        FlexlbConfig config;
        if (lbConfigStr != null) {
            try {
                config = JsonUtils.toObject(lbConfigStr, FlexlbConfig.class);
            } catch (Exception e) {
                throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "Failed to parse FLEXLB_CONFIG JSON: " + e.getMessage(), e);
            }
        } else {
            config = new FlexlbConfig();
        }

        // If corresponding advanced environment variables exist, override and update
        applyEnvironmentOverrides(config, environment);
        applyTrafficPolicyOverride(config, environment);
        applyPrefillFormulaOverride(config, environment);

        warnDeprecatedEnvVars();
        warnUnmatchedEnvVars(environment);

        // Pre-validate critical parsed config at startup (fail-fast).
        // If these throw, startup must abort rather than letting every
        // per-request call fail with a 500.
        // Note: getParsedSloBuckets() is private in FlexlbConfig and silently
        // ignores parse errors, so it is not called here. getDefaultScheduleModeEnum()
        // throws IllegalArgumentException for invalid schedule mode values.
        config.getDefaultScheduleModeEnum();
        validateSloPolicySpecs(config);

        dumpEffectiveConfig(config);
        this.flexlbConfig = config;
    }

    public FlexlbConfig loadBalanceConfig() {
        return flexlbConfig;
    }

    public synchronized void updateTrafficPolicy(TrafficPolicyConfig trafficPolicy) {
        if (trafficPolicy == null) {
            throw new IllegalArgumentException("trafficPolicy cannot be null");
        }
        flexlbConfig.setTrafficPolicy(trafficPolicy);
        log.warn("Traffic policy updated: {}", JsonUtils.toStringOrEmpty(trafficPolicy));
    }

    /**
     * Apply environment variable overrides to configuration
     * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
     * Example: defaultScheduleMode -> DEFAULT_SCHEDULE_MODE
     */
    private void applyEnvironmentOverrides(FlexlbConfig config, Map<String, String> environment) {
        Field[] fields = FlexlbConfig.class.getDeclaredFields();
        for (Field field : fields) {
            // Only process primitive types and wrapper types
            Class<?> fieldType = field.getType();
            if (!isSupportedType(fieldType)) {
                continue;
            }

            String envVarName = camelToUpperSnakeCase(field.getName());
            String envValue = environment.get(envVarName);

            if (envValue != null && !envValue.trim().isEmpty()) {
                boolean isCritical = CRITICAL_CONFIG_FIELDS.contains(field.getName());
                try {
                    field.setAccessible(true);
                    Object parsedValue = parseValue(envValue.trim(), fieldType, envVarName);
                    Object oldValue = field.get(config);
                    field.set(config, parsedValue);
                    log.info(
                            "Environment variable override: {} = {} (field: {}, old value: {})",
                            envVarName,
                            parsedValue,
                            field.getName(),
                            oldValue);
                } catch (ConfigValidationException e) {
                    if (isCritical) {
                        throw e;
                    }
                    log.error(
                            "Failed to apply environment variable {}: {}",
                            envVarName,
                            e.getMessage(),
                            e);
                } catch (Exception e) {
                    if (isCritical) {
                        throw new ConfigValidationException(envVarName, e.getMessage(), e);
                    }
                    log.error(
                            "Failed to apply environment variable {}: {}",
                            envVarName,
                            e.getMessage(),
                            e);
                }
            }
        }
    }

    /**
     * Apply traffic policy from a standalone env var or file.
     * Priority: TRAFFIC_POLICY_CONFIG > TRAFFIC_POLICY_CONFIG_FILE > FLEXLB_CONFIG.trafficPolicy.
     */
    private void applyTrafficPolicyOverride(FlexlbConfig config, Map<String, String> environment) {
        String trafficPolicyConfig = environment.get(TRAFFIC_POLICY_CONFIG_ENV);
        String trafficPolicyConfigFile = environment.get(TRAFFIC_POLICY_CONFIG_FILE_ENV);

        if (StringUtils.isBlank(trafficPolicyConfig) && StringUtils.isNotBlank(trafficPolicyConfigFile)) {
            trafficPolicyConfig = readConfigFile(trafficPolicyConfigFile);
        }

        if (StringUtils.isBlank(trafficPolicyConfig)) {
            return;
        }

        try {
            TrafficPolicyConfig trafficPolicy = JsonUtils.toObject(trafficPolicyConfig, TrafficPolicyConfig.class);
            config.setTrafficPolicy(trafficPolicy);
            log.warn("Traffic policy loaded from standalone config: {}", JsonUtils.toStringOrEmpty(trafficPolicy));
        } catch (Exception e) {
            log.error("Failed to parse traffic policy config, skipping.", e);
        }
    }

    private void applyPrefillFormulaOverride(FlexlbConfig config, Map<String, String> environment) {
        String formula = environment.get(PREFILL_TIME_FORMULA_ENV);
        if (StringUtils.isBlank(formula)) {
            // Blank or unset formula means skip the override, preserving any formula
            // set via FLEXLB_CONFIG or COST_FORMULA. This maintains backward
            // compatibility with deployment scripts that set PREFILL_TIME_FORMULA=""
            // to cancel a formula override without clearing existing config.
            return;
        }
        config.setCostFormula(formula);
        log.warn("Prefill time formula loaded from {}: {}", PREFILL_TIME_FORMULA_ENV, formula);
    }

    private String readConfigFile(String filePath) {
        try {
            return Files.readString(Path.of(filePath), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read config file: " + filePath, e);
        }
    }

    /**
     * Check if the type is supported
     */
    private boolean isSupportedType(Class<?> type) {
        return type == int.class
                || type == Integer.class
                || type == long.class
                || type == Long.class
                || type == double.class
                || type == Double.class
                || type == boolean.class
                || type == Boolean.class
                || type == String.class
                || type.isEnum();
    }

    /**
     * Convert camel case to upper snake case
     * Example: defaultScheduleMode -> DEFAULT_SCHEDULE_MODE
     */
    private static String camelToUpperSnakeCase(String camelCase) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < camelCase.length(); i++) {
            char c = camelCase.charAt(i);
            if (Character.isUpperCase(c) && i > 0) {
                result.append('_');
            }
            result.append(Character.toUpperCase(c));
        }
        return result.toString();
    }

    /**
     * Parse string value based on target type
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private Object parseValue(String value, Class<?> targetType, String fieldName) {
        if (targetType == String.class) {
            return value;
        } else if (targetType == int.class || targetType == Integer.class) {
            return Integer.parseInt(value);
        } else if (targetType == long.class || targetType == Long.class) {
            return Long.parseLong(value);
        } else if (targetType == double.class || targetType == Double.class) {
            return Double.parseDouble(value);
        } else if (targetType == boolean.class || targetType == Boolean.class) {
            return parseStrictBoolean(value, fieldName);
        } else if (targetType.isEnum()) {
            return JsonUtils.toObject("\"" + value + "\"", targetType);
        }
        throw new IllegalArgumentException("Unsupported type: " + targetType);
    }

    /**
     * Strictly parse a boolean value, rejecting unrecognized strings.
     *
     * @throws ConfigValidationException if the value is not a recognized boolean literal.
     */
    static boolean parseStrictBoolean(String value, String fieldName) {
        String v = value.trim().toLowerCase();
        if ("true".equals(v) || "1".equals(v) || "yes".equals(v) || "on".equals(v) || "enabled".equals(v)) return true;
        if ("false".equals(v) || "0".equals(v) || "no".equals(v) || "off".equals(v) || "disabled".equals(v)) return false;
        throw new ConfigValidationException(fieldName,
            "Invalid boolean value '" + value + "'. Expected: true/false/1/0/yes/no/on/off/enabled/disabled");
    }

    /**
     * Log the effective configuration after all overrides have been applied.
     * Only dumps critical scheduling config — no sensitive information.
     */
    private void dumpEffectiveConfig(FlexlbConfig config) {
        log.info("===== FlexLB Effective Configuration =====");
        log.info("scheduleMode={}, batchAlgorithm={}",
            config.getDefaultScheduleMode(), config.getFlexlbBatchAlgorithm());
        log.info("batchMaxCapacity={}, batchMaxInflight={}",
            config.getFlexlbBatchMaxCapacity(), config.getFlexlbBatchMaxInflight());
        log.info("fixedMaxInflightBatches={}, sloMaxInflightBatches={}",
            config.getFlexlbBatchFixedMaxInflightBatches(),
            config.getFlexlbBatchSloMaxInflightBatches());
        log.info("prefillPredictorType={}", config.getPrefillPredictorType());
        log.info("autoTpmEnabled={}, autoTpmDefaultPriority={}",
            config.isAutoTpmEnabled(), config.getAutoTpmDefaultPriority());
        log.info("autoTpmSloLengthBuckets={}, autoTpmPrioritySloMultipliers={}",
            config.getAutoTpmSloLengthBuckets(), config.getAutoTpmPrioritySloMultipliers());
        log.info("autoTpmPrefillQueueEvictEnabled={}, autoTpmDecodeReservedEvictEnabled={}",
            config.isAutoTpmPrefillQueueEvictEnabled(), config.isAutoTpmDecodeReservedEvictEnabled());
        log.info("autoTpmPlanCacheHitBenefitCap={}", config.getAutoTpmPlanCacheHitBenefitCap());
        log.info("autoTpmDecodeAcceptedEvictEnabled={}",
            config.isAutoTpmDecodeAcceptedEvictEnabled());
        log.info("autoTpmCommitWaitReleaseTimeoutMs={}",
            config.getAutoTpmCommitWaitReleaseTimeoutMs());
        log.info("autoTpmCommitStrategy={}, autoTpmVictimGuardMode={}",
            config.getAutoTpmCommitStrategy(), config.getAutoTpmVictimGuardMode());
        log.info("workerTimeoutMs={}",
            config.getWorkerTimeoutMs());
        log.info("==========================================");
    }

    private void warnDeprecatedEnvVars() {
        Map<String, String> env = System.getenv();
        if (env.containsKey("FLEXLB_BATCH_ENABLED")) {
            log.warn("Environment variable FLEXLB_BATCH_ENABLED is deprecated and ignored. Use DEFAULT_SCHEDULE_MODE=BATCH|DIRECT|QUEUE instead.");
        }
        if (env.containsKey("ENABLE_QUEUEING")) {
            log.warn("Environment variable ENABLE_QUEUEING is deprecated and ignored. Use DEFAULT_SCHEDULE_MODE=BATCH|DIRECT|QUEUE instead.");
        }
    }

    /**
     * F4 (P0-4): the SLO bucket/multiplier specs are plain strings whose env
     * assignment never fails, so strictly pre-validate the effective values at
     * startup — an invalid spec aborts instead of silently falling back to the
     * built-in defaults at runtime. Blank means "use built-in default" and is
     * allowed. The lenient runtime fallback in {@link PrioritySloPolicy} itself
     * is intentionally unchanged.
     */
    private static void validateSloPolicySpecs(FlexlbConfig config) {
        String bucketSpec = config.getAutoTpmSloLengthBuckets();
        String invalidBucket = PrioritySloPolicy.firstInvalidBucketEntry(bucketSpec);
        if (invalidBucket != null) {
            throw new ConfigValidationException("autoTpmSloLengthBuckets",
                    "Invalid SLO length bucket fragment '" + invalidBucket + "' in '" + bucketSpec
                            + "'. Expected format like '256:150,1024:300,*:2400'");
        }
        String multiplierSpec = config.getAutoTpmPrioritySloMultipliers();
        String invalidMultiplier = PrioritySloPolicy.firstInvalidMultiplierEntry(multiplierSpec);
        if (invalidMultiplier != null) {
            throw new ConfigValidationException("autoTpmPrioritySloMultipliers",
                    "Invalid priority SLO multiplier fragment '" + invalidMultiplier + "' in '" + multiplierSpec
                            + "'. Expected format like '30:2.0,40:1.5,50:1.0'");
        }
    }

    /**
     * F3 (P0-3): after all overrides are applied, warn about every
     * FLEXLB_/AUTO_TPM_/COST_-prefixed environment variable that matches no
     * config field — previously such misspelled names were silently ignored
     * (e.g. an intended queue-size override never taking effect). Warn-only by
     * design: these prefixes may be shared by unrelated system variables, so
     * aborting would be too aggressive.
     */
    private void warnUnmatchedEnvVars(Map<String, String> environment) {
        Set<String> known = knownEnvVarNames();
        for (String name : findUnmatchedEnvVars(environment)) {
            String suggestion = nearestKnownEnvName(name, known);
            log.warn("环境变量 {} 未匹配任何配置字段，将被忽略{}", name,
                    suggestion == null ? "" : "（did-you-mean: " + suggestion + "？）");
        }
    }

    /**
     * Pure scan (package-private for tests): given an environment map, return
     * the sorted FLEXLB_/AUTO_TPM_/COST_-prefixed variable names that neither
     * map to any {@link FlexlbConfig} field (camelCase → UPPER_SNAKE_CASE) nor
     * are special config entry points nor known deprecated names.
     */
    static List<String> findUnmatchedEnvVars(Map<String, String> environment) {
        Set<String> known = knownEnvVarNames();
        return environment.keySet().stream()
                .filter(ConfigService::hasScannedPrefix)
                .filter(name -> !known.contains(name))
                .filter(name -> !DEPRECATED_ENV_VARS.contains(name))
                .sorted()
                .toList();
    }

    private static boolean hasScannedPrefix(String name) {
        for (String prefix : SCANNED_ENV_PREFIXES) {
            if (name.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    /** Full env-name mapping: every FlexlbConfig field plus the special config entry points. */
    static Set<String> knownEnvVarNames() {
        Set<String> names = new HashSet<>();
        for (Field field : FlexlbConfig.class.getDeclaredFields()) {
            names.add(camelToUpperSnakeCase(field.getName()));
        }
        names.add(FLEXLB_CONFIG_ENV);
        names.add(PREFILL_TIME_FORMULA_ENV);
        names.add(TRAFFIC_POLICY_CONFIG_ENV);
        names.add(TRAFFIC_POLICY_CONFIG_FILE_ENV);
        return names;
    }

    /** Cheap did-you-mean: nearest known env name within edit distance 2, or null. */
    static String nearestKnownEnvName(String name, Set<String> known) {
        String best = null;
        int bestDist = 3;
        for (String candidate : known) {
            int dist = boundedEditDistance(name, candidate, bestDist);
            if (dist < bestDist) {
                bestDist = dist;
                best = candidate;
            }
        }
        return best;
    }

    /** Levenshtein distance capped at {@code limit} (returns {@code limit} when exceeded). */
    private static int boundedEditDistance(String a, String b, int limit) {
        if (Math.abs(a.length() - b.length()) >= limit) {
            return limit;
        }
        int[] prev = new int[b.length() + 1];
        int[] curr = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            curr[0] = i;
            int rowMin = curr[0];
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
                rowMin = Math.min(rowMin, curr[j]);
            }
            if (rowMin >= limit) {
                return limit;
            }
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return Math.min(prev[b.length()], limit);
    }
}
