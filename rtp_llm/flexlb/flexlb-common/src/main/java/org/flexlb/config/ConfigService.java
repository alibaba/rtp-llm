package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.config.ConfigSource;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Loads the strict FLEXLB_CONFIG document and the independent MODEL_SERVICE_CONFIG document.
 * Registered configuration sources may update only the FlexLB behavior snapshot.
 */
@Slf4j
@Component
@DependsOn({"environmentConfigSource", "nacosConfigSource"})
public class ConfigService {

    public static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    public static final String MODEL_SERVICE_CONFIG_ENV = "MODEL_SERVICE_CONFIG";

    private static final ObjectMapper STRICT_MAPPER = JsonMapper.builder()
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
            .enable(DeserializationFeature.FAIL_ON_NUMBERS_FOR_ENUMS)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .disable(DeserializationFeature.ACCEPT_FLOAT_AS_INT)
            .disable(MapperFeature.ALLOW_COERCION_OF_SCALARS)
            .serializationInclusion(JsonInclude.Include.NON_NULL)
            .build();

    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();
    private static final Set<String> MODEL_BEHAVIOR_FIELDS = Set.of(
            "connect_timeout_ms",
            "connectTimeoutMs",
            "read_timeout_ms",
            "readTimeoutMs",
            "poll_interval_ms",
            "pollIntervalMs",
            "max_idle_connections",
            "maxIdleConnections",
            "keep_alive_duration_ms",
            "keepAliveDurationMs",
            "request_timeout_ms",
            "requestTimeoutMs",
            "leader_refresh_interval_ms",
            "leaderRefreshIntervalMs",
            "heartbeat_failure_threshold",
            "heartbeatFailureThreshold",
            "query_failure_threshold",
            "queryFailureThreshold",
            "max_query_retry_count",
            "maxQueryRetryCount",
            "recovery_success_threshold",
            "recoverySuccessThreshold",
            "p2p_host_count",
            "p2pHostCount",
            "local_standby",
            "localStandby");

    private final AtomicReference<FlexlbConfig> currentConfig;
    private final ServiceRoute modelServiceConfig;
    private final List<Consumer<FlexlbConfig>> updateListeners = new ArrayList<>();
    private final Object updateLock = new Object();

    public ConfigService() {
        Map<String, String> environment = System.getenv();
        this.currentConfig = new AtomicReference<>(new FlexlbConfig());
        this.modelServiceConfig = loadModelServiceConfig(environment);

        List<ConfigSource> sources = registeredSources();
        if (sources.isEmpty()) {
            currentConfig.set(loadEnvironmentConfig(environment));
        } else {
            initializeConfigSources(sources);
        }
        logEffectiveConfig(currentConfig.get());
    }

    ConfigService(Map<String, String> environment) {
        this.currentConfig = new AtomicReference<>(loadEnvironmentConfig(environment));
        this.modelServiceConfig = loadModelServiceConfig(environment);
        logEffectiveConfig(currentConfig.get());
    }

    public static synchronized void register(ConfigSource source) {
        if (!CONFIG_SOURCES.contains(source)) {
            CONFIG_SOURCES.add(source);
        }
    }

    private static synchronized List<ConfigSource> registeredSources() {
        List<ConfigSource> sources = new ArrayList<>(CONFIG_SOURCES);
        sources.sort(Comparator.comparingInt(ConfigSource::priority));
        return sources;
    }

    public static FlexlbConfig parse(String document) {
        try {
            JsonNode tree = STRICT_MAPPER.readTree(document);
            rejectJsonNull(tree, "$", FLEXLB_CONFIG_ENV);
            FlexlbConfig config = STRICT_MAPPER.treeToValue(tree, FlexlbConfig.class);
            FlexlbConfigValidator.validate(config);
            return config;
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(FLEXLB_CONFIG_ENV,
                    "Invalid FLEXLB_CONFIG JSON: " + error.getMessage(), error);
        }
    }

    public static String serialize(FlexlbConfig config) {
        FlexlbConfigValidator.validate(config);
        try {
            return STRICT_MAPPER.writeValueAsString(config);
        } catch (Exception error) {
            throw new IllegalStateException("Failed to serialize FlexLB configuration", error);
        }
    }

    public FlexlbConfig loadBalanceConfig() {
        return currentConfig.get();
    }

    static ServiceRoute parseModelServiceConfig(String document) {
        try {
            JsonNode tree = STRICT_MAPPER.readTree(document);
            rejectJsonNull(tree, "$", MODEL_SERVICE_CONFIG_ENV);
            rejectModelBehaviorFields(tree, "$", null);
            return STRICT_MAPPER.treeToValue(tree, ServiceRoute.class);
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(MODEL_SERVICE_CONFIG_ENV,
                    "Invalid MODEL_SERVICE_CONFIG JSON: " + error.getMessage(), error);
        }
    }

    public ServiceRoute loadModelServiceConfig() {
        return modelServiceConfig;
    }

    public void addUpdateListener(Consumer<FlexlbConfig> listener) {
        synchronized (updateLock) {
            updateListeners.add(listener);
            listener.accept(currentConfig.get());
        }
    }

    public void updateTrafficPolicy(TrafficPolicyConfig groupSelector) {
        if (groupSelector == null) {
            throw new IllegalArgumentException("groupSelector cannot be null");
        }
        TrafficPolicyConfig.validate(groupSelector);
        synchronized (updateLock) {
            try {
                ObjectNode document = (ObjectNode) STRICT_MAPPER.valueToTree(currentConfig.get());
                ObjectNode router = (ObjectNode) document.get("router");
                router.set("groupSelector", STRICT_MAPPER.valueToTree(groupSelector));
                FlexlbConfig updated = parse(STRICT_MAPPER.writeValueAsString(document));
                currentConfig.set(updated);
                notifyUpdateListeners(updated);
                log.info("Group selector updated: rules={}", groupSelector.getRules().size());
            } catch (ConfigValidationException error) {
                throw error;
            } catch (Exception error) {
                throw new IllegalStateException("Failed to update group selector", error);
            }
        }
    }

    private static FlexlbConfig loadEnvironmentConfig(Map<String, String> environment) {
        String document = environment.get(FLEXLB_CONFIG_ENV);
        FlexlbConfig config = document == null ? new FlexlbConfig() : parse(document);
        FlexlbConfigValidator.validate(config);
        return config;
    }

    private static ServiceRoute loadModelServiceConfig(Map<String, String> environment) {
        String document = environment.get(MODEL_SERVICE_CONFIG_ENV);
        return document == null || document.isBlank() ? null : parseModelServiceConfig(document);
    }

    private void initializeConfigSources(List<ConfigSource> sources) {
        ConfigSource loadingSource = null;
        try {
            synchronized (updateLock) {
                for (ConfigSource source : sources) {
                    loadingSource = source;
                    source.setUpdateListener(content -> receiveConfigUpdate(source, content));
                    currentConfig.set(mergeConfig(
                            currentConfig.get(), source.load(), source.name()));
                    loadingSource = null;
                    log.info("Loaded FlexLB configuration from {} source", source.name());
                }
            }
        } catch (Exception error) {
            closeConfigSources();
            throw new IllegalStateException("Failed to initialize FlexLB configuration from "
                    + (loadingSource == null ? "configured source" : loadingSource.name()), error);
        }
    }

    private void receiveConfigUpdate(ConfigSource source, String content) {
        synchronized (updateLock) {
            try {
                FlexlbConfig previous = currentConfig.get();
                FlexlbConfig updated = mergeConfig(previous, content, source.name());
                if (updated == previous) {
                    log.info("Ignored empty FlexLB configuration update from {} source",
                            source.name());
                    return;
                }
                currentConfig.set(updated);
                notifyUpdateListeners(updated);
                logEffectiveConfig(updated);
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception error) {
                log.error("Rejected invalid FlexLB configuration update from {} source; "
                                + "keeping last-known-good configuration: {}",
                        source.name(), error.getMessage());
            }
        }
    }

    private static FlexlbConfig mergeConfig(
            FlexlbConfig baseConfig, String content, String sourceName) {
        ObjectNode overrides = parseOverrides(content, sourceName);
        if (overrides.isEmpty()) {
            return baseConfig;
        }
        ObjectNode merged = STRICT_MAPPER.valueToTree(baseConfig);
        deepMerge(merged, overrides);
        try {
            return parse(STRICT_MAPPER.writeValueAsString(merged));
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalArgumentException(
                    "Failed to merge " + sourceName + " configuration", error);
        }
    }

    private static ObjectNode parseOverrides(String content, String sourceName) {
        if (content == null || content.isBlank()) {
            return STRICT_MAPPER.createObjectNode();
        }
        try {
            JsonNode parsed = STRICT_MAPPER.readTree(content);
            rejectJsonNull(parsed, "$", sourceName);
            if (!(parsed instanceof ObjectNode overrides)) {
                throw new IllegalArgumentException(
                        sourceName + " configuration must be a JSON object");
            }
            return overrides;
        } catch (IllegalArgumentException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalArgumentException(
                    "Invalid " + sourceName + " configuration: " + error.getMessage(), error);
        }
    }

    private static void deepMerge(ObjectNode target, ObjectNode overrides) {
        Iterator<Map.Entry<String, JsonNode>> fields = overrides.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            JsonNode current = target.get(field.getKey());
            JsonNode replacement = field.getValue();
            if (current instanceof ObjectNode currentObject
                    && replacement instanceof ObjectNode replacementObject
                    && hasCompatibleType(currentObject, replacementObject)) {
                deepMerge(currentObject, replacementObject);
            } else {
                target.set(field.getKey(), replacement.deepCopy());
            }
        }
    }

    private static boolean hasCompatibleType(
            ObjectNode current, ObjectNode replacement) {
        JsonNode replacementType = replacement.get("type");
        return replacementType == null || replacementType.equals(current.get("type"));
    }

    private void notifyUpdateListeners(FlexlbConfig config) {
        for (Consumer<FlexlbConfig> listener : updateListeners) {
            try {
                listener.accept(config);
            } catch (RuntimeException error) {
                log.error("FlexLB configuration update listener failed", error);
            }
        }
    }

    private static void rejectJsonNull(JsonNode node, String path, String sourceName) {
        if (node == null || node.isNull()) {
            throw new ConfigValidationException(sourceName,
                    "JSON null is not allowed at " + path);
        }
        if (node.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                rejectJsonNull(field.getValue(), path + "." + field.getKey(), sourceName);
            }
        } else if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                rejectJsonNull(node.get(index), path + "[" + index + "]", sourceName);
            }
        }
    }

    private static void rejectModelBehaviorFields(
            JsonNode node,
            String path,
            String parentField) {
        if (node == null || !node.isContainerNode()) {
            return;
        }
        if (node.isArray()) {
            for (int index = 0; index < node.size(); index++) {
                rejectModelBehaviorFields(
                        node.get(index), path + "[" + index + "]", parentField);
            }
            return;
        }

        Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            String name = field.getKey();
            String fieldPath = path + "." + name;
            boolean rootBehavior = "$".equals(path) && "load_balance".equals(name);
            boolean enableBehavior = "enabled".equals(name)
                    && ("kvcm".equals(parentField) || "optimizer".equals(parentField));
            if (rootBehavior || enableBehavior || MODEL_BEHAVIOR_FIELDS.contains(name)) {
                throw new ConfigValidationException(
                        MODEL_SERVICE_CONFIG_ENV,
                        fieldPath + " is a FlexLB behavior field; configure it in FLEXLB_CONFIG");
            }
            rejectModelBehaviorFields(field.getValue(), fieldPath, name);
        }
    }

    private static void logEffectiveConfig(FlexlbConfig config) {
        String scheduler = config.isDirect() ? "DIRECT" : "QUEUE";
        String ordering = config.isDirect() ? "N/A"
                : config.isPriorityOrdering() ? "PRIORITY" : "FIFO";
        String dispatcher = config.isBatchDispatch() ? "BATCH" : "NON_BATCH";
        log.info("FlexLB config loaded: schemaVersion={}, scheduler={}, ordering={}, dispatcher={}, "
                        + "cacheMatching={}, consistency={}, prefillSelector={}, decodeSelector={}, groupRules={}",
                config.getSchemaVersion(), scheduler, ordering, dispatcher,
                config.getCacheMatching().getClass().getSimpleName(),
                config.getConsistency().getClass().getSimpleName(),
                config.getRouter().getRoles().getPrefill().getSelector()
                        .getClass().getSimpleName(),
                config.getRouter().getRoles().getDecode().getSelector()
                        .getClass().getSimpleName(),
                config.getRouter().getGroupSelector() == null ? 0
                        : config.getRouter().getGroupSelector().getRules().size());
    }

    @PreDestroy
    public void close() {
        closeConfigSources();
    }

    private static synchronized void closeConfigSources() {
        for (ConfigSource source : CONFIG_SOURCES) {
            closeQuietly(source);
        }
        CONFIG_SOURCES.clear();
    }

    private static void closeQuietly(ConfigSource source) {
        try {
            source.close();
        } catch (Exception error) {
            log.warn("Failed to close {} configuration source", source.name(), error);
        }
    }
}
