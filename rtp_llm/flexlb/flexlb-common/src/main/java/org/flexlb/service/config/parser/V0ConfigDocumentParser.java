package org.flexlb.service.config.parser;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigSchemaVersion;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts one V0 document into the two configuration documents used by
 * the current runtime: FlexLB behaviour and model-service topology.
 *
 * <p>The parser owns only version selection and normalization. Configuration
 * acquisition and update lifecycle stay in {@link ConfigSource} implementations.</p>
 */
@Slf4j
@Component
public final class V0ConfigDocumentParser implements ConfigDocumentParser {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public V0ConfigDocumentParser() {
        ConfigDocumentParserResolver.register(this);
    }

    @Override
    public int schemaVersion() {
        return ConfigSchemaVersion.V0_COMPATIBILITY;
    }

    @Override
    public NormalizedConfig parse(String rawFlexlbConfig, String rawModelServiceConfig) {
        if (rawFlexlbConfig == null || rawFlexlbConfig.isBlank()) {
            throw new IllegalArgumentException("V0 compatibility configuration document must not be null or blank");
        }
        try {
            ObjectNode document = parseV0Document(rawFlexlbConfig);
            NormalizedConfig converted = convertV0Document(document);
            return rawModelServiceConfig == null || rawModelServiceConfig.isBlank() ? converted
                    : new NormalizedConfig(converted.flexlbConfig(), rawModelServiceConfig, schemaVersion());
        } catch (JsonProcessingException error) {
            throw new IllegalArgumentException("Invalid V0 configuration document", error);
        }
    }

    private static ObjectNode parseV0Document(String v0ConfigDocument) throws JsonProcessingException {
        JsonNode parsed = MAPPER.readTree(v0ConfigDocument);
        if (!(parsed instanceof ObjectNode document)) {
            throw new IllegalArgumentException("V0 configuration must be a JSON object");
        }
        return document;
    }

    private static NormalizedConfig convertV0Document(ObjectNode v0Config) throws JsonProcessingException {
        ObjectNode flexlbConfig = convertFlexlbConfig(v0Config);
        String modelServiceConfig = convertModelServiceConfig(v0Config);
        List<String> unmappedFields = collectUnsupportedFields(v0Config);
        if (!unmappedFields.isEmpty()) {
            log.warn("V0 configuration contains unmapped compatibility fields: {}", unmappedFields);
        }
        return new NormalizedConfig(MAPPER.writeValueAsString(flexlbConfig), modelServiceConfig, ConfigSchemaVersion.V0_COMPATIBILITY);
    }

    private static ObjectNode convertFlexlbConfig(ObjectNode v0Config) {
        ObjectNode flexlbConfig = MAPPER.createObjectNode();
        flexlbConfig.put("schemaVersion", ConfigSchemaVersion.STANDARD);
        copyField(v0Config, flexlbConfig, "blockHashStrategy");
        copyField(v0Config, flexlbConfig, "enableFallback");

        configureScheduling(v0Config, flexlbConfig);
        configureRouting(v0Config, flexlbConfig);
        configureWorkerHealth(v0Config, flexlbConfig);
        configureLogging(v0Config, flexlbConfig);
        configureConsistency(v0Config, flexlbConfig);
        configureKvcmAndOptimizer(v0Config, flexlbConfig);
        return flexlbConfig;
    }

    private static void configureScheduling(ObjectNode v0Config, ObjectNode flexlbConfig) {
        boolean queueingEnabled = v0Config.path("enableQueueing").asBoolean(false);
        ObjectNode scheduler = flexlbConfig.putObject("scheduler");
        scheduler.put("type", queueingEnabled ? "QUEUE" : "DIRECT");
        if (queueingEnabled && v0Config.has("maxQueueSize")) {
            scheduler.putObject("capacity").set("maxOutstandingRequestsGlobal", v0Config.get("maxQueueSize").deepCopy());
        }
        flexlbConfig.putObject("dispatcher").put("type", "NON_BATCH");
    }

    private static void configureRouting(ObjectNode v0Config, ObjectNode flexlbConfig) {
        ObjectNode prefill = flexlbConfig.putObject("router").putObject("roles").putObject("prefill");
        ObjectNode candidateChoice = prefill.putObject("selector").put("type", "ESTIMATED_TTFT").putObject("candidateChoice");
        candidateChoice.put("type", "RANDOM_WITHIN_TOLERANCE");
        copyField(v0Config, candidateChoice, "shortestTtftSimilarityThresholdRatio", "relativeTolerance");
        if (v0Config.has("prefillQueueSizeThreshold")) {
            prefill.putObject("availability").set("maxPendingRequests", v0Config.get("prefillQueueSizeThreshold").deepCopy());
        }
        configureCacheAffinity(v0Config, prefill);
    }

    private static void configureCacheAffinity(ObjectNode v0Config, ObjectNode prefill) {
        boolean configured = "CACHE_AFFINITY_FIRST".equals(v0Config.path("loadBalanceStrategy").asText())
                || v0Config.has("p2pHitDiscount")
                || v0Config.has("cacheAffinityFirstMinHitRate")
                || v0Config.has("cacheAffinityFirstOutstandingUncachedTokensThreshold");
        if (!configured) {
            return;
        }
        ObjectNode cacheAffinity = prefill.putObject("cacheAffinity");
        copyField(v0Config, cacheAffinity, "p2pHitDiscount");
        copyField(v0Config, cacheAffinity, "cacheAffinityFirstMinHitRate", "minPrefixHitPercent");
        copyField(v0Config, cacheAffinity, "cacheAffinityFirstOutstandingUncachedTokensThreshold", "maxOutstandingUncachedTokens");
    }

    private static void configureWorkerHealth(ObjectNode v0Config, ObjectNode flexlbConfig) {
        if (!v0Config.has("syncStatusInterval") && !v0Config.has("syncRequestTimeoutMs")) {
            return;
        }
        ObjectNode health = flexlbConfig.putObject("workerRegistry").putObject("health");
        copyField(v0Config, health, "syncStatusInterval", "statusPollIntervalMs");
        copyField(v0Config, health, "syncRequestTimeoutMs", "statusRpcTimeoutMs");
    }

    private static void configureLogging(ObjectNode v0Config, ObjectNode flexlbConfig) {
        if (!v0Config.has("flexlbLogLevel") && !v0Config.has("enableStdoutLog")) {
            return;
        }
        ObjectNode logging = flexlbConfig.putObject("observability").putObject("logging");
        copyField(v0Config, logging, "flexlbLogLevel", "level");
        copyField(v0Config, logging, "enableStdoutLog", "stdoutEnabled");
    }

    private static void configureConsistency(ObjectNode v0Config, ObjectNode flexlbConfig) {
        ObjectNode consistencyConfig = objectAt(v0Config, "flexlbSyncConsistencyConfig");
        if (consistencyConfig == null) {
            return;
        }
        if (!consistencyConfig.path("needConsistency").asBoolean(false)) {
            flexlbConfig.putObject("consistency").put("type", "NONE");
            return;
        }
        if (!"ZOOKEEPER".equals(consistencyConfig.path("masterElectType").asText())) {
            throw new IllegalArgumentException("V0 configuration supports only ZOOKEEPER master election");
        }
        ObjectNode zookeeperConfig = objectAt(consistencyConfig, "zookeeperConfig");
        if (zookeeperConfig == null || !zookeeperConfig.has("zkHost")) {
            throw new IllegalArgumentException("V0 ZOOKEEPER configuration requires zookeeperConfig.zkHost");
        }
        ObjectNode consistency = flexlbConfig.putObject("consistency");
        consistency.put("type", "ZOOKEEPER");
        copyField(zookeeperConfig, consistency, "zkHost", "connectString");
        copyField(zookeeperConfig, consistency, "zkTimeoutMs", "sessionTimeoutMs");
        copyField(zookeeperConfig, consistency, "zkTimeoutMs", "connectionTimeoutMs");
    }

    private static void configureKvcmAndOptimizer(ObjectNode v0Config, ObjectNode flexlbConfig) {
        ObjectNode modelService = objectAt(v0Config, "modelServiceConfig");
        if (modelService == null) {
            return;
        }
        configureKvcm(flexlbConfig, objectAt(modelService, "kvcm"));
        configureOptimizer(flexlbConfig, objectAt(modelService, "optimizer"));
    }

    private static void configureKvcm(ObjectNode flexlbConfig, ObjectNode kvcmConfig) {
        if (kvcmConfig == null || !kvcmConfig.path("enabled").asBoolean(false)) {
            return;
        }
        ObjectNode cacheMatching = flexlbConfig.putObject("cacheMatching");
        cacheMatching.put("type", "KVCM");
        copyField(kvcmConfig, cacheMatching, "p2p_host_count", "p2pHostCount");
        ObjectNode localStandby = objectAt(kvcmConfig, "local_standby");
        if (localStandby != null) {
            ObjectNode currentLocalStandby = cacheMatching.putObject("localStandby");
            copySnakeCaseFields(localStandby, currentLocalStandby);
        }
    }

    private static void configureOptimizer(ObjectNode flexlbConfig, ObjectNode optimizerConfig) {
        if (optimizerConfig != null && optimizerConfig.has("enabled")) {
            flexlbConfig.putObject("optimizer").set("enabled", optimizerConfig.get("enabled").deepCopy());
        }
    }

    private static List<String> collectUnsupportedFields(ObjectNode v0Config) {
        List<String> unmappedFields = new ArrayList<>();
        if (v0Config.has("cacheAffinityFirstMaxExtraWorkTokens")) {
            unmappedFields.add("cacheAffinityFirstMaxExtraWorkTokens (tokens cannot be converted to cacheAffinity.maxExtraTtftMs)");
        }
        if (v0Config.has("scheduleWorkerSize")) {
            unmappedFields.add("scheduleWorkerSize (runtime worker sizing is no longer public configuration)");
        }
        if (v0Config.has("fixedScheduleWorkerPermits")) {
            unmappedFields.add("fixedScheduleWorkerPermits (no equivalent in the current scheduler)");
        }
        return unmappedFields;
    }

    private static String convertModelServiceConfig(ObjectNode v0Config) throws JsonProcessingException {
        ObjectNode modelService = objectAt(v0Config, "modelServiceConfig");
        if (modelService == null) {
            return null;
        }
        ObjectNode currentModelService = modelService.deepCopy();
        removeField(currentModelService, "kvcm", "enabled");
        removeField(currentModelService, "kvcm", "p2p_host_count");
        removeField(currentModelService, "kvcm", "local_standby");
        removeField(currentModelService, "optimizer", "enabled");
        return MAPPER.writeValueAsString(currentModelService);
    }

    private static ObjectNode objectAt(ObjectNode parent, String fieldName) {
        JsonNode value = parent.get(fieldName);
        return value instanceof ObjectNode object ? object : null;
    }

    private static void removeField(ObjectNode parent, String childName, String fieldName) {
        ObjectNode child = objectAt(parent, childName);
        if (child != null) {
            child.remove(fieldName);
        }
    }

    private static void copySnakeCaseFields(ObjectNode source, ObjectNode target) {
        source.fields().forEachRemaining(entry -> target.set(toCamelCase(entry.getKey()), entry.getValue().deepCopy()));
    }

    private static String toCamelCase(String name) {
        StringBuilder result = new StringBuilder();
        boolean uppercaseNext = false;
        for (int index = 0; index < name.length(); index++) {
            char character = name.charAt(index);
            if (character == '_') {
                uppercaseNext = true;
            } else if (uppercaseNext) {
                result.append(Character.toUpperCase(character));
                uppercaseNext = false;
            } else {
                result.append(character);
            }
        }
        return result.toString();
    }

    private static void copyField(ObjectNode source, ObjectNode target, String fieldName) {
        copyField(source, target, fieldName, fieldName);
    }

    private static void copyField(ObjectNode source, ObjectNode target, String sourceFieldName, String targetFieldName) {
        if (source.has(sourceFieldName)) {
            target.set(targetFieldName, source.get(sourceFieldName).deepCopy());
        }
    }

}
