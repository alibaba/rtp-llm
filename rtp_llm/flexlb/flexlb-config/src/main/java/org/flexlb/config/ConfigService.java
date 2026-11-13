package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.loading.ConfigSchemaVersion;
import org.flexlb.config.loading.ConfigValidationException;
import org.flexlb.config.loading.FlexlbConfigMerger;
import org.flexlb.config.loading.FlexlbConfigValidator;
import org.flexlb.config.loading.NormalizedConfig;
import org.flexlb.config.loading.parser.ConfigDocumentParser;
import org.flexlb.config.loading.parser.ConfigDocumentParserResolver;
import org.flexlb.config.loading.parser.ModelServiceConfigParser;
import org.flexlb.config.loading.parser.StandardConfigDocumentParser;
import org.flexlb.config.loading.parser.V0ConfigDocumentParser;
import org.flexlb.config.loading.source.ConfigSource;
import org.flexlb.config.routing.TrafficPolicyConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.config.service.ServiceRoute;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
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
@DependsOn({"environmentConfigSource", "nacosConfigSource", "uniConfigConfigSource"})
public class ConfigService {
    public static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();

    private final AtomicReference<FlexlbConfig> currentFlexlbConfig;
    private final AtomicReference<ServiceRoute> currentModelServiceConfig;
    private final List<Consumer<FlexlbConfig>> updateListeners = new ArrayList<>();
    private final StartupScheduler startupScheduler;
    private final Object updateLock = new Object();

    private int configSchemaVersion = ConfigSchemaVersion.V0_COMPATIBILITY;

    /**
     * Compatibility constructor for direct construction in tests and embedders.
     */
    public ConfigService() {
        this(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
    }

    public ConfigService(String document) {
        if (document == null || document.isBlank()) {
            throw new ConfigValidationException(
                    FLEXLB_CONFIG_ENV, "is required; configure requestLifecycle.request.timeoutMs");
        }
        FlexlbConfig config = parse(document);
        this.currentFlexlbConfig = new AtomicReference<>(config);
        this.currentModelServiceConfig = new AtomicReference<>();
        this.startupScheduler = StartupScheduler.from(config);
        this.configSchemaVersion = config.getSchemaVersion();
    }

    @Autowired
    public ConfigService(List<ConfigDocumentParser> parsers) {
        if (parsers.isEmpty()) {
            throw new IllegalStateException("No ConfigDocumentParser beans registered");
        }
        parsers.forEach(ConfigDocumentParserResolver::register);
        FlexlbConfig defaults = new FlexlbConfig();
        defaults.getRequestLifecycle().getRequest().setTimeoutMs(300_000L);
        this.currentFlexlbConfig = new AtomicReference<>(defaults);
        this.currentModelServiceConfig = new AtomicReference<>();
        initializeConfigSources();
        this.startupScheduler = StartupScheduler.from(currentFlexlbConfig.get());
        logEffectiveConfig(currentFlexlbConfig.get(), configSchemaVersion);
    }

    ConfigService(Map<String, String> environment) {
        this(requireEnvironmentDocument(environment));
    }

    public static synchronized void register(ConfigSource source) {
        if (!CONFIG_SOURCES.contains(source)) {
            CONFIG_SOURCES.add(source);
            CONFIG_SOURCES.sort(Comparator.comparingInt(ConfigSource::priority));
        }
    }

    public FlexlbConfig loadBalanceConfig() {
        return currentFlexlbConfig.get();
    }

    /**
     * DSV4 compatibility alias retained for existing integrations.
     */
    public ServiceRoute loadModelServiceConfig() {
        return modelServiceConfig();
    }

    public ServiceRoute modelServiceConfig() {
        return currentModelServiceConfig.get();
    }

    public void addUpdateListener(Consumer<FlexlbConfig> listener) {
        synchronized (updateLock) {
            updateListeners.add(listener);
            listener.accept(currentFlexlbConfig.get());
        }
    }

    public void updateTrafficPolicy(TrafficPolicyConfig groupSelector) {
        if (groupSelector == null) {
            throw new IllegalArgumentException("groupSelector cannot be null");
        }
        TrafficPolicyConfig.validate(groupSelector);
        synchronized (updateLock) {
            try {
                ObjectNode document = JsonUtils.strictValueToTree(currentFlexlbConfig.get());
                ObjectNode router = (ObjectNode) document.get("router");
                router.set("groupSelector", JsonUtils.strictValueToTree(groupSelector));
                FlexlbConfig updated = parse(JsonUtils.toStrictString(document));
                currentFlexlbConfig.set(updated);
                notifyUpdateListeners(updated);
                log.info("Group selector updated: rules={}", groupSelector.getRules().size());
            } catch (ConfigValidationException error) {
                throw error;
            } catch (Exception error) {
                throw new IllegalStateException("Failed to update group selector", error);
            }
        }
    }

    public static FlexlbConfig parse(String document) {
        try {
            JsonNode tree = JsonUtils.readStrictTree(document);
            org.flexlb.config.loading.ConfigJsonValidation.rejectJsonNull(
                    tree, "$", FLEXLB_CONFIG_ENV);
            FlexlbConfigValidator.validateDocumentShape(tree);
            FlexlbConfig config = JsonUtils.strictTreeToValue(tree, FlexlbConfig.class);
            FlexlbConfigValidator.validate(config);
            return config;
        } catch (ConfigValidationException error) {
            throw error;
        } catch (Exception error) {
            throw new ConfigValidationException(
                    FLEXLB_CONFIG_ENV, "Invalid FLEXLB_CONFIG JSON: " + error.getMessage(), error);
        }
    }

    @PreDestroy
    public void close() {
        closeConfigSources();
    }

    private static String requireEnvironmentDocument(Map<String, String> environment) {
        List<String> removed =
                environment.keySet().stream()
                        .filter(
                                name ->
                                        name.equals("CACHE_STATUS_MAX_INTERVAL_MS")
                                                || name.equals("DEFAULT_SCHEDULE_MODE")
                                                || name.equals("FLEXLB_MONITOR_MODE")
                                                || name.equals("FLEXLB_MONITOR_METRIC_WHITELIST"))
                        .sorted()
                        .toList();
        if (!removed.isEmpty()) {
            throw new ConfigValidationException(
                    "environment",
                    "Removed legacy FlexLB environment variables are no longer read: "
                            + String.join(", ", removed)
                            + ". Migrate scheduling behavior into FLEXLB_CONFIG with schemaVersion 4."
                            + " Configure metric exposure with --flexlb.monitor.metric-whitelist.");
        }
        return environment.get(FLEXLB_CONFIG_ENV);
    }

    private void initializeConfigSources() {
        try {
            synchronized (updateLock) {
                for (ConfigSource source : CONFIG_SOURCES) {
                    NormalizedConfig normalized;
                    try {
                        source.setUpdateListener(content -> receiveConfigUpdate(source, content));
                        normalized = source.loadConfig();
                    } catch (Exception error) {
                        throw new IllegalStateException(
                                "Failed to initialize FlexLB configuration from " + source.name(),
                                error);
                    }
                    initializeConfigSource(source, normalized);
                }
            }
        } catch (RuntimeException error) {
            closeConfigSources();
            throw error;
        }
    }

    private void receiveConfigUpdate(ConfigSource source, String content) {
        synchronized (updateLock) {
            try {
                FlexlbConfig previous = currentFlexlbConfig.get();
                NormalizedConfig normalized = source.normalize(content);
                String activeDocument = preserveStartupScheduler(normalized.flexlbConfig());
                FlexlbConfig updated =
                        FlexlbConfigMerger.merge(previous, activeDocument, source.name());
                if (updated == previous) {
                    log.info(
                            "Ignored empty FlexLB configuration update from {} source",
                            source.name());
                    return;
                }
                currentFlexlbConfig.set(updated);
                configSchemaVersion = normalized.sourceSchemaVersion();
                notifyUpdateListeners(updated);
                logEffectiveConfig(updated, configSchemaVersion);
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception error) {
                log.error(
                        "Rejected invalid FlexLB configuration update from {} source; "
                                + "keeping last-known-good configuration: {}",
                        source.name(),
                        error.getMessage());
            }
        }
    }

    /**
     * Keeps the startup scheduler type while applying numeric configuration updates before normal
     * merge validation runs.
     *
     * <p>These modes select threads and strategy beans at startup, so a
     * type update cannot take effect safely. Numeric settings continue through hot updates.
     */
    private String preserveStartupScheduler(String content) throws Exception {
        if (startupScheduler == null || content == null || content.isBlank()) {
            return content;
        }
        JsonNode parsed = JsonUtils.readStrictTree(content);
        if (!(parsed instanceof ObjectNode document)) {
            return content;
        }
        boolean changed = false;
        if (document.get("scheduler") instanceof ObjectNode scheduler) {
            changed |=
                    preserveStartupMode(
                            scheduler,
                            "scheduler.type",
                            startupScheduler.schedulerType().name(),
                            Set.of("DIRECT", "QUEUE"));
        }
        return changed ? JsonUtils.toStrictString(document) : content;
    }

    private boolean preserveStartupMode(
            ObjectNode settings, String path, String activeType, Set<String> supportedTypes) {
        JsonNode type = settings.get("type");
        if (type == null
                || !type.isTextual()
                || !supportedTypes.contains(type.textValue())
                || activeType.equals(type.textValue())) {
            return false;
        }
        settings.remove("type");
        log.warn(
                "Ignored runtime update to startup-fixed {}: requested={}, active={}",
                path,
                type.textValue(),
                activeType);
        return true;
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

    private static void logEffectiveConfig(FlexlbConfig config, int configSchemaVersion) {
        String scheduler = config.isDirect() ? "DIRECT" : "QUEUE";
        String ordering =
                config.isDirect() ? "N/A" : config.isPriorityOrdering() ? "PRIORITY" : "FIFO";
        log.info(
                "FlexLB config loaded: schemaVersion={}, scheduler={}, ordering={}, groupRules={}",
                configSchemaVersion,
                scheduler,
                ordering,
                config.getRouter().getGroupSelector() == null
                        ? 0
                        : config.getRouter().getGroupSelector().getRules().size());
    }

    private void initializeConfigSource(ConfigSource source, NormalizedConfig normalized) {
        try {
            FlexlbConfig previous = currentFlexlbConfig.get();
            FlexlbConfig updated =
                    FlexlbConfigMerger.merge(previous, normalized.flexlbConfig(), source.name());
            currentFlexlbConfig.set(updated);
            if (updated != previous) {
                configSchemaVersion = normalized.sourceSchemaVersion();
            }
        } catch (Exception error) {
            throw new IllegalStateException(
                    "Failed to initialize FlexLB configuration from " + source.name(), error);
        }
        String modelServiceDocument = normalized.modelServiceConfig();
        if (currentModelServiceConfig.get() == null
                && modelServiceDocument != null
                && !modelServiceDocument.isBlank()) {
            currentModelServiceConfig.set(ModelServiceConfigParser.parse(modelServiceDocument));
            log.info("Loaded MODEL_SERVICE_CONFIG from {} source", source.name());
        }
        log.info("Loaded FlexLB configuration from {} source", source.name());
    }

    private static synchronized void closeConfigSources() {
        for (ConfigSource source : CONFIG_SOURCES) {
            try {
                source.close();
            } catch (Exception error) {
                log.warn("Failed to close {} configuration source", source.name(), error);
            }
        }
        CONFIG_SOURCES.clear();
    }

    /**
     * Freezes the scheduling topology that must remain consistent with the startup-created
     * components.
     */
    private record StartupScheduler(SchedulerConfig.Type schedulerType) {

        /**
         * Captures the scheduling modes selected when the application starts.
         */
        private static StartupScheduler from(FlexlbConfig config) {
            SchedulerConfig scheduler = config.getScheduler();
            return new StartupScheduler(scheduler.getType());
        }
    }
}
