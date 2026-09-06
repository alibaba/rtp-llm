package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.flexlb.service.config.parser.ConfigDocumentParser;
import org.flexlb.service.config.parser.ModelServiceConfigParser;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
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

    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();

    private final AtomicReference<FlexlbConfig> currentFlexlbConfig;
    private final AtomicReference<ServiceRoute> currentModelServiceConfig;
    private final List<Consumer<FlexlbConfig>> updateListeners = new ArrayList<>();
    private final Object updateLock = new Object();
    private int configSchemaVersion = ConfigSchemaVersion.V0_COMPATIBILITY;

    /** Compatibility constructor for direct construction in tests and embedders. */
    public ConfigService() {
        this(List.of(
                new StandardConfigDocumentParser(),
                new V0ConfigDocumentParser()));
    }

    @Autowired
    public ConfigService(List<ConfigDocumentParser> parsers) {
        if (parsers.isEmpty()) {
            throw new IllegalStateException("No ConfigDocumentParser beans registered");
        }
        this.currentFlexlbConfig = new AtomicReference<>(new FlexlbConfig());
        this.currentModelServiceConfig = new AtomicReference<>();
        initializeConfigSources();
        logEffectiveConfig(currentFlexlbConfig.get(), configSchemaVersion);
    }

    public static synchronized void register(ConfigSource source) {
        if (!CONFIG_SOURCES.contains(source)) {
            CONFIG_SOURCES.add(source);
            CONFIG_SOURCES.sort(Comparator.comparingInt(ConfigSource::priority));
        }
    }

    private static synchronized List<ConfigSource> registeredSources() {
        List<ConfigSource> sources = new ArrayList<>(CONFIG_SOURCES);
        sources.sort(Comparator.comparingInt(ConfigSource::priority));
        return sources;
    }

    public static FlexlbConfig parse(String document) {
        try {
            JsonNode tree = JsonUtils.readStrictTree(document);
            JsonUtils.rejectJsonNull(tree, "$", FLEXLB_CONFIG_ENV);
            FlexlbConfigValidator.validateDocumentShape(tree);
            FlexlbConfig config = JsonUtils.strictTreeToValue(
                    tree, FlexlbConfig.class);
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
            return JsonUtils.toStrictString(config);
        } catch (Exception error) {
            throw new IllegalStateException("Failed to serialize FlexLB configuration", error);
        }
    }

    public FlexlbConfig loadBalanceConfig() {
        return currentFlexlbConfig.get();
    }

    public ServiceRoute modelServiceConfig() {
        return currentModelServiceConfig.get();
    }

    static ServiceRoute parseModelServiceConfig(String document) {
        return ModelServiceConfigParser.parse(document);
    }

    /** DSV4 compatibility alias retained for existing integrations. */
    public ServiceRoute loadModelServiceConfig() {
        return modelServiceConfig();
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
                ObjectNode document = JsonUtils.strictValueToTree(
                        currentFlexlbConfig.get());
                ObjectNode router = (ObjectNode) document.get("router");
                router.set("groupSelector", JsonUtils.strictValueToTree(groupSelector));
                FlexlbConfig updated = parse(JsonUtils.toStrictString(document));
                currentFlexlbConfig.set(updated);
                notifyUpdateListeners(updated);
                log.info("Group selector updated: rules={}",
                        groupSelector.getRules().size());
            } catch (ConfigValidationException error) {
                throw error;
            } catch (Exception error) {
                throw new IllegalStateException(
                        "Failed to update group selector", error);
            }
        }
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
                        throw new IllegalStateException("Failed to initialize FlexLB configuration from " + source.name(), error);
                    }
                    initializeConfigSource(source, normalized);
                }
            }
        } catch (RuntimeException error) {
            closeConfigSources();
            throw error;
        }
    }

    private void initializeConfigSource(ConfigSource source, NormalizedConfig normalized) {
        try {
            FlexlbConfig previous = currentFlexlbConfig.get();
            FlexlbConfig updated = FlexlbConfigMerger.merge(previous, normalized.flexlbConfig(), source.name());
            currentFlexlbConfig.set(updated);
            if (updated != previous) {
                configSchemaVersion = normalized.sourceSchemaVersion();
            }
        } catch (Exception error) {
            throw new IllegalStateException("Failed to initialize FlexLB configuration from " + source.name(), error);
        }
        String modelServiceDocument = normalized.modelServiceConfig();
        if (currentModelServiceConfig.get() == null && modelServiceDocument != null && !modelServiceDocument.isBlank()) {
            currentModelServiceConfig.set(ModelServiceConfigParser.parse(modelServiceDocument));
            log.info("Loaded MODEL_SERVICE_CONFIG from {} source", source.name());
        }
        log.info("Loaded FlexLB configuration from {} source", source.name());
    }

    private void receiveConfigUpdate(ConfigSource source, String content) {
        synchronized (updateLock) {
            try {
                FlexlbConfig previous = currentFlexlbConfig.get();
                NormalizedConfig normalized = source.normalize(content);
                FlexlbConfig updated = FlexlbConfigMerger.merge(previous, normalized.flexlbConfig(), source.name());
                if (updated == previous) {
                    log.info("Ignored empty FlexLB configuration update from {} source",
                            source.name());
                    return;
                }
                currentFlexlbConfig.set(updated);
                configSchemaVersion = normalized.sourceSchemaVersion();
                notifyUpdateListeners(updated);
                logEffectiveConfig(updated, configSchemaVersion);
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception error) {
                log.error("Rejected invalid FlexLB configuration update from {} source; "
                                + "keeping last-known-good configuration: {}",
                        source.name(), error.getMessage());
            }
        }
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
        String ordering = config.isDirect() ? "N/A"
                : config.isPriorityOrdering() ? "PRIORITY" : "FIFO";
        String decision = config.isDirect() ? "N/A"
                : config.isFixedWindowDecision() ? "FIXED_WINDOW" : "SINGLE";
        String dispatcher = config.getDispatcher().typeName();
        log.info("FlexLB config loaded: schemaVersion={}, scheduler={}, ordering={}, decision={}, "
                        + "dispatcher={}, prefillCandidateChoice={}, groupRules={}",
                configSchemaVersion, scheduler, ordering, decision, dispatcher,
                config.getRouter().getRoles().getPrefill()
                        .getCandidateChoice().getType(),
                config.getRouter().getGroupSelector() == null ? 0
                        : config.getRouter().getGroupSelector().getRules().size());
    }

    @PreDestroy
    public void close() {
        closeConfigSources();
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
}
