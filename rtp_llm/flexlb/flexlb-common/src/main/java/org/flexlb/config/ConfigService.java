package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.flexlb.service.config.parser.ConfigDocumentParser;
import org.flexlb.service.config.parser.ModelServiceConfigParser;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Loads the strict FLEXLB_CONFIG document and the independent MODEL_SERVICE_CONFIG document.
 * Registered configuration sources may update only the FlexLB behavior snapshot.
 */
@Slf4j
@Component
@DependsOn({"environmentConfigSource", "nacosConfigSource", "uniConfigConfigSource"})
public class ConfigService {

    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();

    private final AtomicReference<FlexlbConfig> currentFlexlbConfig;
    private final AtomicReference<ServiceRoute> currentModelServiceConfig;
    private final List<ConfigUpdateListener> updateListeners = new ArrayList<>();
    private final Object updateLock = new Object();
    private final Object notificationLock = new Object();
    private int configSchemaVersion = ConfigSchemaVersion.V0_COMPATIBILITY;

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

    public FlexlbConfig loadBalanceConfig() {
        return currentFlexlbConfig.get();
    }

    public ServiceRoute modelServiceConfig() {
        return currentModelServiceConfig.get();
    }

    public void addUpdateListener(Consumer<FlexlbConfig> listener) {
        addUpdateListener(Function.identity(), listener);
    }

    /**
     * Replays the current projected setting and applies subsequent changes in update order.
     * Projections must be side-effect free; appliers run outside the snapshot lock.
     */
    public <T> void addUpdateListener(Function<FlexlbConfig, T> projection, Consumer<T> listener) {
        Objects.requireNonNull(projection);
        Objects.requireNonNull(listener);
        synchronized (notificationLock) {
            T initialValue;
            synchronized (updateLock) {
                initialValue = projection.apply(currentFlexlbConfig.get());
            }
            listener.accept(initialValue);
            synchronized (updateLock) {
                updateListeners.add(new ProjectedConfigUpdateListener<>(projection, listener, initialValue));
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
        synchronized (notificationLock) {
            try {
                List<Runnable> notifications;
                FlexlbConfig updated;
                synchronized (updateLock) {
                    FlexlbConfig previous = currentFlexlbConfig.get();
                    NormalizedConfig normalized = source.normalize(content);
                    updated = FlexlbConfigMerger.merge(previous, normalized.flexlbConfig(), source.name());
                    if (updated == previous) {
                        log.info("Ignored empty FlexLB configuration update from {} source", source.name());
                        return;
                    }
                    notifications = updateListeners.stream()
                            .map(listener -> listener.prepareUpdate(updated)).toList();
                    currentFlexlbConfig.set(updated);
                    configSchemaVersion = normalized.sourceSchemaVersion();
                }
                for (Runnable notification : notifications) {
                    try {
                        notification.run();
                    } catch (RuntimeException error) {
                        log.error("FlexLB configuration update listener failed", error);
                    }
                }
                logEffectiveConfig(updated, configSchemaVersion);
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception error) {
                log.error("Rejected invalid FlexLB configuration update from {} source; "
                                + "keeping last-known-good configuration: {}",
                        source.name(), error.getMessage());
            }
        }
    }

    private interface ConfigUpdateListener {
        Runnable prepareUpdate(FlexlbConfig config);
    }

    /** Tracks the last successfully applied value so later updates retry failed appliers. */
    private static final class ProjectedConfigUpdateListener<T> implements ConfigUpdateListener {
        private final Function<FlexlbConfig, T> projection;
        private final Consumer<T> listener;
        private T currentValue;

        private ProjectedConfigUpdateListener(Function<FlexlbConfig, T> projection,
                                                Consumer<T> listener, T initialValue) {
            this.projection = projection;
            this.listener = listener;
            this.currentValue = initialValue;
        }

        @Override
        public Runnable prepareUpdate(FlexlbConfig config) {
            T updatedValue = projection.apply(config);
            if (Objects.equals(currentValue, updatedValue)) {
                return () -> {};
            }
            return () -> {
                listener.accept(updatedValue);
                currentValue = updatedValue;
            };
        }
    }

    private static void logEffectiveConfig(FlexlbConfig config, int configSchemaVersion) {
        String scheduler = config.isDirect() ? "DIRECT" : "QUEUE";
        String ordering = config.isDirect() ? "N/A"
                : config.isPriorityOrdering() ? "PRIORITY" : "FIFO";
        String dispatcher = config.isBatchDispatch() ? "BATCH" : "NON_BATCH";
        log.info("FlexLB config loaded: schemaVersion={}, scheduler={}, ordering={}, dispatcher={}, "
                        + "cacheMatching={}, consistency={}, prefillSelector={}, decodeSelector={}, groupRules={}",
                configSchemaVersion, scheduler, ordering, dispatcher,
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
            try {
                source.close();
            } catch (Exception error) {
                log.warn("Failed to close {} configuration source", source.name(), error);
            }
        }
        CONFIG_SOURCES.clear();
    }
}
