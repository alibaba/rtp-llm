package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.util.JsonUtils;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

@Slf4j
@Component
@DependsOn({"environmentConfigSource", "nacosConfigSource"})
public class ConfigService {

    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();

    private final AtomicReference<FlexlbConfig> currentConfig;
    private final List<Consumer<FlexlbConfig>> updateListeners = new ArrayList<>();
    private final Object updateLock = new Object();

    public ConfigService() {
        this.currentConfig = new AtomicReference<>(new FlexlbConfig());
        CONFIG_SOURCES.sort(Comparator.comparingInt(ConfigSource::priority));
        initializeConfigSources();
    }

    public static synchronized void register(ConfigSource source) {
        CONFIG_SOURCES.add(source);
    }

    public FlexlbConfig loadBalanceConfig() {
        return currentConfig.get();
    }

    public void addUpdateListener(Consumer<FlexlbConfig> listener) {
        synchronized (updateLock) {
            updateListeners.add(listener);
            listener.accept(currentConfig.get());
        }
    }

    private void initializeConfigSources() {
        ConfigSource loadingSource = null;
        try {
            synchronized (updateLock) {
                for (ConfigSource source : CONFIG_SOURCES) {
                    loadingSource = source;
                    source.setUpdateListener(content -> receiveConfigUpdate(source, content));
                    String initialContent = source.load();
                    currentConfig.set(mergeConfig(currentConfig.get(), initialContent, source.name()));
                    loadingSource = null;
                    log.info("Loaded FlexLB configuration from {} source", source.name());
                }
            }
        } catch (Exception e) {
            closeConfigSources();
            throw new IllegalStateException("Failed to initialize FlexLB configuration from "
                            + (loadingSource == null ? "configured source" : loadingSource.name()), e);
        }
    }

    private void receiveConfigUpdate(ConfigSource source, String content) {
        synchronized (updateLock) {
            try {
                FlexlbConfig newConfig = mergeConfig(currentConfig.get(), content, source.name());
                currentConfig.set(newConfig);
                for (Consumer<FlexlbConfig> listener : updateListeners) {
                    listener.accept(newConfig);
                }
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception e) {
                log.error(
                        "Rejected invalid FlexLB configuration update from {} source; keeping last-known-good configuration: {}",
                        source.name(),
                        e.getMessage());
            }
        }
    }

    private FlexlbConfig mergeConfig(FlexlbConfig baseConfig, String content, String sourceName) {
        if (content == null || content.isBlank()) {
            throw new IllegalArgumentException(sourceName + " configuration must not be blank");
        }

        JsonNode parsed = JsonUtils.toTreeNode(content);
        if (!(parsed instanceof ObjectNode overrides)) {
            throw new IllegalArgumentException(sourceName + " configuration must be a JSON object");
        }
        if (overrides.isEmpty()) {
            throw new IllegalArgumentException(sourceName + " configuration must contain at least one FlexlbConfig field");
        }

        ObjectNode merged = (ObjectNode) JsonUtils.toTreeNode(baseConfig);
        merged.setAll(overrides);
        FlexlbConfig config = JsonUtils.toObject(merged, FlexlbConfig.class);
        log.debug("Resolved FlexLB configuration from {} source", sourceName);
        return config;
    }

    @PreDestroy
    public void close() {
        closeConfigSources();
    }

    private void closeConfigSources() {
        for (ConfigSource source : CONFIG_SOURCES) {
            closeQuietly(source);
        }
        CONFIG_SOURCES.clear();
    }

    private void closeQuietly(ConfigSource source) {
        if (source == null) {
            return;
        }
        try {
            source.close();
        } catch (Exception e) {
            log.warn("Failed to close {} configuration source", source.name(), e);
        }
    }

}
