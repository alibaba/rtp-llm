package org.flexlb.service.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.BlockHashStrategyType;
import org.flexlb.enums.LogLevel;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Locale;
import java.util.function.Consumer;

@Slf4j
@Component
final class EnvironmentConfigSource implements ConfigSource {

    private static final int PRIORITY = 1;
    private static final String BLOCK_HASH_STRATEGY_ENV = "BLOCK_HASH_STRATEGY";
    private static final String LOG_LEVEL_ENV = "FLEXLB_LOG_LEVEL";
    private static final String STDOUT_LOG_ENV = "ENABLE_STDOUT_LOG";

    private String configContent;

    @Override
    public String name() {
        return "environment";
    }

    @Override
    public int priority() {
        return PRIORITY;
    }

    @PostConstruct
    void initialize() {
        String document = System.getenv(ConfigService.FLEXLB_CONFIG_ENV);
        log.info("Loading FLEXLB_CONFIG from environment: configured={}", document != null);
        FlexlbConfig config = document == null
                ? new FlexlbConfig()
                : ConfigService.parse(document);
        applyCompatibilityOverrides(config);
        configContent = ConfigService.serialize(config);
        ConfigService.register(this);
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {}

    @Override
    public String load() {
        return configContent;
    }

    private void applyCompatibilityOverrides(FlexlbConfig config) {
        String blockHashStrategy = System.getenv(BLOCK_HASH_STRATEGY_ENV);
        if (blockHashStrategy != null) {
            config.setBlockHashStrategy(BlockHashStrategyType.valueOf(
                    blockHashStrategy.trim().toUpperCase(Locale.ROOT)));
        }

        String logLevel = System.getenv(LOG_LEVEL_ENV);
        if (logLevel != null) {
            config.getObservability().getLogging().setLevel(
                    LogLevel.valueOf(logLevel.trim().toUpperCase(Locale.ROOT)));
        }

        String stdoutEnabled = System.getenv(STDOUT_LOG_ENV);
        if (stdoutEnabled != null) {
            config.getObservability().getLogging().setStdoutEnabled(
                    parseBoolean(STDOUT_LOG_ENV, stdoutEnabled));
        }
    }

    private static boolean parseBoolean(String name, String value) {
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        if ("true".equals(normalized)) {
            return true;
        }
        if ("false".equals(normalized)) {
            return false;
        }
        throw new IllegalArgumentException(name + " must be true or false");
    }
}
