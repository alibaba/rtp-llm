package org.flexlb.service.config.source;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigSchemaVersion;
import org.flexlb.config.ConfigService;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.function.Consumer;

@Slf4j
@Component
final class EnvironmentConfigSource implements ConfigSource {

    private static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    private static final String MODEL_SERVICE_CONFIG_ENV = "MODEL_SERVICE_CONFIG";
    private static final int PRIORITY = 1;

    private String configContent;
    private String modelServiceConfigContent;

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
        ConfigSourceSelection selection = ConfigSourceSelection.fromEnvironment();
        configContent = selection == ConfigSourceSelection.ENVIRONMENT
                ? System.getenv(FLEXLB_CONFIG_ENV) : null;
        log.info("Selected FlexLB configuration source: {}; environment document configured={}",
                selection, configContent != null);
        // Model topology remains an independent startup document for every behavior source.
        modelServiceConfigContent = System.getenv(MODEL_SERVICE_CONFIG_ENV);
        ConfigService.register(this);
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {}

    @Override
    public String load() {
        if (configContent != null && configContent.isBlank()) {
            throw new IllegalArgumentException(FLEXLB_CONFIG_ENV + " must not be blank when configured");
        }
        return configContent;
    }

    @Override
    public NormalizedConfig loadConfig() {
        String rawFlexlbConfig = load();
        return rawFlexlbConfig == null
                ? new NormalizedConfig(null, modelServiceConfigContent, ConfigSchemaVersion.V0_COMPATIBILITY)
                : normalize(rawFlexlbConfig);
    }

    @Override
    public String loadModelServiceConfig() {
        return modelServiceConfigContent;
    }

}
