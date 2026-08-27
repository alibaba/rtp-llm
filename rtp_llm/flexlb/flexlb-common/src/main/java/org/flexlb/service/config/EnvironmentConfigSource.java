package org.flexlb.service.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.function.Consumer;

@Slf4j
@Component
final class EnvironmentConfigSource implements ConfigSource {

    private static final int PRIORITY = 1;

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
        configContent = ConfigService.serialize(config);
        ConfigService.register(this);
    }

    @Override
    public void setUpdateListener(Consumer<String> listener) {}

    @Override
    public String load() {
        return configContent;
    }

}
