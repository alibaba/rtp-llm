package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.Banner;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.logging.LoggingSystem;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

class FlexlbLoggingStartupTest {

    @Test
    void startsLogManagerWithPackagedApplicationConfiguration() {
        try (var context = new SpringApplicationBuilder(LoggingConfiguration.class)
                .web(WebApplicationType.NONE)
                .bannerMode(Banner.Mode.OFF)
                .logStartupInfo(false)
                .run("--spring.config.location=classpath:/application.yml", "--spring.main.web-application-type=none")) {
            context.getBean(FlexlbLogManager.class).setLogLevel(LogLevel.DEBUG);

            LoggingSystem loggingSystem = context.getBean(LoggingSystem.class);
            for (String loggerName : List.of("org.flexlb", "flexlbLogger", "syncLogger", "syncConsistencyLogger")) {
                assertEquals(org.springframework.boot.logging.LogLevel.DEBUG,
                        loggingSystem.getLoggerConfiguration(loggerName).getConfiguredLevel());
            }
        }
    }

    @Configuration(proxyBeanMethods = false)
    @Import(FlexlbLogManager.class)
    static class LoggingConfiguration {

        @Bean
        ConfigService configService() {
            return mock(ConfigService.class);
        }
    }
}
