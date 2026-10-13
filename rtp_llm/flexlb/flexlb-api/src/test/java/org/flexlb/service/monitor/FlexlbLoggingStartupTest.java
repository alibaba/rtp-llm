package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.Banner;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.logging.LoggerGroups;
import org.springframework.boot.logging.LoggingSystem;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class FlexlbLoggingStartupTest {

    @Test
    void startsLogManagerWithPackagedApplicationConfiguration(@TempDir Path logDirectory) {
        try (var context = new SpringApplicationBuilder(LoggingConfiguration.class)
                .web(WebApplicationType.NONE)
                .bannerMode(Banner.Mode.OFF)
                .logStartupInfo(false)
                .run("--spring.config.location=classpath:/application.yml", "--spring.main.web-application-type=none",
                        "--flexlb.log.path=" + logDirectory, "--flexlb.log.app-path=" + logDirectory)) {
            context.getBean(FlexlbLogManager.class).setLogLevel(LogLevel.DEBUG);

            LoggingSystem loggingSystem = context.getBean(LoggingSystem.class);
            var groupMembers = context.getBean(LoggerGroups.class)
                    .get(FlexlbLogManager.LOG_GROUP_NAME).getMembers();
            assertTrue(!groupMembers.isEmpty());
            for (String loggerName : groupMembers) {
                assertEquals(org.springframework.boot.logging.LogLevel.DEBUG,
                        loggingSystem.getLoggerConfiguration(loggerName).getConfiguredLevel());
            }
            assertTrue(Files.exists(logDirectory.resolve("application.log")));
            assertTrue(Files.exists(logDirectory.resolve("pv.log")));
            assertTrue(Files.exists(logDirectory.resolve("flexlb.log")));
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
