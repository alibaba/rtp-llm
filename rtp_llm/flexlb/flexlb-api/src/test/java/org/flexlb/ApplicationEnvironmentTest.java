package org.flexlb;

import org.flexlb.config.MonitorDisableConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.Banner;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.web.ServerProperties;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.core.SpringProperties;
import org.springframework.core.env.AbstractEnvironment;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

@ExtendWith(SystemStubsExtension.class)
class ApplicationEnvironmentTest {

    @SystemStub
    private EnvironmentVariables environment = new EnvironmentVariables(
            "FLEXLB_MONITOR_PROVIDER", "kmonitor", "SERVER_PORT", "19001");

    private String originalIgnoreGetenv;

    @BeforeEach
    void enableEnvironmentVariables() {
        originalIgnoreGetenv = SpringProperties.getProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME);
        SpringProperties.setProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME, null);
    }

    @AfterEach
    void restoreEnvironmentVariableSetting() {
        SpringProperties.setProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME, originalIgnoreGetenv);
    }

    @Test
    void startupReadsEnvironmentVariablesAndKeepsMonitoringEnabled(@TempDir Path logDirectory) {
        try (var context = startContext(logDirectory)) {
            assertEquals("kmonitor", context.getEnvironment().getProperty("flexlb.monitor.provider"));
            assertEquals("19001", context.getEnvironment().getProperty("server.port"));
            assertEquals(19001, context.getBean(ServerProperties.class).getPort());
            assertFalse(context.containsBean("noOpFlexMonitor"));
        }
    }

    @Test
    void commandLinePortOverridesEnvironmentVariable(@TempDir Path logDirectory) {
        try (var context = startContext(logDirectory, "--server.port=19002")) {
            assertEquals("19002", context.getEnvironment().getProperty("server.port"));
            assertEquals(19002, context.getBean(ServerProperties.class).getPort());
        }
    }

    private ConfigurableApplicationContext startContext(Path logDirectory, String... extraArguments) {
        List<String> arguments = new ArrayList<>(List.of(
                "--spring.config.location=classpath:/application.yml",
                "--spring.main.web-application-type=none",
                "--flexlb.log.path=" + logDirectory,
                "--flexlb.log.app-path=" + logDirectory));
        arguments.addAll(Arrays.asList(extraArguments));
        return new SpringApplicationBuilder(EnvironmentConfiguration.class)
                .web(WebApplicationType.NONE)
                .bannerMode(Banner.Mode.OFF)
                .logStartupInfo(false)
                .run(arguments.toArray(String[]::new));
    }

    @Configuration(proxyBeanMethods = false)
    @EnableConfigurationProperties(ServerProperties.class)
    @Import(MonitorDisableConfig.class)
    static class EnvironmentConfiguration {
    }
}
