package org.flexlb;

import org.flexlb.config.MonitorDisableConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.SpringApplication;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.SpringProperties;
import org.springframework.core.env.AbstractEnvironment;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;

@ExtendWith(SystemStubsExtension.class)
class ApplicationEnvironmentTest {

    @SystemStub
    private EnvironmentVariables environment = new EnvironmentVariables(
            "FLEXLB_MONITOR_PROVIDER", "kmonitor", "SERVER_PORT", "19001");

    @Test
    void startupReadsEnvironmentVariablesAndKeepsMonitoringEnabled() {
        String originalIgnoreGetenv = SpringProperties.getProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME);
        SpringProperties.setProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME, null);
        var startedContext = new AtomicReference<AnnotationConfigApplicationContext>();

        try (var telemetry = mockStatic(OpenTelemetryBootstrap.class);
             var springApplication = mockStatic(SpringApplication.class)) {
            springApplication.when(() -> SpringApplication.run(eq(Application.class), any(String[].class)))
                    .thenAnswer(invocation -> {
                        var context = new AnnotationConfigApplicationContext();
                        startedContext.set(context);
                        context.register(MonitorDisableConfig.class);
                        context.refresh();
                        return context;
                    });

            Application.main(new String[0]);

            var context = startedContext.get();
            assertEquals("kmonitor", context.getEnvironment().getProperty("flexlb.monitor.provider"));
            assertEquals("19001", context.getEnvironment().getProperty("server.port"));
            assertFalse(context.containsBean("noOpFlexMonitor"));
        } finally {
            if (startedContext.get() != null) {
                startedContext.get().close();
            }
            SpringProperties.setProperty(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME, originalIgnoreGetenv);
        }
    }
}
