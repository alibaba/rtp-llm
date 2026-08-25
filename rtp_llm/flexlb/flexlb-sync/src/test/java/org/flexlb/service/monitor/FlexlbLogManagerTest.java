package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.logging.LoggerGroups;
import org.springframework.boot.logging.LoggingSystem;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class FlexlbLogManagerTest {

    @Test
    void updatesEveryLoggerInFlexlbGroup() {
        LoggingSystem loggingSystem = mock(LoggingSystem.class);
        List<String> members = List.of("org.flexlb", "flexlbLogger", "syncLogger", "syncConsistencyLogger");
        LoggerGroups loggerGroups = new LoggerGroups(Map.of("flexlb", members));
        ConfigService configService = mock(ConfigService.class);
        LogbackStdoutController stdoutController = mock(LogbackStdoutController.class);
        FlexlbLogManager manager = new FlexlbLogManager(
                loggingSystem, loggerGroups, configService, stdoutController);

        assertEquals(LogLevel.DEBUG, manager.setLogLevel(LogLevel.DEBUG));

        for (String member : members) {
            verify(loggingSystem).setLogLevel(member, org.springframework.boot.logging.LogLevel.DEBUG);
        }
    }

    @Test
    void appliesConfigUpdates() {
        LoggingSystem loggingSystem = mock(LoggingSystem.class);
        List<String> members = List.of("org.flexlb", "flexlbLogger");
        LoggerGroups loggerGroups = new LoggerGroups(Map.of("flexlb", members));
        ConfigService configService = mock(ConfigService.class);
        LogbackStdoutController stdoutController = mock(LogbackStdoutController.class);
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getLogging().setLevel(LogLevel.WARN);
        config.getObservability().getLogging().setStdoutEnabled(true);
        doAnswer(invocation -> {
            Consumer<FlexlbConfig> listener = invocation.getArgument(0);
            listener.accept(config);
            return null;
        }).when(configService).addUpdateListener(any());

        new FlexlbLogManager(loggingSystem, loggerGroups, configService, stdoutController);

        for (String member : members) {
            verify(loggingSystem).setLogLevel(member, org.springframework.boot.logging.LogLevel.WARN);
        }
        verify(stdoutController).setEnabled(true);
    }
}
