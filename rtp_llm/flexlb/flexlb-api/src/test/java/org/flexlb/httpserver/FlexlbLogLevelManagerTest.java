package org.flexlb.httpserver;

import org.flexlb.enums.LogLevel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.logging.LoggerGroups;
import org.springframework.boot.logging.LoggingSystem;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class FlexlbLogLevelManagerTest {

    @Test
    void updatesEveryLoggerInFlexlbGroup() {
        LoggingSystem loggingSystem = mock(LoggingSystem.class);
        List<String> members = List.of(
                "org.flexlb", "flexlbLogger", "syncLogger", "syncConsistencyLogger");
        LoggerGroups loggerGroups = new LoggerGroups(Map.of("flexlb", members));
        FlexlbLogLevelManager manager = new FlexlbLogLevelManager(loggingSystem, loggerGroups);

        assertEquals(LogLevel.DEBUG, manager.setLogLevel(LogLevel.DEBUG));

        for (String member : members) {
            verify(loggingSystem).setLogLevel(member, org.springframework.boot.logging.LogLevel.DEBUG);
        }
    }
}
