package org.flexlb.httpserver;

import org.flexlb.enums.LogLevel;
import org.springframework.boot.logging.LoggerGroup;
import org.springframework.boot.logging.LoggerGroups;
import org.springframework.boot.logging.LoggingSystem;
import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
public class FlexlbLogLevelManager {

    static final String LOG_GROUP_NAME = "flexlb";

    private final LoggingSystem loggingSystem;
    private final LoggerGroup loggerGroup;

    public FlexlbLogLevelManager(LoggingSystem loggingSystem, LoggerGroups loggerGroups) {
        this.loggingSystem = loggingSystem;
        this.loggerGroup = Objects.requireNonNull(
                loggerGroups.get(LOG_GROUP_NAME), "Logging group 'flexlb' is not configured");
    }

    public LogLevel setLogLevel(LogLevel logLevel) {
        Objects.requireNonNull(logLevel, "log_level must not be null");
        org.springframework.boot.logging.LogLevel springLogLevel =
                org.springframework.boot.logging.LogLevel.valueOf(logLevel.name());
        loggerGroup.configureLogLevel(springLogLevel, loggingSystem::setLogLevel);
        return logLevel;
    }
}
