package org.flexlb.service.monitor;

import org.flexlb.config.ConfigService;
import org.flexlb.enums.LogLevel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.logging.LoggerGroup;
import org.springframework.boot.logging.LoggerGroups;
import org.springframework.boot.logging.LoggingSystem;
import org.springframework.stereotype.Service;

import java.util.Objects;

@Service
public class FlexlbLogManager {

    static final String LOG_GROUP_NAME = "flexlb";

    private final LoggingSystem loggingSystem;
    private final LoggerGroup loggerGroup;
    private final LogbackStdoutController stdoutController;

    @Autowired
    public FlexlbLogManager(LoggingSystem loggingSystem, LoggerGroups loggerGroups, ConfigService configService) {
        this(loggingSystem, loggerGroups, configService, new LogbackStdoutController());
    }

    FlexlbLogManager(LoggingSystem loggingSystem, LoggerGroups loggerGroups, ConfigService configService,
                     LogbackStdoutController stdoutController) {
        this.loggingSystem = loggingSystem;
        this.loggerGroup = Objects.requireNonNull(
                loggerGroups.get(LOG_GROUP_NAME), "Logging group 'flexlb' is not configured");
        this.stdoutController = stdoutController;
        configService.addUpdateListener(config -> {
            setLogLevel(config.getFlexlbLogLevel());
            setStdoutLogEnabled(config.isEnableStdoutLog());
        });
    }

    public LogLevel setLogLevel(LogLevel logLevel) {
        Objects.requireNonNull(logLevel, "log_level must not be null");
        org.springframework.boot.logging.LogLevel springLogLevel =
                org.springframework.boot.logging.LogLevel.valueOf(logLevel.name());
        loggerGroup.configureLogLevel(springLogLevel, loggingSystem::setLogLevel);
        return logLevel;
    }

    public boolean setStdoutLogEnabled(boolean enabled) {
        stdoutController.setEnabled(enabled);
        return enabled;
    }
}
