package org.flexlb.service.monitor;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.Appender;
import org.slf4j.LoggerFactory;

import java.util.Objects;

final class LogbackStdoutController {

    private static final String CONSOLE_APPENDER = "CONSOLE-async";
    private static final String PV_LOGGER = "pvLogger";

    private final Logger rootLogger;
    private final Logger pvLogger;
    private final Appender<ILoggingEvent> consoleAppender;

    LogbackStdoutController() {
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
        this.rootLogger = loggerContext.getLogger(Logger.ROOT_LOGGER_NAME);
        this.pvLogger = loggerContext.getLogger(PV_LOGGER);
        this.consoleAppender = requireConsoleAppender(rootLogger);
    }

    LogbackStdoutController(Logger rootLogger, Logger pvLogger, Appender<ILoggingEvent> consoleAppender) {
        this.rootLogger = rootLogger;
        this.pvLogger = pvLogger;
        this.consoleAppender = consoleAppender;
    }

    synchronized void setEnabled(boolean enabled) {
        if (enabled) {
            rootLogger.addAppender(consoleAppender);
            pvLogger.addAppender(consoleAppender);
        } else {
            rootLogger.detachAppender(consoleAppender);
            pvLogger.detachAppender(consoleAppender);
        }
    }

    private Appender<ILoggingEvent> requireConsoleAppender(Logger rootLogger) {
        return Objects.requireNonNull(
                rootLogger.getAppender(CONSOLE_APPENDER), "Logback appender 'CONSOLE-async' is not configured");
    }
}
