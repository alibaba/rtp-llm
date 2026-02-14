package org.flexlb.util;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LogLevel;
import org.slf4j.LoggerFactory;

/**
 * Logging utility class, in order to log when enable global switch or set log level in master request
 *
 * <p>The {@code info} {@code warn} and {@code error} level in enabled by default.</p>
 *
 * @see LogLevel
 */
public class Logger {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger("businessLogger");

    @Getter
    @Setter
    private static LogLevel globalLogLevel;

    static {
        String logLevelStr = System.getenv("LOG_LEVEL");
        if (logLevelStr != null) {
            try {
                globalLogLevel = LogLevel.valueOf(logLevelStr.toUpperCase().trim());
            } catch (IllegalArgumentException e) {
                log.warn("Invalid LOG_LEVEL value: '{}'. Valid values are: TRACE, DEBUG, INFO, WARN, ERROR.", logLevelStr);
            }
        }
    }

    public static void trace(String format, Object... args) {
        log(LogLevel.TRACE, () -> log.trace(format, args));
    }

    public static void debug(String format, Object... args) {
        log(LogLevel.DEBUG, () -> log.debug(format, args));
    }

    public static void info(String format, Object... args) {
        log(LogLevel.INFO, () -> log.info(format, args), false);
    }

    public static void warn(String format, Object... args) {
        log(LogLevel.WARN, () -> log.warn(format, args), false);
    }

    public static void error(String format, Object... args) {
        log(LogLevel.ERROR, () -> log.error(format, args), false);
    }

    private static void log(LogLevel targetLevel, Runnable logAction) {
        log(targetLevel, logAction, true);
    }

    private static void log(LogLevel targetLevel, Runnable logAction, boolean checkGlobalLevel) {
        boolean shouldLog = shouldLog(targetLevel, checkGlobalLevel);

        if (shouldLog) {
            logAction.run();
        }
    }

    private static boolean shouldLog(LogLevel targetLevel, boolean checkGlobalLevel) {
        boolean shouldLog;
        if (checkGlobalLevel) {
            shouldLog = globalLogLevel != null && globalLogLevel.compareTo(targetLevel) <= 0;
        } else {
            // warn and error are enabled by default
            shouldLog = globalLogLevel == null || globalLogLevel.compareTo(targetLevel) <= 0;
        }
        return shouldLog;
    }
}
