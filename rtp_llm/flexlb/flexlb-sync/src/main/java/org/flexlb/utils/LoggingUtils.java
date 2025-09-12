package org.flexlb.utils;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.domain.balance.WhaleMasterConfig;
import org.flexlb.enums.LogLevel;

import java.util.function.Supplier;

/**
 * Logging utility class, in order to log when enable global switch or set log level in master request
 *
 * <p>The {@code warn} and {@code error} level in enabled by default.</p>
 *
 * @see WhaleMasterConfig#getLogLevel()
 * @see LogLevel
 */
@Slf4j
public class LoggingUtils {

    public static void trace(String format, Object... args) {
        log(LogLevel.TRACE, () -> log.trace(format, args));
    }

    public static void debug(String format, Object... args) {
        log(LogLevel.DEBUG, () -> log.debug(format, args));
    }

    public static void info(String format, Object... args) {
        log(LogLevel.INFO, () -> log.info(format, args));
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
        LogLevel globalLogLevel = WhaleMasterConfig.getLogLevel();
        boolean shouldLog;
        if (checkGlobalLevel) {
            shouldLog = globalLogLevel != null && globalLogLevel.compareTo(targetLevel) <= 0;
        } else {
            // warn and error are enabled by default
            shouldLog = globalLogLevel == null || globalLogLevel.compareTo(targetLevel) <= 0;
        }

        if (shouldLog) {
            logAction.run();
        }
    }
}
