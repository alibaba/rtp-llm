package org.flexlb.util;

import ch.qos.logback.classic.Level;
import org.flexlb.enums.LogLevel;
import org.slf4j.LoggerFactory;

/**
 * Logging utility wrapping SLF4J with runtime log-level control via logback.
 *
 * <p>All filtering is delegated to logback — there is no custom gating.
 * {@link #setLevel(LogLevel)} directly updates the {@code flexlbLogger} logback logger,
 * so the {@code update_log_level} API and the logback configuration stay in sync.</p>
 *
 * <p>{@code INFO}, {@code WARN} and {@code ERROR} are enabled by default (logback
 * {@code flexlbLogger} starts at {@code INFO}).</p>
 */
public class Logger {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger("flexlbLogger");

    static {
        String logLevelStr = System.getenv("LOG_LEVEL");
        if (logLevelStr != null) {
            try {
                setLevel(LogLevel.valueOf(logLevelStr.toUpperCase().trim()));
            } catch (IllegalArgumentException e) {
                log.warn("Invalid LOG_LEVEL value: '{}'. Valid values are: TRACE, DEBUG, INFO, WARN, ERROR.", logLevelStr);
            }
        }
    }

    // ---- Logging methods (delegate directly to SLF4J) ----

    public static void trace(String format, Object... args) {
        log.trace(format, args);
    }

    public static void debug(String format, Object... args) {
        log.debug(format, args);
    }

    public static void info(String format, Object... args) {
        log.info(format, args);
    }

    public static void warn(String format, Object... args) {
        log.warn(format, args);
    }

    public static void error(String format, Object... args) {
        log.error(format, args);
    }

    // ---- Runtime level control ----

    /**
     * Returns the effective log level of the underlying logback logger.
     */
    public static LogLevel getLevel() {
        ch.qos.logback.classic.Logger lbLogger = logbackLogger();
        Level level = lbLogger.getLevel();
        if (level == null) {
            return null;
        }
        try {
            return LogLevel.valueOf(level.levelStr.toUpperCase());
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    /**
     * Sets the log level of the underlying {@code flexlbLogger}.
     * A {@code null} level resets the logger to {@code INFO} (the safe production default).
     */
    public static void setLevel(LogLevel level) {
        Level lbLevel = level != null
                ? Level.toLevel(level.name(), Level.INFO)
                : Level.INFO;
        logbackLogger().setLevel(lbLevel);
    }

    private static ch.qos.logback.classic.Logger logbackLogger() {
        return (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
    }
}
