package org.flexlb.util;

import ch.qos.logback.classic.Level;
import org.flexlb.enums.LogLevel;
import org.slf4j.LoggerFactory;

/**
 * Business logger facade. Log filtering is delegated to the configured SLF4J backend.
 */
public final class Logger {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger("flexlbLogger");

    private Logger() {
    }

    public static boolean isDebugEnabled() {
        return log.isDebugEnabled();
    }

    public static void trace(String message) {
        log.trace(message);
    }

    public static void trace(String format, Object argument) {
        log.trace(format, argument);
    }

    public static void trace(String format, Object firstArgument, Object secondArgument) {
        log.trace(format, firstArgument, secondArgument);
    }

    public static void trace(String message, Throwable throwable) {
        log.trace(message, throwable);
    }

    public static void debug(String message) {
        log.debug(message);
    }

    public static void debug(String format, Object argument) {
        log.debug(format, argument);
    }

    public static void debug(String format, Object firstArgument, Object secondArgument) {
        log.debug(format, firstArgument, secondArgument);
    }

    public static void debug(String message, Throwable throwable) {
        log.debug(message, throwable);
    }

    public static void debug(
            String format, Object firstArgument, Object secondArgument, Object thirdArgument) {
        if (log.isDebugEnabled()) {
            log.debug(format, firstArgument, secondArgument, thirdArgument);
        }
    }

    public static void debug(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument) {
        if (log.isDebugEnabled()) {
            log.debug(format, firstArgument, secondArgument, thirdArgument, fourthArgument);
        }
    }

    public static void debug(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument,
            Object fifthArgument) {
        if (log.isDebugEnabled()) {
            log.debug(format, firstArgument, secondArgument, thirdArgument, fourthArgument, fifthArgument);
        }
    }

    public static void debug(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument,
            Object fifthArgument,
            Object sixthArgument) {
        if (log.isDebugEnabled()) {
            log.debug(format,
                    firstArgument,
                    secondArgument,
                    thirdArgument,
                    fourthArgument,
                    fifthArgument,
                    sixthArgument);
        }
    }

    public static void info(String message) {
        log.info(message);
    }

    public static void info(String format, Object argument) {
        log.info(format, argument);
    }

    public static void info(String format, Object firstArgument, Object secondArgument) {
        log.info(format, firstArgument, secondArgument);
    }

    public static void info(String message, Throwable throwable) {
        log.info(message, throwable);
    }

    public static void info(
            String format, Object firstArgument, Object secondArgument, Object thirdArgument) {
        if (log.isInfoEnabled()) {
            log.info(format, firstArgument, secondArgument, thirdArgument);
        }
    }

    public static void warn(String message) {
        log.warn(message);
    }

    public static void warn(String format, Object argument) {
        log.warn(format, argument);
    }

    public static void warn(String format, Object firstArgument, Object secondArgument) {
        log.warn(format, firstArgument, secondArgument);
    }

    public static void warn(String message, Throwable throwable) {
        log.warn(message, throwable);
    }

    public static void warn(
            String format, Object firstArgument, Object secondArgument, Object thirdArgument) {
        if (log.isWarnEnabled()) {
            log.warn(format, firstArgument, secondArgument, thirdArgument);
        }
    }

    public static void warn(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument) {
        if (log.isWarnEnabled()) {
            log.warn(format, firstArgument, secondArgument, thirdArgument, fourthArgument);
        }
    }

    public static void warn(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument,
            Object fifthArgument) {
        if (log.isWarnEnabled()) {
            log.warn(format, firstArgument, secondArgument, thirdArgument, fourthArgument, fifthArgument);
        }
    }

    public static void error(String message) {
        log.error(message);
    }

    public static void error(String format, Object argument) {
        log.error(format, argument);
    }

    public static void error(String format, Object firstArgument, Object secondArgument) {
        log.error(format, firstArgument, secondArgument);
    }

    public static void error(String message, Throwable throwable) {
        log.error(message, throwable);
    }

    public static void error(
            String format, Object firstArgument, Object secondArgument, Object thirdArgument) {
        if (log.isErrorEnabled()) {
            log.error(format, firstArgument, secondArgument, thirdArgument);
        }
    }

    public static void error(
            String format,
            Object firstArgument,
            Object secondArgument,
            Object thirdArgument,
            Object fourthArgument) {
        if (log.isErrorEnabled()) {
            log.error(format, firstArgument, secondArgument, thirdArgument, fourthArgument);
        }
    }
    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12);
        }
    }

    public static void debug(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12, Object argument13) {
        if (log.isDebugEnabled()) {
            log.debug(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12, argument13);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12);
        }
    }

    public static void info(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12, Object argument13) {
        if (log.isInfoEnabled()) {
            log.info(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12, argument13);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12);
        }
    }

    public static void warn(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12, Object argument13) {
        if (log.isWarnEnabled()) {
            log.warn(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12, argument13);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12);
        }
    }

    public static void error(String format, Object argument1, Object argument2, Object argument3, Object argument4, Object argument5, Object argument6, Object argument7, Object argument8, Object argument9, Object argument10, Object argument11, Object argument12, Object argument13) {
        if (log.isErrorEnabled()) {
            log.error(format, argument1, argument2, argument3, argument4, argument5, argument6, argument7, argument8, argument9, argument10, argument11, argument12, argument13);
        }
    }

    /**
     * Returns the explicit log level of the underlying FlexLB logger.
     */
    public static LogLevel getLevel() {
        Level level = logbackLogger().getLevel();
        if (level == null) {
            return null;
        }
        try {
            return LogLevel.valueOf(level.levelStr.toUpperCase());
        } catch (IllegalArgumentException error) {
            return null;
        }
    }

    /**
     * Updates the runtime FlexLB log level. A null value restores the INFO default.
     */
    public static void setLevel(LogLevel level) {
        Level logbackLevel = level == null
                ? Level.INFO
                : Level.toLevel(level.name(), Level.INFO);
        logbackLogger().setLevel(logbackLevel);
    }

    private static ch.qos.logback.classic.Logger logbackLogger() {
        return (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
    }

}
