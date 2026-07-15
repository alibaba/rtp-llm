package org.flexlb.util;

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
}
