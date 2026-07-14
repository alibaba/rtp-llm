package org.flexlb.util;

import org.slf4j.LoggerFactory;

/**
 * Business logger facade. Log filtering is delegated to the configured SLF4J backend.
 */
public final class Logger {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger("flexlbLogger");

    private Logger() {
    }

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
}
