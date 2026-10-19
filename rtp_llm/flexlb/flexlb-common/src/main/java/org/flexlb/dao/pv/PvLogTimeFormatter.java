package org.flexlb.dao.pv;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * Formats absolute PV timestamps while preserving their original epoch-millisecond fields.
 */
final class PvLogTimeFormatter {

    private static final DateTimeFormatter FORMAT = DateTimeFormatter
            .ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
            .withZone(ZoneId.of("Asia/Shanghai"));

    private PvLogTimeFormatter() {
    }

    static String format(long epochMillis) {
        if (epochMillis <= 0 || epochMillis == Long.MAX_VALUE) {
            return null;
        }
        return FORMAT.format(Instant.ofEpochMilli(epochMillis));
    }
}
