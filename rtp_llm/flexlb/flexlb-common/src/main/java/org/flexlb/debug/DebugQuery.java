package org.flexlb.debug;

/** Bounds apply before copying entries, including when a request filter misses. */
public record DebugQuery(int limit, int scanLimit, Long requestId) {
    public DebugQuery {
        if (limit < 1 || limit > 5000 || scanLimit < limit || scanLimit > 20000) {
            throw new IllegalArgumentException("limit must be 1..5000; scan_limit must be limit..20000");
        }
    }

    public boolean matches(long id) {
        return requestId == null || requestId == id;
    }
}
