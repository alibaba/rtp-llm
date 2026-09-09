package org.flexlb.debug;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Per-capture builder, never stored on a business owner. */
public final class DebugRows {
    private final DebugQuery query;
    private final long started = System.currentTimeMillis();
    private final List<Map<String, Object>> rows = new ArrayList<>();
    private int scanned;
    private boolean truncated;

    public DebugRows(DebugQuery query) {
        this.query = query;
    }

    public boolean visit() {
        if (scanned >= query.scanLimit() || rows.size() >= query.limit()) {
            truncated = true;
            return false;
        }
        scanned++;
        return true;
    }

    public void add(Map<String, Object> row) {
        rows.add(Map.copyOf(row));
    }

    public DebugPage finish(String consistency, Map<String, Object> metadata) {
        Map<String, Object> coverage = new LinkedHashMap<>(metadata);
        coverage.put("row_limit", query.limit());
        coverage.put("scan_limit", query.scanLimit());
        if (truncated) {
            coverage.put("truncation_reason", rows.size() >= query.limit() ? "row_limit" : "scan_limit");
        }
        return new DebugPage(consistency, truncated ? "partial" : "ok", started,
                System.currentTimeMillis(), scanned, truncated, rows, coverage);
    }

    public static Map<String, Object> fields(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            if (pairs[i + 1] != null) {
                result.put((String) pairs[i], pairs[i + 1]);
            }
        }
        return Map.copyOf(result);
    }
}
