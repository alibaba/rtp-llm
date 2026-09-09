package org.flexlb.debug;

import java.util.List;
import java.util.Map;

/** Diagnostic values only: never contains a live owner, capability or mutable context. */
public record DebugPage(String consistency, String status,
                        long captureStartedAtMs, long captureFinishedAtMs,
                        int scannedCount, boolean truncated,
                        List<Map<String, Object>> rows, Map<String, Object> metadata) {
    public DebugPage {
        rows = rows.stream().map(Map::copyOf).toList();
        metadata = Map.copyOf(metadata);
    }

    public static DebugPage unavailable(String consistency, String status, long start) {
        return new DebugPage(consistency, status, start, System.currentTimeMillis(),
                0, false, List.of(), Map.of());
    }
}
