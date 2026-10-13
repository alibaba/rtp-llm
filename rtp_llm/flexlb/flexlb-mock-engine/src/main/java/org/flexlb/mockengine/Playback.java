package org.flexlb.mockengine;

import java.util.AbstractList;
import java.util.List;
import java.util.Map;

/** Bounded laps with optional whole-lap token translation. Pacing belongs to the sender. */
final class Playback {
    static final List<String> RETIRED_CONTROLS = List.of("RATE_CURVE", "ARRIVAL_PROCESS",
            "LAP_RETAIN_SCHEDULE", "LAP_RETAIN_PROBABILITY", "PLAYBACK_SEED",
            "BURST_FACTOR", "BURST_PERIOD_SECONDS", "BURST_DUTY", "DIURNAL_AMPLITUDE",
            "DIURNAL_PERIOD_SECONDS", "GRADIENT", "GRADIENT_START_SPEED", "GRADIENT_MAX_SPEED");
    final int maxLaps;
    final String identity;

    Playback(Map<String, String> env, boolean loop, long duration) {
        for (String field : RETIRED_CONTROLS) {
            if (!env.getOrDefault(field, "").isBlank())
                throw new IllegalArgumentException("unsupported playback control: " + field);
        }
        maxLaps = Integer.parseInt(text(env, "MAX_LAPS", loop ? "0" : "1"));
        if (maxLaps < 0 || (maxLaps == 0 && duration <= 0))
            throw new IllegalArgumentException("loop playback requires max_laps or bounded duration");
        identity = text(env, "LAP_IDENTITY", "structural-relabel");
        if (!List.of("none", "structural-relabel").contains(identity))
            throw new IllegalArgumentException("invalid lap identity");
    }

    private static String text(Map<String, String> env, String key, String fallback) {
        String value = env.get(key);
        return value == null || value.isBlank() ? fallback : value;
    }

    boolean more(int completedLap) { return maxLaps == 0 || completedLap + 1 < maxLaps; }

    Map<String, Object> manifest() { return Map.of("max_laps", maxLaps, "identity", identity); }

    /** Preserve sharing inside each lap while keeping token ranges disjoint across laps. */
    List<Integer> relabel(List<Integer> tokens, int lap, long tokenStride) {
        if (lap == 0 || identity.equals("none")) return tokens;
        if (lap < 0 || tokenStride < 1 || (lap + 1L) * tokenStride - 1 > Integer.MAX_VALUE)
            throw new IllegalArgumentException("lap token identity budget exhausted");
        long offset = lap * tokenStride;
        return new AbstractList<>() {
            @Override public Integer get(int index) { return Math.toIntExact(tokens.get(index) + offset); }
            @Override public int size() { return tokens.size(); }
        };
    }
}
