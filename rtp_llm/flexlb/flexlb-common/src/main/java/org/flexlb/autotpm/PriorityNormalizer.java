package org.flexlb.autotpm;

import org.flexlb.config.FlexlbConfig;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Normalizes request priority from proto field and/or HTTP header into the
 * canonical priority set defined in {@link FlexlbConfig#getAutoTpmPriorityLevels()}.
 *
 * <p>Resolution order: proto field (if > 0) → header string → no priority.
 *
 * <p>D12 (task40 revision) semantics:
 * <ul>
 *   <li>explicitly carried legal value → kept as-is</li>
 *   <li>explicitly carried illegal value → normalized to the default priority</li>
 *   <li>header carrying an explicit "0" → {@link #NO_PRIORITY} sentinel
 *       (treated as "not carried")</li>
 *   <li>neither proto nor header carried → {@link #NO_PRIORITY} sentinel;
 *       the request is scheduled on the legacy path end-to-end (no priority
 *       ordering/yield/eviction, never a victim, no preemption, no priority
 *       metrics)</li>
 * </ul>
 */
public class PriorityNormalizer {

    /** Sentinel for "no priority carried" — the request takes the legacy path. */
    public static final int NO_PRIORITY = 0;

    private final Set<Integer> legalPriorities;
    private final int defaultPriority;

    public PriorityNormalizer(FlexlbConfig config) {
        this.defaultPriority = config.getAutoTpmDefaultPriority();
        this.legalPriorities = parseLevels(config.getAutoTpmPriorityLevels());
    }

    /**
     * Normalize a priority value.
     *
     * @param protoPriority priority from the proto field (0 = unset in proto3)
     * @param headerPriority priority from the HTTP/gRPC header (may be null or non-numeric)
     * @return normalized priority in the legal set, or {@link #NO_PRIORITY}
     *         when the request carried no priority at all
     */
    public int normalize(int protoPriority, String headerPriority) {
        // 1. Proto field takes precedence (> 0 means explicitly set)
        if (protoPriority > 0) {
            return legalPriorities.contains(protoPriority) ? protoPriority : defaultPriority;
        }
        // 2. Header string: an explicit "0" is the no-priority sentinel; any
        //    other carried value is either legal or normalized to the default.
        if (headerPriority != null && !headerPriority.isBlank()) {
            try {
                int parsed = Integer.parseInt(headerPriority.trim());
                if (parsed == 0) {
                    return NO_PRIORITY;
                }
                if (parsed > 0 && legalPriorities.contains(parsed)) {
                    return parsed;
                }
            } catch (NumberFormatException ignored) {
                // carried but unparseable → default
            }
            return defaultPriority;
        }
        // 3. Neither carried → no-priority sentinel (legacy scheduling path)
        return NO_PRIORITY;
    }

    private static Set<Integer> parseLevels(String levels) {
        if (levels == null || levels.isBlank()) {
            return Collections.emptySet();
        }
        Set<Integer> result = new HashSet<>();
        for (String part : levels.split(",")) {
            try {
                result.add(Integer.parseInt(part.trim()));
            } catch (NumberFormatException ignored) {
                // skip malformed entries
            }
        }
        return Collections.unmodifiableSet(result);
    }
}
