package org.flexlb.autotpm;

import org.flexlb.config.FlexlbConfig;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Normalizes request priority from proto field and/or HTTP header into the
 * canonical priority set defined in {@link FlexlbConfig#getAutoTpmPriorityLevels()}.
 *
 * <p>Resolution order: proto field (if > 0) → header string (if parseable as int) → default.
 * Any value not in the legal set is replaced by the default priority.
 */
public class PriorityNormalizer {

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
     * @return normalized priority guaranteed to be in the legal set
     */
    public int normalize(int protoPriority, String headerPriority) {
        // 1. Proto field takes precedence (> 0 means explicitly set)
        if (protoPriority > 0) {
            return legalPriorities.contains(protoPriority) ? protoPriority : defaultPriority;
        }
        // 2. Header string
        if (headerPriority != null && !headerPriority.isBlank()) {
            try {
                int parsed = Integer.parseInt(headerPriority.trim());
                if (parsed > 0 && legalPriorities.contains(parsed)) {
                    return parsed;
                }
            } catch (NumberFormatException ignored) {
                // fall through to default
            }
        }
        // 3. Default
        return defaultPriority;
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
