package org.flexlb.dispatcher;

import java.util.Locale;

/**
 * Source of per-chunk frontend assignments on the dispatcher fanout path.
 *
 * <p>{@link #MASTER} gives every dispatcher instance one fleet-wide cursor. {@link #LOCAL} is the
 * explicit availability escape hatch: it uses this node's health-filtered {@link FePool} cursor and
 * avoids making FE availability depend on the elected master. Local mode may produce less even
 * fleet-wide distribution when several dispatcher instances are active, so it is intended for
 * incidents and deployments that value availability over globally attributable allocation.
 */
public enum FeAllocationMode {
    MASTER,
    LOCAL;

    public static FeAllocationMode parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("feAllocation must be one of [master, local]");
        }
        try {
            return valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "feAllocation must be one of [master, local], got '" + value + "'", e);
        }
    }

    public String configValue() {
        return name().toLowerCase(Locale.ROOT);
    }
}
