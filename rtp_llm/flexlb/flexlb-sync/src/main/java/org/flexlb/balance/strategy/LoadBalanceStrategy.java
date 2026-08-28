package org.flexlb.balance.strategy;

import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;

public interface LoadBalanceStrategy {

    /** Whether this leaf is the exact owner for one role/config pair. */
    boolean supports(
            RoleType role,
            RoutingConfig.EndpointSelectorConfig configured);

    /**
     * Select one exact endpoint generation.  A {@code null} result is the
     * ordinary no-available-worker outcome; a non-null result owns one pin and
     * must be consumed or closed by the caller.
     */
    SelectedRole select(BalanceContext context, RoleType roleType, String group);

    /** Select one endpoint for a queue placement attempt. */
    default EndpointSelection selectForQueue(
            BalanceContext context, RoleType roleType, String group) {
        SelectedRole selected = select(context, roleType, group);
        return selected == null
                ? EndpointSelection.unavailable(roleType)
                : EndpointSelection.selected(selected);
    }
}
