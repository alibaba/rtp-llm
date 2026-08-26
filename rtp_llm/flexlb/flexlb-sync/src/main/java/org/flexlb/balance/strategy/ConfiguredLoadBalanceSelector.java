package org.flexlb.balance.strategy;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

/** Executes the unique leaf supporting the request's exact selector config. */
@Component
public final class ConfiguredLoadBalanceSelector {

    private final List<LoadBalanceStrategy> strategies;

    public ConfiguredLoadBalanceSelector(
            List<LoadBalanceStrategy> strategies) {
        this.strategies = List.copyOf(strategies);
        if (this.strategies.isEmpty()) {
            throw new IllegalStateException(
                    "At least one load-balance strategy is required");
        }
    }

    public SelectedRole select(
            BalanceContext context,
            RoleType role,
            String group) {
        FlexlbConfig config = Objects.requireNonNull(
                Objects.requireNonNull(context, "context").getConfig(),
                "request config");
        RoleType exactRole = Objects.requireNonNull(role, "role");
        RoutingConfig.EndpointSelectorConfig configured =
                Objects.requireNonNull(
                        config.getRouter().selectorFor(exactRole),
                        "endpoint selector config");

        LoadBalanceStrategy match = null;
        for (LoadBalanceStrategy candidate : strategies) {
            if (!candidate.supports(exactRole, configured)) {
                continue;
            }
            if (match != null) {
                throw new IllegalStateException(
                        "Multiple load-balance strategies support role="
                                + exactRole + ", selector="
                                + configured.getClass().getName() + ": "
                                + match.getClass().getName() + ", "
                                + candidate.getClass().getName());
            }
            match = candidate;
        }
        if (match == null) {
            throw new IllegalStateException(
                    "No load-balance strategy supports role=" + exactRole
                            + ", selector="
                            + configured.getClass().getName());
        }
        return match.select(context, exactRole, group);
    }
}
