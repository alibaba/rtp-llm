package org.flexlb.balance.strategy;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.List;

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
        return strategyFor(context, role).select(context, role, group);
    }

    public EndpointSelection selectForQueue(
            BalanceContext context,
            RoleType role,
            String group) {
        return strategyFor(context, role).selectForQueue(
                context, role, group);
    }

    private LoadBalanceStrategy strategyFor(
            BalanceContext context, RoleType role) {
        FlexlbConfig config = context.getConfig();
        RoleType exactRole = role;
        RoutingConfig.EndpointSelectorConfig configured =
                config.getRouter().selectorFor(exactRole);

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
        return match;
    }
}
