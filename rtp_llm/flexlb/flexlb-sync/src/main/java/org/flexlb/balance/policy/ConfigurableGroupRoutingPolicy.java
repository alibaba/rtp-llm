package org.flexlb.balance.policy;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.springframework.stereotype.Component;

@Component
public class ConfigurableGroupRoutingPolicy implements GroupRoutingPolicy {

    private static final String POLICY_NAME = "trafficPolicy";

    private final ConfigService configService;

    public ConfigurableGroupRoutingPolicy(ConfigService configService) {
        this.configService = configService;
    }

    @Override
    public GroupRoutingDecision route(BalanceContext balanceContext) {
        FlexlbConfig config = balanceContext.getConfig() != null ? balanceContext.getConfig() : configService.loadBalanceConfig();
        if (config == null || config.getTrafficPolicy() == null) {
            return GroupRoutingDecision.none();
        }

        return config.getTrafficPolicy()
                .resolveTargetGroup(balanceContext.getRequest())
                .map(group -> GroupRoutingDecision.of(group, POLICY_NAME))
                .orElseGet(GroupRoutingDecision::none);
    }
}
