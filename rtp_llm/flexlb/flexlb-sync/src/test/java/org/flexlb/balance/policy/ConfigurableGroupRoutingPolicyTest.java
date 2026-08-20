package org.flexlb.balance.policy;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ConfigurableGroupRoutingPolicyTest {

    @Test
    void should_select_group_before_host_load_balancing() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        TrafficPolicyConfig.Rule rule = new TrafficPolicyConfig.Rule();
        rule.setName("long-context");
        rule.setMatch(inputTokensAtLeast(8192L));
        rule.setTargets(List.of(target("long-group", 1)));

        TrafficPolicyConfig trafficPolicyConfig = new TrafficPolicyConfig();
        trafficPolicyConfig.setRules(List.of(rule));
        flexlbConfig.getRouter().setGroupSelector(trafficPolicyConfig);

        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(10000L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(flexlbConfig);
        balanceContext.setRequest(request);
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        ConfigurableGroupRoutingPolicy policy = new ConfigurableGroupRoutingPolicy(configService);
        GroupRoutingDecision decision = policy.route(balanceContext);

        assertEquals("long-group", decision.group());
        assertEquals("trafficPolicy", decision.policyName());
    }

    @Test
    void should_select_weighted_group_before_host_load_balancing() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();

        TrafficPolicyConfig.Rule rule = new TrafficPolicyConfig.Rule();
        rule.setName("split");
        rule.setMatch(inputTokensAtLeast(1L));
        rule.setTargets(List.of(target("blue", 1), target("green", 100)));

        TrafficPolicyConfig trafficPolicyConfig = new TrafficPolicyConfig();
        trafficPolicyConfig.setRules(List.of(rule));
        flexlbConfig.getRouter().setGroupSelector(trafficPolicyConfig);

        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(flexlbConfig);
        balanceContext.setRequest(request);
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        ConfigurableGroupRoutingPolicy policy = new ConfigurableGroupRoutingPolicy(configService);
        GroupRoutingDecision decision = policy.route(balanceContext);

        assertEquals("green", decision.group());
    }

    @Test
    void should_return_empty_decision_when_no_rule_matches() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig flexlbConfig = new FlexlbConfig();

        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(flexlbConfig);
        balanceContext.setRequest(request);

        ConfigurableGroupRoutingPolicy policy = new ConfigurableGroupRoutingPolicy(configService);
        GroupRoutingDecision decision = policy.route(balanceContext);

        assertFalse(decision.hasGroup());
    }

    private static TrafficPolicyConfig.Match inputTokensAtLeast(long min) {
        TrafficPolicyConfig.InputTokens inputTokens = new TrafficPolicyConfig.InputTokens();
        inputTokens.setMin(min);
        TrafficPolicyConfig.Match match = new TrafficPolicyConfig.Match();
        match.setInputTokens(inputTokens);
        return match;
    }

    private static TrafficPolicyConfig.Target target(String group, long weight) {
        TrafficPolicyConfig.Target target = new TrafficPolicyConfig.Target();
        target.setGroup(group);
        target.setWeight(weight);
        return target;
    }
}
