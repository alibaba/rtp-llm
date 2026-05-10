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
        TrafficPolicyConfig.TrafficPolicyRule rule = new TrafficPolicyConfig.TrafficPolicyRule();
        rule.setName("long-context");
        rule.setMinSeqLen(8192L);
        rule.setTargetGroup("long-group");

        TrafficPolicyConfig trafficPolicyConfig = new TrafficPolicyConfig();
        trafficPolicyConfig.setRules(List.of(rule));
        flexlbConfig.setTrafficPolicy(trafficPolicyConfig);

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

        TrafficPolicyConfig.TrafficTargetGroup blue = new TrafficPolicyConfig.TrafficTargetGroup();
        blue.setGroup("blue");
        blue.setWeight(0);
        TrafficPolicyConfig.TrafficTargetGroup green = new TrafficPolicyConfig.TrafficTargetGroup();
        green.setGroup("green");
        green.setWeight(100);

        TrafficPolicyConfig.TrafficPolicyRule rule = new TrafficPolicyConfig.TrafficPolicyRule();
        rule.setName("split");
        rule.setMinSeqLen(1L);
        rule.setTargetGroups(List.of(blue, green));

        TrafficPolicyConfig trafficPolicyConfig = new TrafficPolicyConfig();
        trafficPolicyConfig.setRules(List.of(rule));
        flexlbConfig.setTrafficPolicy(trafficPolicyConfig);

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
}
