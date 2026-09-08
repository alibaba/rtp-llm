package org.flexlb.config;

import org.flexlb.dao.loadbalance.Request;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrafficPolicyConfigTest {

    @Test
    void should_resolve_group_by_first_matching_api_key_rule() {
        TrafficPolicyConfig config = JsonUtils.toObject("""
                {
                  "rules": [
                    {
                      "name": "vip",
                      "api_keys": ["key-a", "key-b"],
                      "target_group": "vip-group"
                    },
                    {
                      "name": "long-context",
                      "min_seq_len": 8192,
                      "target_group": "long-group"
                    }
                  ]
                }
                """, TrafficPolicyConfig.class);

        Request request = new Request();
        request.setApiKey("key-a");
        request.setSeqLen(16000);

        assertEquals("vip-group", config.resolveTargetGroup(request).orElseThrow());
    }

    @Test
    void should_resolve_group_by_seq_len_range_rule() {
        TrafficPolicyConfig config = JsonUtils.toObject("""
                {
                  "rules": [
                    {
                      "name": "short",
                      "max_seq_len": 2048,
                      "target_group": "short-group"
                    },
                    {
                      "name": "long",
                      "min_seq_len": 2049,
                      "target_group": "long-group"
                    }
                  ]
                }
                """, TrafficPolicyConfig.class);

        Request request = new Request();
        request.setSeqLen(4096);

        assertEquals("long-group", config.resolveTargetGroup(request).orElseThrow());
    }

    @Test
    void should_return_empty_when_disabled() {
        TrafficPolicyConfig config = JsonUtils.toObject("""
                {
                  "enabled": false,
                  "default_group": "default-group"
                }
                """, TrafficPolicyConfig.class);

        Request request = new Request();
        request.setSeqLen(4096);

        assertTrue(config.resolveTargetGroup(request).isEmpty());
    }

    @Test
    void should_resolve_group_by_weighted_target_groups() {
        TrafficPolicyConfig config = JsonUtils.toObject("""
                {
                  "rules": [
                    {
                      "name": "split",
                      "min_seq_len": 1,
                      "target_groups": [
                        {"group": "blue", "weight": 0},
                        {"group": "green", "weight": 100}
                      ]
                    }
                  ]
                }
                """, TrafficPolicyConfig.class);

        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128);

        assertEquals("green", config.resolveTargetGroup(request).orElseThrow());
    }

    @Test
    void should_resolve_group_by_default_weighted_target_groups_when_no_rule_matches() {
        TrafficPolicyConfig config = JsonUtils.toObject("""
                {
                  "default_target_groups": [
                    {"group": "default-a", "weight": 0},
                    {"group": "default-b", "weight": 100}
                  ],
                  "rules": [
                    {
                      "name": "long",
                      "min_seq_len": 8192,
                      "target_group": "long-group"
                    }
                  ]
                }
                """, TrafficPolicyConfig.class);

        Request request = new Request();
        request.setRequestId(12345L);
        request.setSeqLen(128);

        assertEquals("default-b", config.resolveTargetGroup(request).orElseThrow());
    }

    @Test
    void should_report_only_effective_enabled_routing_rules_as_active() {
        TrafficPolicyConfig empty = new TrafficPolicyConfig();
        assertFalse(empty.hasActiveRoutingRules(),
                "the default empty policy must not disable dispatcher pre-assignment");

        TrafficPolicyConfig disabled = JsonUtils.toObject("""
                {"enabled":false,"default_group":"tenant-a"}
                """, TrafficPolicyConfig.class);
        assertFalse(disabled.hasActiveRoutingRules());

        TrafficPolicyConfig invalidTargets = JsonUtils.toObject("""
                {
                  "default_target_groups":[{"group":"", "weight":10}],
                  "rules":[{"api_keys":["key-a"],"target_groups":[{"group":"x","weight":0}]}]
                }
                """, TrafficPolicyConfig.class);
        assertFalse(invalidTargets.hasActiveRoutingRules());

        TrafficPolicyConfig apiKeyRule = JsonUtils.toObject("""
                {"rules":[{"api_keys":["key-a"],"target_group":"tenant-a"}]}
                """, TrafficPolicyConfig.class);
        assertTrue(apiKeyRule.hasActiveRoutingRules());

        TrafficPolicyConfig lengthRule = JsonUtils.toObject("""
                {"rules":[{"min_seq_len":4096,"target_group":"long-context"}]}
                """, TrafficPolicyConfig.class);
        assertTrue(lengthRule.hasActiveRoutingRules());
    }
}
