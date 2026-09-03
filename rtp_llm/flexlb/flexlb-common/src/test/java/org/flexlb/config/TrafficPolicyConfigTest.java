package org.flexlb.config;

import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TrafficPolicyConfigTest {

    @Test
    void first_matching_rule_wins_and_match_constraints_are_anded() {
        TrafficPolicyConfig config = parseGroupSelector("""
                {
                  "rules": [
                    {
                      "name": "vip-long",
                      "match": {
                        "apiKeys": ["key-a"],
                        "inputTokens": {"min": 4096}
                      },
                      "targets": [{"group": "vip-long", "weight": 1}]
                    },
                    {
                      "name": "long",
                      "match": {"inputTokens": {"min": 4096}},
                      "targets": [{"group": "long", "weight": 1}]
                    }
                  ]
                }
                """);

        Request vip = request("1", "key-a", 8192);
        Request regular = request("2", "key-b", 8192);
        assertEquals("vip-long", config.resolveTargetGroup(vip).orElseThrow());
        assertEquals("long", config.resolveTargetGroup(regular).orElseThrow());
    }

    @Test
    void falls_back_to_default_targets_and_weighted_choice_is_retry_stable() {
        TrafficPolicyConfig config = parseGroupSelector("""
                {
                  "defaultTargets": [
                    {"group": "blue", "weight": 90},
                    {"group": "green", "weight": 10}
                  ],
                  "rules": []
                }
                """);
        Request request = request("12345", "key", 128);

        String first = config.resolveTargetGroup(request).orElseThrow();
        assertEquals(first, config.resolveTargetGroup(request).orElseThrow());
        Assertions.assertTrue(Set.of("blue", "green").contains(first));
    }

    @Test
    void rejects_duplicate_shapes_and_invalid_rules() {
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "defaultGroup":"blue",
                  "defaultTargets":[{"group":"blue","weight":1}],
                  "rules":[]
                }
                """));
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "rules":[{
                    "name":"unconditional",
                    "match":{},
                    "targets":[{"group":"blue","weight":1}]
                  }]
                }
                """));
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "rules":[{
                    "name":"bad-range",
                    "match":{"inputTokens":{"min":10,"max":5}},
                    "targets":[{"group":"blue","weight":1}]
                  }]
                }
                """));
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "defaultTargets":[{"group":"blue","weight":0}],
                  "rules":[]
                }
                """));
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "rules":[{
                    "name":"duplicate-key",
                    "match":{"apiKeys":["key-a","key-a"]},
                    "targets":[{"group":"blue","weight":1}]
                  }]
                }
                """));
        assertThrows(ConfigValidationException.class, () -> parseGroupSelector("""
                {
                  "rules":[{
                    "name":"blank-key",
                    "match":{"apiKeys":["  "]},
                    "targets":[{"group":"blue","weight":1}]
                  }]
                }
                """));
    }

    private static TrafficPolicyConfig parseGroupSelector(String json) {
        String document = "{\"router\":{\"groupSelector\":" + json + "}}";
        return FlexlbConfigMerger.mergeWithDefaults(document).getRouter().getGroupSelector();
    }

    private static Request request(String id, String apiKey, long inputTokens) {
        Request request = new Request();
        request.setRequestId(String.valueOf(id));
        request.setApiKey(apiKey);
        request.setSeqLen(inputTokens);
        return request;
    }
}
