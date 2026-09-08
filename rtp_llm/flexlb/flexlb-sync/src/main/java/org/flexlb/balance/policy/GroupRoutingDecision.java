package org.flexlb.balance.policy;

import org.apache.commons.lang3.StringUtils;

public record GroupRoutingDecision(String group, String policyName) {

    public static GroupRoutingDecision none() {
        return new GroupRoutingDecision(null, null);
    }

    public static GroupRoutingDecision of(String group, String policyName) {
        return new GroupRoutingDecision(group, policyName);
    }

    public boolean hasGroup() {
        return StringUtils.isNotBlank(group);
    }
}
