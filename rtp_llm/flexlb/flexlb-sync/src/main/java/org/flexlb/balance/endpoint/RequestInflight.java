package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

record RequestInflight(
        long kvTokens,
        long createdAtMs
) implements InflightEvictor.TtlTracked {
    RequestInflight(long kvTokens) {
        this(kvTokens, System.currentTimeMillis());
    }
}
