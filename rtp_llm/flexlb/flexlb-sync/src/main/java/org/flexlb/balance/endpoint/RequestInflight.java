package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;

public record RequestInflight(
        long requestId,
        long kvTokens,
        long createdAtMs
) implements InflightEvictor.TtlTracked {
    public RequestInflight(long requestId, long kvTokens) {
        this(requestId, kvTokens, System.currentTimeMillis());
    }
}
