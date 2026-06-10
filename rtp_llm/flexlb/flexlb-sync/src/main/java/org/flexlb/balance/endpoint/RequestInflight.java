package org.flexlb.balance.endpoint;

public record RequestInflight(
        long requestId,
        long kvTokens
) {}
