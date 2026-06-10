package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.RequestProfile;

import java.util.List;

public record BatchInflight(
        long batchId,
        long predictTimeMs,
        List<Long> requestIds,
        List<RequestProfile> profiles
) {}
