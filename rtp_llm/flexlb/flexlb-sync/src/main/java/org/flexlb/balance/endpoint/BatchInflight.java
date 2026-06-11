package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.BatchRequest;

import java.util.List;

public record BatchInflight(
        long batchId,
        long predictTimeMs,
        List<BatchRequest> requests
) {}
