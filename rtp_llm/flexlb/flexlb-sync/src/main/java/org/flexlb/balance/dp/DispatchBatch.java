package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.List;

/**
 * Group of requests assembled when a batch trigger fires, pre-grouped by dp_rank.
 * On the engine side, one DispatchBatch corresponds to a single DP-aligned atomic step.
 */
public record DispatchBatch(
        ServerStatus prefillTarget,
        List<List<PendingRequest>> rankedRequests,
        int dpSize,
        long blockSize) {

    public int size() {
        return rankedRequests.stream().mapToInt(List::size).sum();
    }

    public List<PendingRequest> requests() {
        return rankedRequests.stream().flatMap(List::stream).toList();
    }

    public String prefillIp() {
        return prefillTarget.getServerIp();
    }

    public int prefillGrpcPort() {
        return prefillTarget.getGrpcPort();
    }
}
