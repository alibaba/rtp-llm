package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.List;

/**
 * Group of requests assembled when a batch trigger fires. On the engine side, one
 * PrefillBatch corresponds to a single DP-aligned atomic step (DP barrier contract:
 * forward in the same step, no splitting, no interleaving with other requests).
 */
public record PrefillBatch(
        ServerStatus prefillTarget,
        List<PendingRequest> requests,
        int dpSize) {

    public int size() {
        return requests.size();
    }

    public String prefillIp() {
        return prefillTarget.getServerIp();
    }

    public int prefillGrpcPort() {
        return prefillTarget.getGrpcPort();
    }
}
