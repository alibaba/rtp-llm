package org.flexlb.balance.preemption;

/** Original Prefill route which owns propagation of one Engine cancel. */
public record CancelTarget(String prefillIp, int prefillGrpcPort) {

    public boolean isRoutable() {
        return prefillIp != null
                && !prefillIp.isBlank()
                && prefillGrpcPort > 0;
    }
}
