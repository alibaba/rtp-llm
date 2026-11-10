package org.flexlb.balance.preemption;

/** Decode control route and the original Prefill endpoint for early cancellation. */
public record CancelTarget(String decodeIp, int decodeGrpcPort, String prefillAddress) {

    public boolean isRoutable() {
        return decodeIp != null && !decodeIp.isBlank() && decodeGrpcPort > 0
                && prefillAddress != null && !prefillAddress.isBlank();
    }
}
