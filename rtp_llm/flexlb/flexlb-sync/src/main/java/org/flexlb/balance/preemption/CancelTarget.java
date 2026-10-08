package org.flexlb.balance.preemption;

/** Selected Decode control endpoint and its Prefill route. */
public record CancelTarget(String decodeIp, int decodeGrpcPort, String prefillAddress) {
    public static CancelTarget of(String decodeIp, int decodeGrpcPort, String prefillIp, int prefillGrpcPort) {
        String host = prefillIp != null && prefillIp.contains(":") ? "[" + prefillIp + "]" : prefillIp;
        return new CancelTarget(decodeIp, decodeGrpcPort, host + ":" + prefillGrpcPort);
    }

    public boolean isRoutable() {
        return decodeIp != null && !decodeIp.isBlank() && decodeGrpcPort > 0
                && prefillAddress != null && !prefillAddress.isBlank();
    }
}
