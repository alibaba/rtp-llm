package org.flexlb.engine.grpc.core;

/**
 * Network target for one gRPC channel.
 */
public record GrpcTarget(String host, int port) {
    @Override
    public String toString() {
        return authority();
    }

    public String authority() {
        return host + ":" + port;
    }
}
