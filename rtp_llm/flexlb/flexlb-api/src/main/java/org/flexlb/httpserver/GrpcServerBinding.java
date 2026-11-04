package org.flexlb.httpserver;

import io.grpc.netty.NettyServerBuilder;
import java.net.InetSocketAddress;
import org.springframework.core.env.Environment;

/** Align gRPC with an explicitly configured HTTP bind address. */
final class GrpcServerBinding {
    private GrpcServerBinding() {}

    static NettyServerBuilder builder(Environment environment, int port) {
        InetSocketAddress address = configuredAddress(environment, port);
        return address == null ? NettyServerBuilder.forPort(port) : NettyServerBuilder.forAddress(address);
    }

    static InetSocketAddress configuredAddress(Environment environment, int port) {
        String address = environment.getProperty("server.address");
        return address == null || address.isBlank() ? null : new InetSocketAddress(address, port);
    }
}
