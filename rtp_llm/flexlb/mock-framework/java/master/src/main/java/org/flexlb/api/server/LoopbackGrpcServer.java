package org.flexlb.api.server;

import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.util.metrics.GrpcReporter;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;

import java.net.InetSocketAddress;

/**
 * Binds the FlexLB gRPC listener to the local Mock Master's assigned loopback address.
 * The base server manages RPC serving and graceful shutdown.
 */
public final class LoopbackGrpcServer extends FlexlbGrpcServer {
    private final String loopbackAddress;

    public LoopbackGrpcServer(
            FlexlbServiceImpl flexlbServiceImpl,
            ConfigService configService,
            Environment environment,
            @Qualifier("grpcServerEventLoopGroup") EventLoopGroup grpcServerEventLoopGroup,
            GrpcReporter monitor,
            GrpcServerTimingInterceptor grpcServerTimingInterceptor,
            GrpcRequestHeadersInterceptor grpcRequestHeadersInterceptor,
            @Value("${server.address}") String loopbackAddress) {
        super(
                flexlbServiceImpl,
                configService,
                environment,
                grpcServerEventLoopGroup,
                monitor,
                grpcServerTimingInterceptor,
                grpcRequestHeadersInterceptor);
        this.loopbackAddress = loopbackAddress;
    }

    @Override
    protected NettyServerBuilder createServerBuilder(int port) {
        return NettyServerBuilder.forAddress(new InetSocketAddress(loopbackAddress, port));
    }
}
