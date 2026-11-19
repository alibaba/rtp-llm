package org.flexlb.mock.master;

import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.ConfigService;
import org.flexlb.httpserver.FlexlbGrpcServer;
import org.flexlb.httpserver.FlexlbServiceImpl;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.metric.FlexMonitor;
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
            FlexMonitor monitor,
            GrpcServerTimingInterceptor grpcServerTimingInterceptor,
            GrpcQosHeaderInterceptor grpcQosHeaderInterceptor,
            @Value("${server.address}") String loopbackAddress) {
        super(
                flexlbServiceImpl,
                configService,
                environment,
                grpcServerEventLoopGroup,
                monitor,
                grpcServerTimingInterceptor,
                grpcQosHeaderInterceptor);
        this.loopbackAddress = loopbackAddress;
    }

    @Override
    protected NettyServerBuilder createServerBuilder(int port) {
        return NettyServerBuilder.forAddress(new InetSocketAddress(loopbackAddress, port));
    }
}
