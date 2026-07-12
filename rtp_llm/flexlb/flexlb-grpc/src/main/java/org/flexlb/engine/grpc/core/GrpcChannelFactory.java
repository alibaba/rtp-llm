package org.flexlb.engine.grpc.core;

import io.grpc.ManagedChannel;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioSocketChannel;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Creates gRPC channels with the shared FlexLB transport resources and options.
 */
@Slf4j
@Getter
@Component
public class GrpcChannelFactory {

    private final ThreadPoolExecutor executor;
    private final EventLoopGroup eventLoopGroup;

    public GrpcChannelFactory(
            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup) {
        this.executor = executor;
        this.eventLoopGroup = eventLoopGroup;
    }

    public ManagedChannel create(GrpcTarget target) {
        log.info("Creating gRPC channel: {}", target);
        return NettyChannelBuilder.forAddress(target.host(), target.port())
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                // Connection timeout in milliseconds
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 20)
                // Write buffer water mark: prevents memory accumulation and pendingTasks buildup
                .withOption(ChannelOption.WRITE_BUFFER_WATER_MARK,
                        new WriteBufferWaterMark(64 * 1024, 128 * 1024))
                // Receive/send buffer size
                .withOption(ChannelOption.SO_RCVBUF, 512 * 1024)
                .withOption(ChannelOption.SO_SNDBUF, 512 * 1024)
                // Maximum message size limit (8MB)
                .maxInboundMessageSize(8 * 1024 * 1024)
                // HTTP/2 initial flow control window: prevents transmission issues due to flow control
                .initialFlowControlWindow(2 * 1024 * 1024)
                // Send an HTTP/2 PING after this interval without read activity
                .keepAliveTime(2, TimeUnit.SECONDS)
                // Close the connection if no read activity is observed within this timeout after a PING
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                // Keep probing even when there are no active RPC calls
                .keepAliveWithoutCalls(true)
                .executor(executor)
                .eventLoopGroup(eventLoopGroup)
                .usePlaintext()
                .proxyDetector(GrpcUtil.NOOP_PROXY_DETECTOR)
                .disableRetry()
                .build();
    }
}
