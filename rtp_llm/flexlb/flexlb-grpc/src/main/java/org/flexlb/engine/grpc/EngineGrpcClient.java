package org.flexlb.engine.grpc;

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
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Engine gRPC client for worker status queries
 */
@Component
@Slf4j
public class EngineGrpcClient extends AbstractGrpcClient<RpcServiceGrpc.RpcServiceBlockingStub> {

    @Getter
    private final Executor executor;
    @Getter
    private final EventLoopGroup eventLoopGroup;

    public EngineGrpcClient(CustomNameResolver nameResolver,
                            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
                            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup,
                            EngineLocalView engineLocalView,
                            GlobalCacheIndex globalCacheIndex,
                            GrpcReporter grpcReporter) {
        super(engineLocalView, globalCacheIndex, grpcReporter);
        this.executor = executor;
        this.eventLoopGroup = eventLoopGroup;
        nameResolver.start(this);
    }

    /**
     * Common method to execute gRPC calls with proper channel management and error handling
     *
     * @param requestTimeoutMs request timeout in milliseconds.
     * @param serviceType      the service type for channel selection
     */
    private <R> R executeGrpcCall(String ip, int port,
                                  Function<RpcServiceGrpc.RpcServiceBlockingStub, R> grpcCall,
                                  long requestTimeoutMs,
                                  ServiceType serviceType) {

        String channelKey = createKey(ip, port, serviceType);
        Invoker invoker = getInvoker(channelKey);
        boolean shutdownAfterInvoke = false;

        if (invoker == null) {
            shutdownAfterInvoke = true;
            log.warn("ip:{} {} grpc channel not found, creating new channel", ip, serviceType);
            invoker = new Invoker(channelKey, createChannel(channelKey));
        } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
            shutdownAfterInvoke = true;
            log.warn("ip:{} {} grpc channel is shutdown or terminated, recreating", ip, serviceType);
            invoker = new Invoker(invoker.getChannelKey(), createChannel(invoker.getChannelKey()));
        }

        try {
            RpcServiceGrpc.RpcServiceBlockingStub rpcServiceStub = invoker.getRpcServiceStub()
                    .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            R result = grpcCall.apply(rpcServiceStub);

            if (shutdownAfterInvoke) {
                invoker.shutdown();
            }

            return result;
        } catch (Exception e) {
            log.error("Exception during {} gRPC call setup for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        }
    }

    /**
     * Get worker status via gRPC
     */
    public EngineRpcService.WorkerStatusPB getWorkerStatus(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getWorkerStatus(request), requestTimeoutMs, ServiceType.WORKER_STATUS);
    }

    /**
     * Get cache status via gRPC
     */
    public EngineRpcService.CacheStatusPB getCacheStatus(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getCacheStatus(request), requestTimeoutMs, ServiceType.CACHE_STATUS);
    }

    @Override
    protected ManagedChannel createChannel(String channelKey) {
        String[] parts = parseServiceKey(channelKey);
        String ip = parts[0];
        int port = Integer.parseInt(parts[1]);
        log.info("Creating new channel for ip: {}, port: {}", ip, port);
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                // 25ms 连接超时
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 25)
                // 写缓冲区水位线：防止内存堆积和 pendingTasks 累积
                .withOption(ChannelOption.WRITE_BUFFER_WATER_MARK, new WriteBufferWaterMark(32 * 1024, 64 * 1024))
                // 接收/发送缓冲区大小优化
                .withOption(ChannelOption.SO_RCVBUF, 256 * 1024)
                .withOption(ChannelOption.SO_SNDBUF, 256 * 1024)
                // 最大消息大小限制（4MB）
                .maxInboundMessageSize(4 * 1024 * 1024)
                .executor(executor)
                .eventLoopGroup(eventLoopGroup)
                .usePlaintext()
                .proxyDetector(GrpcUtil.NOOP_PROXY_DETECTOR)
                .disableRetry()
                .build();
    }

    @Override
    protected RpcServiceGrpc.RpcServiceBlockingStub createStub(ManagedChannel channel) {
        return RpcServiceGrpc.newBlockingStub(channel);
    }
}