package org.flexlb.engine.grpc;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import io.grpc.ManagedChannel;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import lombok.Getter;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * Engine gRPC client for worker status queries
 */
@Component
public class EngineGrpcClient extends AbstractGrpcClient<AbstractGrpcClient.GrpcStubWrapper> {

    private static final Logger LOGGER = LoggerFactory.getLogger(EngineGrpcClient.class);

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
                                  Function<GrpcStubWrapper, R> grpcCall,
                                  long requestTimeoutMs,
                                  ServiceType serviceType) {

        String channelKey = createKey(ip, port, serviceType);
        Invoker invoker = getInvoker(channelKey);
        boolean shutdownAfterInvoke = false;

        if (invoker == null) {
            shutdownAfterInvoke = true;
            LOGGER.warn("ip:{} {} grpc channel not found, creating new channel", ip, serviceType);
            invoker = new Invoker(channelKey, createChannel(channelKey));
        } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
            shutdownAfterInvoke = true;
            LOGGER.warn("ip:{} {} grpc channel is shutdown or terminated, recreating", ip, serviceType);
            invoker = new Invoker(invoker.getChannelKey(), createChannel(invoker.getChannelKey()));
        }

        try {
            GrpcStubWrapper stubWrapper = invoker.getRpcServiceStub()
                    .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            R result = grpcCall.apply(stubWrapper);

            if (shutdownAfterInvoke) {
                invoker.shutdown();
            }

            return result;
        } catch (Exception e) {
            LOGGER.error("Exception during {} gRPC call setup for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        }
    }

    /**
     * Get worker status via gRPC
     */
    public EngineRpcService.WorkerStatusPB getWorkerStatus(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.WORKER_STATUS);
    }

    /**
     * Get cache status via gRPC
     */
    public EngineRpcService.CacheStatusPB getCacheStatus(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getRpcServiceStub().getCacheStatus(request), requestTimeoutMs, ServiceType.CACHE_STATUS);
    }

    /**
     * Get multimodal worker status via gRPC
     */
    public EngineRpcService.WorkerStatusPB getMultimodalWorkerStatus(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getMultimodalRpcServiceStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_WORKER_STATUS);
    }

    /**
     * Get multimodal cache status via gRPC
     */
    public EngineRpcService.CacheStatusPB getMultimodalCacheStatus(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCall(ip, port, stub -> stub.getMultimodalRpcServiceStub().getCacheStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_CACHE_STATUS);
    }

    @Override
    protected ManagedChannel createChannel(String channelKey) {
        String[] parts = parseServiceKey(channelKey);
        String ip = parts[0];
        int port = Integer.parseInt(parts[1]);
        LOGGER.info("Creating new channel for ip: {}, port: {}", ip, port);
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                // 25ms 连接超时
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 25)
                .executor(executor)
                .eventLoopGroup(eventLoopGroup)
                .usePlaintext()
                .proxyDetector(GrpcUtil.NOOP_PROXY_DETECTOR)
                .disableRetry()
                .build();
    }

    @Override
    protected GrpcStubWrapper createStub(ManagedChannel channel) {
        return new GrpcStubWrapper(
                RpcServiceGrpc.newBlockingStub(channel),
                MultimodalRpcServiceGrpc.newBlockingStub(channel)
        );
    }
}