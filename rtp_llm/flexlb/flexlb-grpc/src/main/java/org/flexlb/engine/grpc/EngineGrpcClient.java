package org.flexlb.engine.grpc;

import com.google.protobuf.MessageLite;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
import io.netty.channel.socket.nio.NioSocketChannel;
import lombok.Getter;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.Logger;
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
public class EngineGrpcClient extends AbstractGrpcClient<AbstractGrpcClient.GrpcStubWrapper> {

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

        if (invoker == null) {
            Logger.warn("ip:{} {} grpc channel not found, creating and adding to pool", ip, serviceType);
            ManagedChannel newChannel = createChannel(channelKey);
            invoker = new Invoker(channelKey, newChannel);
            channelPool.put(channelKey, invoker);
        } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
            Logger.warn("ip:{} {} grpc channel is shutdown or terminated, recreating and updating pool", ip, serviceType);
            ManagedChannel newChannel = createChannel(channelKey);
            invoker = new Invoker(channelKey, newChannel);
            channelPool.put(channelKey, invoker);
        }

        try {
            invoker.updateLastUsedTime();
            GrpcStubWrapper stubWrapper = invoker.getRpcServiceStub()
                    .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            long startTime = System.nanoTime() / 1000;
            R response = grpcCall.apply(stubWrapper);
            long endTime = System.nanoTime() / 1000;

            // Calculate response body size in bytes
            int responseSize = 0;
            if (response instanceof MessageLite messageLite) {
                responseSize = messageLite.getSerializedSize();
            }

            // Record statistics
            long duration = endTime - startTime;
            grpcReporter.reportCallMetrics(ip, serviceType.getOperationName(), duration, responseSize, false);

            return response;
        } catch (StatusRuntimeException e) {
            if (isConnectionBrokenError(e)) {
                invoker.markExpired();
                long connectionDuration = invoker.getConnectionDuration();
                grpcReporter.reportConnectionDuration(ip, serviceType.getOperationName(), connectionDuration);
                Logger.warn("Connection broken for {}:{} {}, duration: {}μs, recreating channel and retrying once, msh:{}",
                        ip, port, serviceType, connectionDuration, e.getMessage());
                return retryWithNewChannel(channelKey, grpcCall, requestTimeoutMs, ip, port, serviceType);
            }
            Logger.error("Exception during {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        } catch (Exception e) {
            Logger.error("Exception during {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, e);
            throw e;
        }
    }

    private boolean isConnectionBrokenError(StatusRuntimeException e) {
        String message = e.getMessage();
        return message != null &&
               (message.contains("end-of-stream mid-frame") ||
                message.contains("Connection reset") ||
                message.contains("Broken pipe") ||
                message.contains("http2 exception") ||
                message.contains("Incomplete header block fragment"));
    }

    private <R> R retryWithNewChannel(String channelKey,
                                      Function<GrpcStubWrapper, R> grpcCall,
                                      long requestTimeoutMs,
                                      String ip, int port,
                                      ServiceType serviceType) {
        ManagedChannel newChannel = createChannel(channelKey);
        Invoker newInvoker = new Invoker(channelKey, newChannel);
        channelPool.put(channelKey, newInvoker);

        Logger.info("Retrying gRPC call with new channel for {}:{} {}", ip, port, serviceType);

        GrpcStubWrapper stubWrapper = newInvoker.getRpcServiceStub()
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

        long startTime = System.nanoTime() / 1000;
        R response = grpcCall.apply(stubWrapper);
        long endTime = System.nanoTime() / 1000;

        // Calculate response body size in bytes
        int responseSize = 0;
        if (response instanceof MessageLite messageLite) {
            responseSize = messageLite.getSerializedSize();
        }

        // Record retry statistics
        long duration = endTime - startTime;
        grpcReporter.reportCallMetrics(ip, serviceType.getOperationName(), duration, responseSize, true);

        return response;
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
        Logger.info("Creating new channel for ip: {}, port: {}", ip, port);
        return NettyChannelBuilder.forAddress(ip, port)
                .channelType(NioSocketChannel.class)
                .withOption(ChannelOption.TCP_NODELAY, true)
                .withOption(ChannelOption.SO_KEEPALIVE, true)
                .withOption(ChannelOption.ALLOCATOR, PooledByteBufAllocator.DEFAULT)
                // Connection timeout in milliseconds
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS, 20)
                // Write buffer water mark: prevents memory accumulation and pendingTasks buildup
                .withOption(ChannelOption.WRITE_BUFFER_WATER_MARK, new WriteBufferWaterMark(64 * 1024, 128 * 1024))
                // Receive/send buffer size
                .withOption(ChannelOption.SO_RCVBUF, 512 * 1024)
                .withOption(ChannelOption.SO_SNDBUF, 512 * 1024)
                // Maximum message size limit (8MB)
                .maxInboundMessageSize(8 * 1024 * 1024)
                // HTTP/2 initial flow control window: prevents transmission issues due to flow control
                .initialFlowControlWindow(2 * 1024 * 1024)
                // gRPC keepalive configuration: keeps connection active, prevents disconnection by intermediate devices
                .keepAliveTime(2, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .keepAliveWithoutCalls(true)
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