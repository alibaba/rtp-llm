package org.flexlb.engine.grpc;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
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
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Engine gRPC client for worker status queries
 */
@Component
public class EngineGrpcClient extends AbstractGrpcClient {

    private static final int DEFAULT_CONNECT_TIMEOUT_MILLIS = 20;
    public static final String CONNECT_TIMEOUT_PROPERTY =
            "flexlb.engine-grpc.connect-timeout-ms";

    @Getter
    private final Executor executor;
    @Getter
    private final EventLoopGroup eventLoopGroup;
    private final int connectTimeoutMillis;

    @Autowired
    public EngineGrpcClient(CustomNameResolver nameResolver,
                            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
                            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup,
                            GrpcReporter grpcReporter,
                            @Value("${" + CONNECT_TIMEOUT_PROPERTY + ":"
                                    + DEFAULT_CONNECT_TIMEOUT_MILLIS + "}")
                            int connectTimeoutMillis) {
        super(grpcReporter);
        if (connectTimeoutMillis <= 0) {
            throw new IllegalArgumentException(
                    "connectTimeoutMillis must be positive");
        }
        this.executor = executor;
        this.eventLoopGroup = eventLoopGroup;
        this.connectTimeoutMillis = connectTimeoutMillis;
        nameResolver.start(this);
    }

    private <R> CompletableFuture<R> executeGrpcCallAsync(String ip, int port,
                                                           Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
                                                           long requestTimeoutMs,
                                                           ServiceType serviceType) {
        return executeGrpcCallAsync(ip, port, grpcCall, requestTimeoutMs, serviceType,
                retriesBrokenConnections(serviceType));
    }

    static boolean retriesBrokenConnections(ServiceType serviceType) {
        return serviceType != ServiceType.BATCH_ENQUEUE
                && serviceType != ServiceType.ENGINE_CANCEL;
    }

    private <R> CompletableFuture<R> executeGrpcCallAsync(String ip, int port,
                                                           Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
                                                           long requestTimeoutMs,
                                                           ServiceType serviceType,
                                                           boolean retryOnBrokenConnection) {
        CompletableFuture<R> resultFuture = new CompletableFuture<>();
        long startTime = System.nanoTime();

        try {
            String channelKey = createKey(ip, port, serviceType);
            Invoker invoker = getInvoker(channelKey);

            if (invoker == null) {
                Logger.debug("ip:{} {} grpc channel not found, creating and adding to pool", ip, serviceType);
                ManagedChannel newChannel = createChannel(channelKey);
                invoker = putInvokerIfAbsent(channelKey, newChannel);
            } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
                Logger.debug("ip:{} {} grpc channel is shutdown or terminated, recreating and updating pool", ip, serviceType);
                ManagedChannel newChannel = createChannel(channelKey);
                invoker = replaceInvoker(channelKey, invoker, newChannel);
            }

            invoker.updateLastUsedTime();
            final Invoker finalInvoker = invoker;
            GrpcFutureStubWrapper stubWrapper = new GrpcFutureStubWrapper(
                    RpcServiceGrpc.newFutureStub(finalInvoker.getChannel()),
                    MultimodalRpcServiceGrpc.newFutureStub(finalInvoker.getChannel())
            ).withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            ListenableFuture<R> listenableFuture = grpcCall.apply(stubWrapper);

            Futures.addCallback(listenableFuture, new FutureCallback<R>() {
                @Override
                public void onSuccess(R response) {
                    long endTime = System.nanoTime();
                    int responseSize = 0;
                    if (response instanceof MessageLite messageLite) {
                        responseSize = messageLite.getSerializedSize();
                    }
                    long duration = TimeUnit.NANOSECONDS.toMillis(endTime - startTime);
                    grpcReporter.reportCallMetrics(
                            serviceType.getOperationName(), duration, responseSize, false);
                    resultFuture.complete(response);
                }

                @Override
                public void onFailure(Throwable t) {
                    if (retryOnBrokenConnection
                            && t instanceof StatusRuntimeException e
                            && isConnectionBrokenError(e)) {
                        finalInvoker.markExpired();
                        long connectionDuration = finalInvoker.getConnectionDuration();
                        grpcReporter.reportConnectionDuration(
                                serviceType.getOperationName(), connectionDuration);
                        Logger.debug("Connection broken for {}:{} {}, duration: {}μs, recreating channel and retrying once async, msh:{}",
                                ip, port, serviceType, connectionDuration, e.getMessage());
                        retryWithNewChannelAsync(channelKey, finalInvoker, grpcCall, requestTimeoutMs,
                                ip, port, serviceType, resultFuture, startTime);
                    } else {
                        Logger.debug("Exception during async {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, t);
                        resultFuture.completeExceptionally(t);
                    }
                }
            }, Runnable::run);

        } catch (Exception e) {
            Logger.debug("Exception initiating async {} gRPC call for {}:{}", serviceType.getOperationName(), ip, port, e);
            resultFuture.completeExceptionally(e);
        }

        return resultFuture;
    }

    private <R> void retryWithNewChannelAsync(String channelKey,
                                               Invoker staleInvoker,
                                               Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
                                               long requestTimeoutMs,
                                               String ip, int port,
                                               ServiceType serviceType,
                                               CompletableFuture<R> resultFuture,
                                               long originalStartTime) {
        try {
            ManagedChannel newChannel = createChannel(channelKey);
            Invoker newInvoker = replaceInvoker(channelKey, staleInvoker, newChannel);

            Logger.debug("Retrying async gRPC call with new channel for {}:{} {}", ip, port, serviceType);

            GrpcFutureStubWrapper stubWrapper = new GrpcFutureStubWrapper(
                    RpcServiceGrpc.newFutureStub(newInvoker.getChannel()),
                    MultimodalRpcServiceGrpc.newFutureStub(newInvoker.getChannel())
            ).withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

            ListenableFuture<R> listenableFuture = grpcCall.apply(stubWrapper);

            Futures.addCallback(listenableFuture, new FutureCallback<R>() {
                @Override
                public void onSuccess(R response) {
                    long endTime = System.nanoTime();
                    int responseSize = 0;
                    if (response instanceof MessageLite messageLite) {
                        responseSize = messageLite.getSerializedSize();
                    }
                    long duration = TimeUnit.NANOSECONDS.toMillis(endTime - originalStartTime);
                    grpcReporter.reportCallMetrics(
                            serviceType.getOperationName(), duration, responseSize, true);
                    resultFuture.complete(response);
                }

                @Override
                public void onFailure(Throwable t) {
                    Logger.debug("Async retry failed for {}:{} {}", ip, port, serviceType, t);
                    resultFuture.completeExceptionally(t);
                }
            }, Runnable::run);

        } catch (Exception e) {
            Logger.debug("Exception during async retry for {}:{} {}", ip, port, serviceType, e);
            resultFuture.completeExceptionally(e);
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

    /**
     * Get worker status via gRPC (async)
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCallAsync(ip, port, stub -> stub.getRpcServiceFutureStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.WORKER_STATUS);
    }

    /**
     * Get cache status via gRPC (async)
     */
    public CompletableFuture<EngineRpcService.CacheStatusPB> getCacheStatusAsync(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCallAsync(ip, port, stub -> stub.getRpcServiceFutureStub().getCacheStatus(request), requestTimeoutMs, ServiceType.CACHE_STATUS);
    }

    /**
     * Get multimodal worker status via gRPC (async)
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getMultimodalWorkerStatusAsync(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCallAsync(ip, port, stub -> stub.getMultimodalFutureStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_WORKER_STATUS);
    }

    /**
     * Get multimodal cache status via gRPC (async)
     */
    public CompletableFuture<EngineRpcService.CacheStatusPB> getMultimodalCacheStatusAsync(String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
        return executeGrpcCallAsync(ip, port, stub -> stub.getMultimodalFutureStub().getCacheStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_CACHE_STATUS);
    }

    /**
     * Submit a batch of already-routed requests to a Prefill worker (async)
     */
    public CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> batchEnqueueAsync(String ip, int port, EngineRpcService.EnqueueBatchRequestPB request, long requestTimeoutMs) {
        // EnqueueBatch is not safe to replay after an ambiguous connection
        // failure: the Engine may have accepted the first invocation even
        // though its ACK was lost. Reconciliation is owned by the scheduler's
        // request-id cancel fence, so this client must expose the ambiguity
        // instead of issuing a second EnqueueBatch automatically.
        return executeGrpcCallAsync(ip, port, stub -> stub.getRpcServiceFutureStub().enqueueBatch(request),
                requestTimeoutMs, ServiceType.BATCH_ENQUEUE);
    }

    /**
     * AutoTPM priority-preemption Cancel to the victim's original Prefill.
     * No transport retry (channel is built with disableRetry); idempotency
     * rides on request_id.
     */
    public CompletableFuture<EngineRpcService.CancelResponsePB> cancelAsync(String ip, int port, EngineRpcService.CancelRequestPB request, long requestTimeoutMs) {
        // Do not retry Cancel after an ambiguous connection failure: the first
        // attempt may already have been ACCEPTED and completed, in which case a
        // retry could return NOT_FOUND and turn a successful cancel into a false
        // negative at the Master.
        return executeGrpcCallAsync(ip, port, stub -> stub.getRpcServiceFutureStub().cancel(request),
                requestTimeoutMs, ServiceType.ENGINE_CANCEL);
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
                .withOption(ChannelOption.CONNECT_TIMEOUT_MILLIS,
                        connectTimeoutMillis)
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

}
