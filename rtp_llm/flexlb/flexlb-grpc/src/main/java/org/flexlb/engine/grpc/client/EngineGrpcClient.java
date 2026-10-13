package org.flexlb.engine.grpc.client;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcChannelPool;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Engine gRPC client for status, batch enqueue, and cancel requests.
 */
@Component
public class EngineGrpcClient implements EngineAddressResolver.Listener {

    private final GrpcChannelFactory channelFactory;
    private final GrpcChannelPool<EngineChannelKey> channelPool;
    private final GrpcReporter grpcReporter;
    private final boolean kvcmEnabled;
    private final long enqueueTimeoutMillis;

    public EngineGrpcClient(
            EngineAddressResolver addressResolver,
            GrpcChannelFactory channelFactory,
            GrpcReporter grpcReporter,
            CacheMatchConfiguration cacheMatchConfiguration) {
        this(addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration, 5_000L);
    }

    private enum MetricOperation { CALL_METRICS, CONNECTION_DURATION }
    private static final int MAX_METRICS_ERROR_MESSAGE_CHARS = 256;
    private static final long METRICS_WARNING_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(30);
    private final AtomicLongArray lastMetricsWarningNanos = new AtomicLongArray(MetricOperation.values().length);

    @Autowired
    public EngineGrpcClient(
            EngineAddressResolver addressResolver,
            GrpcChannelFactory channelFactory,
            GrpcReporter grpcReporter,
            CacheMatchConfiguration cacheMatchConfiguration,
            @Value("${flexlb.engine-grpc.enqueue-timeout-ms:5000}") long enqueueTimeoutMillis) {
        this.channelFactory = channelFactory;
        this.channelPool = new GrpcChannelPool<>(key -> channelFactory.create(key.target()));
        this.grpcReporter = grpcReporter;
        this.kvcmEnabled = cacheMatchConfiguration.isKvcmEnabled();
        if (enqueueTimeoutMillis <= 0L) {
            throw new IllegalArgumentException("enqueueTimeoutMillis must be positive");
        }
        this.enqueueTimeoutMillis = enqueueTimeoutMillis;
        addressResolver.subscribe(this);
    }

    @Override
    public void onAddressUpdate(List<WorkerHost> hosts) {
        if (hosts == null) {
            Logger.error("received null host list");
            return;
        }
        Set<EngineChannelKey> activeKeys = new HashSet<>();
        Set<String> activeIps = new HashSet<>();
        for (WorkerHost host : hosts) {
            String ip = host.getIp();
            activeIps.add(ip);
            for (ServiceType serviceType : ServiceType.values()) {
                if (kvcmEnabled && serviceType.isCacheStatusService()) {
                    continue;
                }
                int port = serviceType.isStatusService()
                        ? host.getWorkerStatusPort()
                        : host.getGrpcPort();
                activeKeys.add(new EngineChannelKey(ip, port, serviceType));
            }
        }

        Logger.info("engine address update, host count:{}, channel pool size:{}",
                hosts.size(), channelPool.size());
        for (EngineChannelKey key : activeKeys) {
            try {
                channelPool.getOrCreate(key);
            } catch (Exception e) {
                Logger.error("create channel for {} failed", key, e);
            }
        }
        channelPool.removeChannelsForInactiveGroups(activeIps, EngineChannelKey::ip);
    }

    private <R> CompletableFuture<R> executeGrpcCallAsync(
            String ip,
            int port,
            Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
            long requestTimeoutMs,
            ServiceType serviceType) {
        CompletableFuture<R> result = new CompletableFuture<>();
        EngineChannelKey channelKey = new EngineChannelKey(ip, port, serviceType);
        long startTime = System.nanoTime();
        try {
            GrpcChannelPool.PooledChannel pooledChannel = channelPool.getOrCreate(channelKey);
            invokeAsync(
                    channelKey,
                    pooledChannel,
                    grpcCall,
                    requestTimeoutMs,
                    serviceType,
                    retriesBrokenConnections(serviceType),
                    false,
                    startTime,
                    result);
        } catch (Exception error) {
            result.completeExceptionally(error);
        }
        return result;
    }

    private <R> void invokeAsync(
            EngineChannelKey channelKey,
            GrpcChannelPool.PooledChannel pooledChannel,
            Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
            long requestTimeoutMs,
            ServiceType serviceType,
            boolean retryOnBrokenConnection,
            boolean retry,
            long startTime,
            CompletableFuture<R> result) {
        try {
            pooledChannel.markUsed();
            GrpcFutureStubWrapper stub = createFutureStub(pooledChannel.getChannel())
                    .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);
            ListenableFuture<R> response = grpcCall.apply(stub);
            Futures.addCallback(response, new FutureCallback<>() {
                @Override
                public void onSuccess(R value) {
                    try {
                        reportSuccessfulCall(serviceType, startTime, retry, channelKey.ip(), channelKey.port());
                    } finally {
                        result.complete(value);
                    }
                }

                @Override
                public void onFailure(Throwable error) {
                    if (retryOnBrokenConnection
                            && error instanceof StatusRuntimeException statusError
                            && isConnectionBrokenError(statusError)) {
                        try {
                            pooledChannel.markExpired();
                            reportConnectionDuration(serviceType,
                                    pooledChannel.getConnectionDurationUs(), channelKey.ip(), channelKey.port());
                            GrpcChannelPool.PooledChannel replacement =
                                    channelPool.replace(channelKey, pooledChannel);
                            invokeAsync(
                                    channelKey,
                                    replacement,
                                    grpcCall,
                                    requestTimeoutMs,
                                    serviceType,
                                    false,
                                    true,
                                    startTime,
                                    result);
                        } catch (Exception retryError) {
                            result.completeExceptionally(retryError);
                        }
                        return;
                    }
                    result.completeExceptionally(error);
                }
            }, Runnable::run);
        } catch (Exception error) {
            result.completeExceptionally(error);
        }
    }

    private void reportSuccessfulCall(ServiceType serviceType,
                                      long originalStartTime, boolean retry, String ip, int port) {
        try {
            long duration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - originalStartTime);
            grpcReporter.reportCallMetrics(serviceType.getOperationName(), duration, retry);
        } catch (RuntimeException failure) {
            // Metrics cannot change the response or prevent Future completion.
            logMetricsFailure(MetricOperation.CALL_METRICS, serviceType, ip, port, failure);
        }
    }

    private void reportConnectionDuration(ServiceType serviceType, long duration, String ip, int port) {
        try {
            grpcReporter.reportConnectionDuration(serviceType.getOperationName(), duration);
        } catch (RuntimeException failure) {
            logMetricsFailure(MetricOperation.CONNECTION_DURATION, serviceType, ip, port, failure);
        }
    }

    /** At most one warning per failure stage every 30 seconds; successful calls touch no limiter state. */
    private void logMetricsFailure(MetricOperation operation, ServiceType serviceType,
                                   String ip, int port, RuntimeException failure) {
        long now = System.nanoTime();
        int slot = operation.ordinal();
        long previous = lastMetricsWarningNanos.get(slot);
        if ((previous == 0L || now - previous >= METRICS_WARNING_INTERVAL_NANOS)
                && lastMetricsWarningNanos.compareAndSet(slot, previous, now)) {
            try {
                String message = failure.getMessage();
                String summary = message == null ? "UNKNOWN"
                        : message.substring(0, Math.min(message.length(), MAX_METRICS_ERROR_MESSAGE_CHARS))
                                .replace('\n', ' ').replace('\r', ' ');
                Logger.warn("event=grpc_metrics_failed stage={} service={} endpoint={}:{} errorType={} error={}",
                        operation, serviceType.getOperationName(), ip, port, failure.getClass().getSimpleName(), summary);
            } catch (RuntimeException | LinkageError loggingFailure) {
                // Failure reporting must not prevent the RPC future from completing.
            }
        }
    }

    private boolean isConnectionBrokenError(StatusRuntimeException e) {
        String message = e.getMessage();
        return message != null
                && (message.contains("end-of-stream mid-frame")
                || message.contains("Connection reset")
                || message.contains("Broken pipe")
                || message.contains("http2 exception")
                || message.contains("Incomplete header block fragment"));
    }

    static boolean retriesBrokenConnections(ServiceType serviceType) {
        return serviceType != ServiceType.BATCH_ENQUEUE
                && serviceType != ServiceType.ENGINE_CANCEL;
    }

    /**
     * Queries worker status asynchronously, recreating a broken connection and retrying once.
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(
            String ip,
            int port,
            EngineRpcService.StatusVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.rpcServiceStub().getWorkerStatus(request),
                requestTimeoutMs,
                ServiceType.WORKER_STATUS);
    }

    /**
     * Queries cache status asynchronously, recreating a broken connection and retrying once.
     */
    public CompletableFuture<EngineRpcService.CacheStatusPB> getCacheStatusAsync(
            String ip,
            int port,
            EngineRpcService.CacheVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.rpcServiceStub().getCacheStatus(request),
                requestTimeoutMs,
                ServiceType.CACHE_STATUS);
    }

    /**
     * Queries multimodal worker status asynchronously, recreating a broken connection and retrying once.
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getMultimodalWorkerStatusAsync(
            String ip,
            int port,
            EngineRpcService.StatusVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.multimodalRpcServiceStub().getWorkerStatus(request),
                requestTimeoutMs,
                ServiceType.MULTIMODAL_WORKER_STATUS);
    }

    /**
     * Queries multimodal cache status asynchronously, recreating a broken connection and retrying once.
     */
    public CompletableFuture<EngineRpcService.CacheStatusPB> getMultimodalCacheStatusAsync(
            String ip,
            int port,
            EngineRpcService.CacheVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.multimodalRpcServiceStub().getCacheStatus(request),
                requestTimeoutMs,
                ServiceType.MULTIMODAL_CACHE_STATUS);
    }

    /**
     * Enqueues a batch asynchronously without transport retry because replay may duplicate accepted work.
     */
    public CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> batchEnqueueAsync(
            String ip,
            int port,
            EngineRpcService.EnqueueBatchRequestPB request) {
        return batchEnqueueAsync(ip, port, request, enqueueTimeoutMillis);
    }

    public CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> batchEnqueueAsync(
            String ip,
            int port,
            EngineRpcService.EnqueueBatchRequestPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.rpcServiceStub().enqueueBatch(request),
                requestTimeoutMs,
                ServiceType.BATCH_ENQUEUE);
    }

    /**
     * Cancels a request asynchronously without transport retry because the original cancel may have succeeded.
     */
    public CompletableFuture<EngineRpcService.CancelResponsePB> cancelAsync(
            String ip,
            int port,
            EngineRpcService.CancelRequestPB request,
            long requestTimeoutMs) {
        return executeGrpcCallAsync(
                ip,
                port,
                stub -> stub.rpcServiceStub().cancel(request),
                requestTimeoutMs,
                ServiceType.ENGINE_CANCEL);
    }

    public Executor getExecutor() {
        return channelFactory.getExecutor();
    }

    public EventLoopGroup getEventLoopGroup() {
        return channelFactory.getEventLoopGroup();
    }

    @Scheduled(fixedRate = 2000)
    public void reportChannelPoolSize() {
        grpcReporter.reportChannelPoolSize(channelPool.size());
    }

    @PreDestroy
    public void shutdown() {
        channelPool.shutdown(1, TimeUnit.SECONDS);
    }

    private GrpcFutureStubWrapper createFutureStub(ManagedChannel channel) {
        return new GrpcFutureStubWrapper(
                RpcServiceGrpc.newFutureStub(channel),
                MultimodalRpcServiceGrpc.newFutureStub(channel));
    }

    private record EngineChannelKey(String ip, int port, ServiceType serviceType) {

        GrpcTarget target() {
            return new GrpcTarget(ip, port);
        }

        @Override
        public String toString() {
            return target() + ":" + serviceType.getSuffix();
        }
    }

    private record GrpcFutureStubWrapper(RpcServiceGrpc.RpcServiceFutureStub rpcServiceStub,
                                         MultimodalRpcServiceGrpc.MultimodalRpcServiceFutureStub multimodalRpcServiceStub) {

        GrpcFutureStubWrapper withDeadlineAfter(long timeout, TimeUnit unit) {
            return new GrpcFutureStubWrapper(
                    rpcServiceStub.withDeadlineAfter(timeout, unit),
                    multimodalRpcServiceStub.withDeadlineAfter(timeout, unit));
        }
    }

    enum ServiceType {
        WORKER_STATUS("worker", "GetWorkerStatus"),
        CACHE_STATUS("cache", "GetCacheStatus"),
        MULTIMODAL_WORKER_STATUS("multimodal_worker", "GetWorkerStatus"),
        MULTIMODAL_CACHE_STATUS("multimodal_cache", "GetCacheStatus"),
        BATCH_ENQUEUE("batch_enqueue", "EnqueueBatch"),
        ENGINE_CANCEL("engine_cancel", "Cancel");

        private final String suffix;
        private final String operationName;

        ServiceType(String suffix, String operationName) {
            this.suffix = suffix;
            this.operationName = operationName;
        }

        String getSuffix() {
            return suffix;
        }

        String getOperationName() {
            return operationName;
        }

        boolean isStatusService() {
            return this != BATCH_ENQUEUE && this != ENGINE_CANCEL;
        }

        boolean isCacheStatusService() {
            return this == CACHE_STATUS || this == MULTIMODAL_CACHE_STATUS;
        }
    }
}
