package org.flexlb.engine.grpc.client;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.netty.channel.EventLoopGroup;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.GrpcCallTrace;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.engine.grpc.GrpcTransportObserver;
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
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.function.Function;


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

    private enum MetricOperation {
        CALL_METRICS,
        CONNECTION_DURATION,
        CALL_FAILURE,
        DELIVERY_DELAY
    }

    private final GrpcTransportObserver transportObserver = new GrpcTransportObserver();
    private static final int MAX_METRICS_ERROR_MESSAGE_CHARS = 256;
    private static final long METRICS_WARNING_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(30);
    private final AtomicLongArray lastMetricsWarningNanos =
            new AtomicLongArray(MetricOperation.values().length);

    public EngineGrpcClient(
            EngineAddressResolver addressResolver,
            GrpcChannelFactory channelFactory,
            GrpcReporter grpcReporter,
            CacheMatchConfiguration cacheMatchConfiguration) {
        this(addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration, 5_000L);
    }

    @Autowired
    public EngineGrpcClient(
            EngineAddressResolver addressResolver,
            GrpcChannelFactory channelFactory,
            GrpcReporter grpcReporter,
            CacheMatchConfiguration cacheMatchConfiguration,
            @Value("${flexlb.engine-grpc.enqueue-timeout-ms:5000}") long enqueueTimeoutMillis) {
        this.channelFactory = channelFactory;
        this.channelPool =
                new GrpcChannelPool<>(
                        key -> channelFactory.create(key.target(), transportObserver));
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
                int port =
                        serviceType.isStatusService()
                                ? host.getWorkerStatusPort()
                                : host.getGrpcPort();
                activeKeys.add(new EngineChannelKey(ip, port, serviceType));
            }
        }

        Logger.info(
                "engine address update, host count:{}, channel pool size:{}",
                hosts.size(),
                channelPool.size());
        for (EngineChannelKey key : activeKeys) {
            try {
                channelPool.getOrCreate(key);
            } catch (Exception e) {
                Logger.error("create channel for {} failed", key, e);
            }
        }
        channelPool.removeChannelsForInactiveGroups(activeIps, EngineChannelKey::ip);
    }

    static boolean shouldRetry(ServiceType serviceType, Throwable failure, GrpcCallTrace trace) {
        if (!retriesBrokenConnections(serviceType)) {
            return false;
        }
        try {
            GrpcFailure.Analysis analysis = GrpcFailure.analyze(failure);
            GrpcCallTrace.Snapshot transport = trace.snapshot();
            // Netty may close the channel after a read-allocation OOM without retaining that cause.
            if ("MASTER".equals(transport.failureOrigin(analysis))) {
                return false;
            }
            // Clean socket EOF may reach gRPC as bare UNAVAILABLE. Use the same socket's
            // observed cause only when the final exception supplies no more specific cause.
            if (analysis.kind() == GrpcFailure.Kind.RPC_FAILURE
                    && analysis.grpcCode() == Status.Code.UNAVAILABLE
                    && transport.transportCauseKind() != null) {
                analysis =
                        new GrpcFailure.Analysis(
                                transport.transportCauseKind(),
                                analysis.grpcCode(),
                                analysis.rootCause());
            }
            return analysis.retryableTransportFailure();
        } catch (RuntimeException | Error ignored) {
            // An unreadable cause must not replace the original RPC failure.
            return false;
        }
    }

    private <R> CompletableFuture<R> executeGrpcCallAsync(
            String ip,
            int port,
            Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
            long requestTimeoutMs,
            ServiceType serviceType) {
        CompletableFuture<R> resultFuture = new CompletableFuture<>();
        Context.CancellableContext operationContext = Context.current().withCancellation();
        resultFuture.whenComplete((ignored, failure) -> operationContext.cancel(failure));
        long startTime = System.nanoTime();
        Deadline deadline = Deadline.after(requestTimeoutMs, TimeUnit.MILLISECONDS);
        GrpcChannelPool.PooledChannel invoker = null;
        String phase = "CHANNEL_SETUP";
        GrpcCallTrace trace = null;

        try {
            trace = new GrpcCallTrace(transportObserver);
            final GrpcCallTrace callTrace = trace;
            EngineChannelKey channelKey = new EngineChannelKey(ip, port, serviceType);
            invoker = channelPool.getOrCreate(channelKey);

            invoker.markUsed();
            final GrpcChannelPool.PooledChannel finalInvoker = invoker;
            phase = "CALL_START";
            GrpcFutureStubWrapper stubWrapper =
                    createFutureStub(finalInvoker.getChannel())
                            .withDeadline(deadline)
                            .withTrace(callTrace);

            ListenableFuture<R> listenableFuture =
                    startInContext(operationContext, grpcCall, stubWrapper);

            phase = "CALLBACK_REGISTRATION";
            Futures.addCallback(
                    listenableFuture,
                    new FutureCallback<R>() {
                        @Override
                        public void onSuccess(R response) {
                            if (resultFuture.isDone()) {
                                return;
                            }
                            callTrace.markCallbackStarted();
                            try {
                                reportSuccessfulCall(
                                        serviceType, startTime, false, ip, port, callTrace);
                                reportDeliveryDelay(serviceType, false, callTrace, ip, port);
                            } finally {
                                resultFuture.complete(response);
                            }
                        }

                        @Override
                        public void onFailure(Throwable t) {
                            if (resultFuture.isDone()) {
                                return;
                            }
                            callTrace.markCallbackStarted();
                            try {
                                if (!resultFuture.isDone()
                                        && shouldRetry(serviceType, t, callTrace)) {
                                    finalInvoker.markExpired();
                                    long connectionDuration =
                                            finalInvoker.getConnectionDurationUs();
                                    reportConnectionDuration(
                                            serviceType, connectionDuration, ip, port);
                                    retryWithNewChannelAsync(
                                            channelKey,
                                            finalInvoker,
                                            grpcCall,
                                            operationContext,
                                            deadline,
                                            ip,
                                            port,
                                            serviceType,
                                            resultFuture,
                                            startTime,
                                            t);
                                } else {
                                    completeFailure(
                                            resultFuture,
                                            t,
                                            null,
                                            serviceType,
                                            false,
                                            "CALL_RESULT",
                                            finalInvoker,
                                            callTrace,
                                            ip,
                                            port);
                                }
                            } catch (RuntimeException | Error failure) {
                                // This callback runs after executeGrpcCallAsync has returned. Its
                                // failures
                                // must complete the same future rather than escape into the gRPC
                                // executor.
                                completeFailure(
                                        resultFuture,
                                        failure,
                                        t,
                                        serviceType,
                                        false,
                                        "CALL_RESULT",
                                        finalInvoker,
                                        callTrace,
                                        ip,
                                        port);
                            }
                        }
                    },
                    Runnable::run);

        } catch (RuntimeException | Error failure) {
            completeFailure(
                    resultFuture,
                    failure,
                    null,
                    serviceType,
                    false,
                    phase,
                    invoker,
                    trace,
                    ip,
                    port);
        }

        return resultFuture;
    }

    private <R> void retryWithNewChannelAsync(
            EngineChannelKey channelKey,
            GrpcChannelPool.PooledChannel staleInvoker,
            Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
            Context operationContext,
            Deadline deadline,
            String ip,
            int port,
            ServiceType serviceType,
            CompletableFuture<R> resultFuture,
            long originalStartTime,
            Throwable firstFailure) {
        if (resultFuture.isDone()) {
            return;
        }
        if (deadline.isExpired() || operationContext.isCancelled()) {
            Status status =
                    operationContext.isCancelled()
                            ? Contexts.statusFromCancelled(operationContext)
                            : Status.DEADLINE_EXCEEDED.withDescription(
                                    "RPC deadline expired before retry");
            completeFailure(
                    resultFuture,
                    status.asRuntimeException(),
                    firstFailure,
                    serviceType,
                    true,
                    "RETRY_DEADLINE",
                    staleInvoker,
                    null,
                    ip,
                    port);
            return;
        }
        GrpcChannelPool.PooledChannel invoker = null;
        String phase = "CHANNEL_SETUP";
        GrpcCallTrace trace = null;
        try {
            trace = new GrpcCallTrace(transportObserver);
            final GrpcCallTrace callTrace = trace;
            invoker = channelPool.replace(channelKey, staleInvoker);
            final GrpcChannelPool.PooledChannel newInvoker = invoker;
            if (resultFuture.isDone()) {
                return;
            }

            debugSafely(
                    "Retrying async gRPC call with new channel for {}:{} {}",
                    ip,
                    port,
                    serviceType);

            phase = "CALL_START";
            GrpcFutureStubWrapper stubWrapper =
                    createFutureStub(newInvoker.getChannel())
                            .withDeadline(deadline)
                            .withTrace(callTrace);

            ListenableFuture<R> listenableFuture =
                    startInContext(operationContext, grpcCall, stubWrapper);

            phase = "CALLBACK_REGISTRATION";
            Futures.addCallback(
                    listenableFuture,
                    new FutureCallback<R>() {
                        @Override
                        public void onSuccess(R response) {
                            if (resultFuture.isDone()) {
                                return;
                            }
                            callTrace.markCallbackStarted();
                            try {
                                reportSuccessfulCall(
                                        serviceType, originalStartTime, true, ip, port, callTrace);
                                reportDeliveryDelay(serviceType, true, callTrace, ip, port);
                            } finally {
                                resultFuture.complete(response);
                            }
                        }

                        @Override
                        public void onFailure(Throwable t) {
                            if (resultFuture.isDone()) {
                                return;
                            }
                            callTrace.markCallbackStarted();
                            completeFailure(
                                    resultFuture,
                                    t,
                                    firstFailure,
                                    serviceType,
                                    true,
                                    "CALL_RESULT",
                                    newInvoker,
                                    callTrace,
                                    ip,
                                    port);
                        }
                    },
                    Runnable::run);

        } catch (RuntimeException | Error failure) {
            completeFailure(
                    resultFuture,
                    failure,
                    firstFailure,
                    serviceType,
                    true,
                    phase,
                    invoker,
                    trace,
                    ip,
                    port);
        }
    }

    private static <R> ListenableFuture<R> startInContext(
            Context context,
            Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
            GrpcFutureStubWrapper stub) {
        // One owner covers both attempts, including cancellation while the stub
        // is starting and has not yet returned its future. gRPC propagates this
        // context to the transport and checks it before starting a cancelled call.
        Context previous = context.attach();
        try {
            return grpcCall.apply(stub);
        } finally {
            context.detach(previous);
        }
    }

    private void completeFailure(
            CompletableFuture<?> resultFuture,
            Throwable failure,
            Throwable firstFailure,
            ServiceType serviceType,
            boolean retry,
            String phase,
            GrpcChannelPool.PooledChannel invoker,
            GrpcCallTrace trace,
            String ip,
            int port) {
        try {
            try {
                if (firstFailure != null && firstFailure != failure) {
                    failure.addSuppressed(firstFailure);
                }
            } catch (RuntimeException | Error ignored) {
                // Attaching an earlier cause is best effort under memory pressure.
            }
            if (trace != null) {
                trace.markCallbackStarted();
            }
            reportCallFailure(serviceType, retry, phase, failure, invoker, trace, ip, port);
            reportDeliveryDelay(serviceType, retry, trace, ip, port);
        } finally {
            resultFuture.completeExceptionally(failure);
        }
    }

    private static void debugSafely(String format, Object... arguments) {
        try {
            Logger.debug(format, arguments);
        } catch (RuntimeException | Error ignored) {
            // Diagnostics must not prevent an RPC from starting or reaching its terminal result.
        }
    }

    private void reportSuccessfulCall(
            ServiceType serviceType,
            long originalStartTime,
            boolean retry,
            String ip,
            int port,
            GrpcCallTrace trace) {
        try {
            long duration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - originalStartTime);
            grpcReporter.reportCallMetrics(serviceType.getOperationName(), duration, retry);
        } catch (RuntimeException | Error failure) {
            // Metrics cannot change the response or prevent Future completion.
            logMetricsFailure(MetricOperation.CALL_METRICS, serviceType, ip, port, failure);
        }
    }

    private void reportCallFailure(
            ServiceType serviceType,
            boolean retry,
            String phase,
            Throwable cause,
            GrpcChannelPool.PooledChannel invoker,
            GrpcCallTrace trace,
            String ip,
            int port) {
        try {
            GrpcFailure.Analysis failure = GrpcFailure.analyze(cause);
            GrpcCallTrace.Snapshot transport =
                    trace == null ? GrpcCallTrace.Snapshot.EMPTY : trace.snapshot();
            GrpcChannelPool.CloseObservation localClose =
                    invoker == null ? null : invoker.getCloseObservation();
            debugSafely(
                    "event=engine_grpc_call_failed service={} endpoint={}:{} retry={} phase={} "
                            + "reason={} origin={} grpcCode={} rootType={} channelKey={} createTimeUs={} "
                            + "localCloseReason={} localCloseForced={} localCloseAgeMs={} transportTrace={}",
                    serviceType.getSuffix(),
                    ip,
                    port,
                    retry,
                    phase,
                    failure.kind(),
                    transport.failureOrigin(failure),
                    failure.grpcCode(),
                    failure.rootCause().getClass().getName(),
                    invoker == null ? "NONE" : invoker.getDescription(),
                    invoker == null ? -1L : invoker.getCreateTimeUs(),
                    localClose == null ? "NONE" : localClose.reason(),
                    localClose != null && localClose.forced(),
                    localClose == null
                            ? -1L
                            : TimeUnit.NANOSECONDS.toMillis(
                                    System.nanoTime() - localClose.nanoTime()),
                    transport,
                    cause);
            grpcReporter.reportCallFailure(
                    serviceType.getSuffix(), retry, phase, failure, transport);
        } catch (RuntimeException | Error diagnosticFailure) {
            logMetricsFailure(
                    MetricOperation.CALL_FAILURE, serviceType, ip, port, diagnosticFailure);
        }
    }

    private void reportDeliveryDelay(
            ServiceType serviceType, boolean retry, GrpcCallTrace trace, String ip, int port) {
        if (trace == null) {
            return;
        }
        try {
            grpcReporter.reportDeliveryDelay(serviceType.getSuffix(), retry, trace.snapshot());
        } catch (RuntimeException | Error failure) {
            logMetricsFailure(MetricOperation.DELIVERY_DELAY, serviceType, ip, port, failure);
        }
    }

    private void reportConnectionDuration(
            ServiceType serviceType, long duration, String ip, int port) {
        try {
            grpcReporter.reportConnectionDuration(serviceType.getOperationName(), duration);
        } catch (RuntimeException | Error failure) {
            logMetricsFailure(MetricOperation.CONNECTION_DURATION, serviceType, ip, port, failure);
        }
    }

    /**
     * At most one warning per failure stage every 30 seconds; successful calls touch no limiter state.
     */
    private void logMetricsFailure(
            MetricOperation operation,
            ServiceType serviceType,
            String ip,
            int port,
            Throwable failure) {
        long now = System.nanoTime();
        int slot = operation.ordinal();
        long previous = lastMetricsWarningNanos.get(slot);
        if ((previous == 0L || now - previous >= METRICS_WARNING_INTERVAL_NANOS)
                && lastMetricsWarningNanos.compareAndSet(slot, previous, now)) {
            try {
                String message = failure.getMessage();
                String summary =
                        message == null
                                ? "UNKNOWN"
                                : message.substring(
                                                0,
                                                Math.min(
                                                        message.length(),
                                                        MAX_METRICS_ERROR_MESSAGE_CHARS))
                                        .replace('\n', ' ')
                                        .replace('\r', ' ');
                Logger.warn(
                        "event=grpc_metrics_failed stage={} service={} endpoint={}:{} errorType={} error={}",
                        operation,
                        serviceType.getOperationName(),
                        ip,
                        port,
                        failure.getClass().getSimpleName(),
                        summary);
            } catch (RuntimeException | Error loggingFailure) {
                // Failure reporting must not prevent the RPC future from completing.
            }
        }
    }

    static boolean retriesBrokenConnections(ServiceType serviceType) {
        return serviceType != ServiceType.BATCH_ENQUEUE && serviceType != ServiceType.ENGINE_CANCEL;
    }

    /**
     * Queries worker status asynchronously, recreating a broken connection and retrying once.
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(
            String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
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
            String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
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
            String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
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
            String ip, int port, EngineRpcService.CacheVersionPB request, long requestTimeoutMs) {
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
            String ip, int port, EngineRpcService.EnqueueBatchRequestPB request) {
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

    record GrpcFutureStubWrapper(
            RpcServiceGrpc.RpcServiceFutureStub rpcServiceStub,
            MultimodalRpcServiceGrpc.MultimodalRpcServiceFutureStub multimodalRpcServiceStub) {

        GrpcFutureStubWrapper withDeadline(Deadline deadline) {
            return new GrpcFutureStubWrapper(
                    rpcServiceStub.withDeadline(deadline),
                    multimodalRpcServiceStub.withDeadline(deadline));
        }

        GrpcFutureStubWrapper withTrace(GrpcCallTrace trace) {
            return new GrpcFutureStubWrapper(
                    rpcServiceStub.withOption(GrpcCallTrace.CALL_TRACE, trace),
                    multimodalRpcServiceStub.withOption(GrpcCallTrace.CALL_TRACE, trace));
        }

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
