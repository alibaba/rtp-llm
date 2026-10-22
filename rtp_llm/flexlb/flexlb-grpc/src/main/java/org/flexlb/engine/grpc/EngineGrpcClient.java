package org.flexlb.engine.grpc;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import io.grpc.ManagedChannel;
import io.grpc.Deadline;
import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Status;
import io.grpc.internal.GrpcUtil;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.buffer.PooledByteBufAllocator;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.WriteBufferWaterMark;
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
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.function.Function;

/**
 * Engine gRPC client for worker status queries
 */
@Component
public class EngineGrpcClient extends AbstractGrpcClient {

    private static final int DEFAULT_CONNECT_TIMEOUT_MILLIS = 20;
    public static final String CONNECT_TIMEOUT_PROPERTY =
            "flexlb.engine-grpc.connect-timeout-ms";
    public static final String VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY =
            "flexlb.engine-grpc.vit-cache-max-inbound-message-bytes";
    private static final int DEFAULT_MAX_INBOUND_MESSAGE_BYTES = 8 * 1024 * 1024;
    private static final int DEFAULT_VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES = 16 * 1024 * 1024;

    @Value("${" + VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY + ":"
            + DEFAULT_VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES + "}")
    private int vitCacheMaxInboundMessageBytes = DEFAULT_VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES;

    @Getter
    private final Executor executor;
    @Getter
    private final EventLoopGroup eventLoopGroup;
    public static final String ENQUEUE_TIMEOUT_PROPERTY =
            "flexlb.engine-grpc.enqueue-timeout-ms";
    private static final long DEFAULT_ENQUEUE_TIMEOUT_MILLIS = 5000L;
    private final int connectTimeoutMillis;
    private final long enqueueTimeoutMillis;

    private enum MetricOperation { CALL_METRICS, CONNECTION_DURATION, CALL_FAILURE, DELIVERY_DELAY }
    private final GrpcTransportObserver transportObserver = new GrpcTransportObserver();
    private static final int MAX_METRICS_ERROR_MESSAGE_CHARS = 256;
    private static final long METRICS_WARNING_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(30);
    private final AtomicLongArray lastMetricsWarningNanos = new AtomicLongArray(MetricOperation.values().length);

    @Autowired
    public EngineGrpcClient(CustomNameResolver nameResolver,
                            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
                            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoopGroup,
                            GrpcReporter grpcReporter,
                            @Value("${" + CONNECT_TIMEOUT_PROPERTY + ":"
                                    + DEFAULT_CONNECT_TIMEOUT_MILLIS + "}")
                            int connectTimeoutMillis,
                            @Value("${" + ENQUEUE_TIMEOUT_PROPERTY + ":"
                                    + DEFAULT_ENQUEUE_TIMEOUT_MILLIS + "}")
                            long enqueueTimeoutMillis) {
        super(grpcReporter);
        if (enqueueTimeoutMillis <= 0L) {
            throw new IllegalArgumentException("enqueueTimeoutMillis must be positive");
        }
        this.enqueueTimeoutMillis = enqueueTimeoutMillis;
        if (connectTimeoutMillis <= 0) {
            throw new IllegalArgumentException(
                    "connectTimeoutMillis must be positive");
        }
        this.executor = executor;
        this.eventLoopGroup = eventLoopGroup;
        this.connectTimeoutMillis = connectTimeoutMillis;
        nameResolver.start(this);
    }

    static boolean retriesBrokenConnections(ServiceType serviceType) {
        return serviceType != ServiceType.BATCH_ENQUEUE
                && serviceType != ServiceType.ENGINE_CANCEL;
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
                analysis = new GrpcFailure.Analysis(
                        transport.transportCauseKind(), analysis.grpcCode(), analysis.rootCause());
            }
            return analysis.retryableTransportFailure();
        } catch (RuntimeException | Error ignored) {
            // An unreadable cause must not replace the original RPC failure.
            return false;
        }
    }

    private <R> CompletableFuture<R> executeGrpcCallAsync(String ip, int port,
                                                           Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
                                                           long requestTimeoutMs,
                                                           ServiceType serviceType) {
        CompletableFuture<R> resultFuture = new CompletableFuture<>();
        Context.CancellableContext operationContext = Context.current().withCancellation();
        resultFuture.whenComplete((ignored, failure) -> operationContext.cancel(failure));
        long startTime = System.nanoTime();
        Deadline deadline = Deadline.after(requestTimeoutMs, TimeUnit.MILLISECONDS);
        Invoker invoker = null;
        String phase = "CHANNEL_SETUP";
        GrpcCallTrace trace = null;

        try {
            trace = new GrpcCallTrace(transportObserver);
            final GrpcCallTrace callTrace = trace;
            String channelKey = createKey(ip, port, serviceType);
            invoker = getInvoker(channelKey);

            if (invoker == null) {
                debugSafely("ip:{} {} grpc channel not found, creating and adding to pool", ip, serviceType);
                ManagedChannel newChannel = createChannel(channelKey);
                invoker = putInvokerIfAbsent(channelKey, newChannel);
            } else if (invoker.getChannel().isShutdown() || invoker.getChannel().isTerminated()) {
                debugSafely("ip:{} {} grpc channel is shutdown or terminated, recreating and updating pool", ip, serviceType);
                ManagedChannel newChannel = createChannel(channelKey);
                invoker = replaceInvoker(channelKey, invoker, newChannel);
            }

            invoker.updateLastUsedTime();
            final Invoker finalInvoker = invoker;
            phase = "CALL_START";
            GrpcFutureStubWrapper stubWrapper = new GrpcFutureStubWrapper(
                    RpcServiceGrpc.newFutureStub(finalInvoker.getChannel()),
                    MultimodalRpcServiceGrpc.newFutureStub(finalInvoker.getChannel())
            ).withDeadline(deadline).withTrace(callTrace);

            ListenableFuture<R> listenableFuture = startInContext(operationContext, grpcCall, stubWrapper);

            phase = "CALLBACK_REGISTRATION";
            Futures.addCallback(listenableFuture, new FutureCallback<R>() {
                @Override
                public void onSuccess(R response) {
                    if (resultFuture.isDone()) { return; }
                    callTrace.markCallbackStarted();
                    try {
                        reportSuccessfulCall(serviceType, startTime, false, ip, port);
                        reportDeliveryDelay(serviceType, false, callTrace, ip, port);
                    } finally {
                        resultFuture.complete(response);
                    }
                }

                @Override
                public void onFailure(Throwable t) {
                    if (resultFuture.isDone()) { return; }
                    callTrace.markCallbackStarted();
                    try {
                        if (!resultFuture.isDone() && shouldRetry(serviceType, t, callTrace)) {
                            finalInvoker.markExpired();
                            long connectionDuration = finalInvoker.getConnectionDuration();
                            reportConnectionDuration(serviceType, connectionDuration, ip, port);
                            retryWithNewChannelAsync(channelKey, finalInvoker, grpcCall, operationContext, deadline,
                                    ip, port, serviceType, resultFuture, startTime, t);
                        } else {
                            completeFailure(resultFuture, t, null, serviceType, false, "CALL_RESULT",
                                    finalInvoker, callTrace, ip, port);
                        }
                    } catch (RuntimeException | Error failure) {
                        // This callback runs after executeGrpcCallAsync has returned. Its failures
                        // must complete the same future rather than escape into the gRPC executor.
                        completeFailure(resultFuture, failure, t, serviceType, false, "CALL_RESULT",
                                finalInvoker, callTrace, ip, port);
                    }
                }
            }, Runnable::run);

        } catch (RuntimeException | Error failure) {
            completeFailure(resultFuture, failure, null, serviceType, false, phase, invoker, trace, ip, port);
        }

        return resultFuture;
    }

    private <R> void retryWithNewChannelAsync(String channelKey,
                                               Invoker staleInvoker,
                                               Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
                                               Context operationContext,
                                               Deadline deadline,
                                               String ip, int port,
                                               ServiceType serviceType,
                                               CompletableFuture<R> resultFuture,
                                               long originalStartTime,
                                               Throwable firstFailure) {
        if (resultFuture.isDone()) {
            return;
        }
        if (deadline.isExpired() || operationContext.isCancelled()) {
            Status status = operationContext.isCancelled() ? Contexts.statusFromCancelled(operationContext)
                    : Status.DEADLINE_EXCEEDED.withDescription("RPC deadline expired before retry");
            completeFailure(resultFuture,
                    status.asRuntimeException(),
                    firstFailure, serviceType, true, "RETRY_DEADLINE", staleInvoker, null, ip, port);
            return;
        }
        Invoker invoker = null;
        String phase = "CHANNEL_SETUP";
        GrpcCallTrace trace = null;
        try {
            trace = new GrpcCallTrace(transportObserver);
            final GrpcCallTrace callTrace = trace;
            ManagedChannel newChannel = createChannel(channelKey);
            invoker = replaceInvoker(channelKey, staleInvoker, newChannel);
            final Invoker newInvoker = invoker;
            if (resultFuture.isDone()) {
                return;
            }

            debugSafely("Retrying async gRPC call with new channel for {}:{} {}", ip, port, serviceType);

            phase = "CALL_START";
            GrpcFutureStubWrapper stubWrapper = new GrpcFutureStubWrapper(
                    RpcServiceGrpc.newFutureStub(newInvoker.getChannel()),
                    MultimodalRpcServiceGrpc.newFutureStub(newInvoker.getChannel())
            ).withDeadline(deadline).withTrace(callTrace);

            ListenableFuture<R> listenableFuture = startInContext(operationContext, grpcCall, stubWrapper);

            phase = "CALLBACK_REGISTRATION";
            Futures.addCallback(listenableFuture, new FutureCallback<R>() {
                @Override
                public void onSuccess(R response) {
                    if (resultFuture.isDone()) { return; }
                    callTrace.markCallbackStarted();
                    try {
                        reportSuccessfulCall(serviceType, originalStartTime, true, ip, port);
                        reportDeliveryDelay(serviceType, true, callTrace, ip, port);
                    } finally {
                        resultFuture.complete(response);
                    }
                }

                @Override
                public void onFailure(Throwable t) {
                    if (resultFuture.isDone()) { return; }
                    callTrace.markCallbackStarted();
                    completeFailure(resultFuture, t, firstFailure, serviceType, true, "CALL_RESULT",
                            newInvoker, callTrace, ip, port);
                }
            }, Runnable::run);

        } catch (RuntimeException | Error failure) {
            completeFailure(resultFuture, failure, firstFailure, serviceType, true, phase, invoker, trace, ip, port);
        }
    }

    private static <R> ListenableFuture<R> startInContext(
            Context context, Function<GrpcFutureStubWrapper, ListenableFuture<R>> grpcCall,
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

    private void completeFailure(CompletableFuture<?> resultFuture, Throwable failure, Throwable firstFailure,
                                 ServiceType serviceType, boolean retry, String phase,
                                 Invoker invoker, GrpcCallTrace trace, String ip, int port) {
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

    private void reportSuccessfulCall(ServiceType serviceType,
                                      long originalStartTime, boolean retry, String ip, int port) {
        try {
            long duration = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - originalStartTime);
            grpcReporter.reportCallMetrics(serviceType.getOperationName(), duration, retry);
        } catch (RuntimeException | Error failure) {
            // Metrics cannot change the response or prevent Future completion.
            logMetricsFailure(MetricOperation.CALL_METRICS, serviceType, ip, port, failure);
        }
    }

    private void reportCallFailure(ServiceType serviceType, boolean retry, String phase,
                                   Throwable cause, Invoker invoker, GrpcCallTrace trace, String ip, int port) {
        try {
            GrpcFailure.Analysis failure = GrpcFailure.analyze(cause);
            GrpcCallTrace.Snapshot transport = trace == null ? GrpcCallTrace.Snapshot.EMPTY : trace.snapshot();
            CloseObservation localClose = invoker == null ? null : invoker.getCloseObservation();
            debugSafely("event=engine_grpc_call_failed service={} endpoint={}:{} retry={} phase={} "
                            + "reason={} origin={} grpcCode={} rootType={} channelKey={} createTimeUs={} "
                            + "localCloseReason={} localCloseForced={} localCloseAgeMs={} transportTrace={}",
                    serviceType.getSuffix(), ip, port, retry, phase, failure.kind(),
                    transport.failureOrigin(failure), failure.grpcCode(),
                    failure.rootCause().getClass().getName(), invoker == null ? "NONE" : invoker.getChannelKey(),
                    invoker == null ? -1L : invoker.getCreateTime(), localClose == null ? "NONE" : localClose.reason(),
                    localClose != null && localClose.forced(),
                    localClose == null ? -1L : TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - localClose.nanoTime()),
                    transport, cause);
            grpcReporter.reportCallFailure(serviceType.getSuffix(), retry, phase, failure, transport);
        } catch (RuntimeException | Error diagnosticFailure) {
            logMetricsFailure(MetricOperation.CALL_FAILURE, serviceType, ip, port, diagnosticFailure);
        }
    }

    private void reportDeliveryDelay(ServiceType serviceType, boolean retry, GrpcCallTrace trace,
                                     String ip, int port) {
        if (trace == null) {
            return;
        }
        try {
            grpcReporter.reportDeliveryDelay(serviceType.getSuffix(), retry, trace.snapshot());
        } catch (RuntimeException | Error failure) {
            logMetricsFailure(MetricOperation.DELIVERY_DELAY, serviceType, ip, port, failure);
        }
    }

    private void reportConnectionDuration(ServiceType serviceType, long duration, String ip, int port) {
        try {
            grpcReporter.reportConnectionDuration(serviceType.getOperationName(), duration);
        } catch (RuntimeException | Error failure) {
            logMetricsFailure(MetricOperation.CONNECTION_DURATION, serviceType, ip, port, failure);
        }
    }

    /** At most one warning per failure stage every 30 seconds; successful calls touch no limiter state. */
    private void logMetricsFailure(MetricOperation operation, ServiceType serviceType,
                                   String ip, int port, Throwable failure) {
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
            } catch (RuntimeException | Error loggingFailure) {
                // Failure reporting must not prevent the RPC future from completing.
            }
        }
    }

    /**
     * Get worker status via gRPC (async)
     */
    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(String ip, int port, EngineRpcService.StatusVersionPB request, long requestTimeoutMs) {
        return executeGrpcCallAsync(ip, port,
                stub -> stub.getRpcServiceFutureStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.WORKER_STATUS);
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
        return executeGrpcCallAsync(ip, port,
                stub -> stub.getMultimodalFutureStub().getWorkerStatus(request), requestTimeoutMs, ServiceType.MULTIMODAL_WORKER_STATUS);
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
    public CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> batchEnqueueAsync(
            String ip, int port, EngineRpcService.EnqueueBatchRequestPB request) {
        return batchEnqueueAsync(ip, port, request, enqueueTimeoutMillis);
    }

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
        int maxInboundMessageBytes = ServiceType.MULTIMODAL_CACHE_STATUS.getSuffix().equals(parts[2])
                ? vitCacheMaxInboundMessageBytes : DEFAULT_MAX_INBOUND_MESSAGE_BYTES;
        Logger.info("Creating new channel for ip: {}, port: {}", ip, port);
        return NettyChannelBuilder.forAddress(ip, port)
                .channelFactory(transportObserver::newChannel)
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
                // ViT directory snapshots have a separate configurable message limit.
                .maxInboundMessageSize(maxInboundMessageBytes)
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
                .intercept(GrpcCallTrace.INTERCEPTOR)
                .disableRetry()
                .build();
    }

}
