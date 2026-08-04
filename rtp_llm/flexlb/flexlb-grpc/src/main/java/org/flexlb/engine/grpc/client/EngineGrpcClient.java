package org.flexlb.engine.grpc.client;

import com.google.protobuf.MessageLite;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ClientInterceptors;
import io.grpc.ManagedChannel;
import io.grpc.MethodDescriptor;
import io.grpc.StatusRuntimeException;
import io.netty.channel.EventLoopGroup;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.ReactorRpcServiceGrpc;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcChannelPool;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.GrpcRuntimeMetrics;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import javax.annotation.PreDestroy;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

/**
 * Engine gRPC client for worker and cache status queries.
 */
@Component
public class EngineGrpcClient implements EngineAddressResolver.Listener {

    private final GrpcChannelFactory channelFactory;
    private final GrpcChannelPool<EngineChannelKey> channelPool;
    private final GrpcReporter grpcReporter;
    private final GrpcRuntimeMetrics grpcRuntimeMetrics;

    public EngineGrpcClient(
            EngineAddressResolver addressResolver,
            GrpcChannelFactory channelFactory,
            GrpcReporter grpcReporter,
            GrpcRuntimeMetrics grpcRuntimeMetrics) {
        this.channelFactory = channelFactory;
        this.channelPool = new GrpcChannelPool<>(key -> channelFactory.create(key.target()));
        this.grpcReporter = grpcReporter;
        this.grpcRuntimeMetrics = grpcRuntimeMetrics;
        addressResolver.subscribe(this);
    }

    @Override
    public void onAddressUpdate(List<String> ipPortList) {
        if (ipPortList == null) {
            Logger.error("received null ipPort list");
            return;
        }

        Set<EngineChannelKey> activeKeys = new HashSet<>();
        Set<String> activeIps = new HashSet<>();
        for (String ipPort : ipPortList) {
            String[] parts = ipPort.split(":");
            String ip = parts[0];
            activeIps.add(ip);
            int grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(parts[1]));
            for (ServiceType serviceType : ServiceType.values()) {
                activeKeys.add(new EngineChannelKey(ip, grpcPort, serviceType));
            }
        }

        Logger.info("address update, ip:port list size:{}, channel pool size:{}",
                ipPortList.size(), channelPool.size());
        for (EngineChannelKey key : activeKeys) {
            try {
                channelPool.getOrCreate(key);
            } catch (Exception e) {
                Logger.error("create channel for {} failed", key, e);
            }
        }
        channelPool.removeChannelsForInactiveGroups(activeIps, EngineChannelKey::ip);
    }

    private <R> R executeGrpcCall(
            String ip,
            int port,
            Function<GrpcStubWrapper, R> grpcCall,
            long requestTimeoutMs,
            ServiceType serviceType) {
        EngineChannelKey channelKey = new EngineChannelKey(ip, port, serviceType);
        GrpcChannelPool.PooledChannel pooledChannel = channelPool.getOrCreate(channelKey);

        try {
            return invoke(pooledChannel, grpcCall, requestTimeoutMs, ip, serviceType, false);
        } catch (StatusRuntimeException e) {
            if (isConnectionBrokenError(e)) {
                pooledChannel.markExpired();
                long connectionDuration = pooledChannel.getConnectionDurationUs();
                grpcReporter.reportConnectionDuration(
                        ip, serviceType.getOperationName(), connectionDuration);
                Logger.warn("Connection broken for {}:{} {}, duration: {}us, recreating channel and retrying once, msg:{}",
                        ip, port, serviceType, connectionDuration, e.getMessage());
                GrpcChannelPool.PooledChannel replacement =
                        channelPool.replace(channelKey, pooledChannel);
                return invoke(replacement, grpcCall, requestTimeoutMs, ip, serviceType, true);
            }
            Logger.error("Exception during {} gRPC call for {}:{}",
                    serviceType.getOperationName(), ip, port, e);
            throw e;
        } catch (Exception e) {
            Logger.error("Exception during {} gRPC call for {}:{}",
                    serviceType.getOperationName(), ip, port, e);
            throw e;
        }
    }

    private <R> R invoke(
            GrpcChannelPool.PooledChannel pooledChannel,
            Function<GrpcStubWrapper, R> grpcCall,
            long requestTimeoutMs,
            String ip,
            ServiceType serviceType,
            boolean retry) {
        pooledChannel.markUsed();
        GrpcStubWrapper stubWrapper = createStub(pooledChannel.getChannel())
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS);

        long startTime = System.nanoTime() / 1000;
        GrpcRuntimeMetrics.GrpcCallHandle callHandle =
                grpcRuntimeMetrics.recordCallStarted("engine", serviceType.getOperationName(), requestTimeoutMs);
        try {
            R response = grpcCall.apply(stubWrapper);
            long duration = System.nanoTime() / 1000 - startTime;
            int responseSize = response instanceof MessageLite messageLite
                    ? messageLite.getSerializedSize()
                    : 0;
            grpcReporter.reportCallMetrics(
                    ip, serviceType.getOperationName(), duration, responseSize, retry);
            return response;
        } finally {
            grpcRuntimeMetrics.recordCallCompleted(callHandle);
        }
    }

    private <RequestT, ResponseT extends MessageLite> Mono<ResponseT> executeReactiveGrpcCall(
            String ip,
            int port,
            RequestT request,
            long requestTimeoutMs,
            ServiceType serviceType,
            ReactiveGrpcCall<RequestT, ResponseT> grpcCall) {
        EngineChannelKey channelKey = new EngineChannelKey(ip, port, serviceType);
        return Mono.defer(() -> {
            GrpcChannelPool.PooledChannel pooledChannel = channelPool.getOrCreate(channelKey);
            return invokeReactiveGrpcCall(
                            pooledChannel, request, requestTimeoutMs, ip, serviceType, false, grpcCall)
                    .onErrorResume(StatusRuntimeException.class, error -> {
                        if (!isConnectionBrokenError(error)) {
                            return Mono.error(error);
                        }
                        pooledChannel.markExpired();
                        long connectionDuration = pooledChannel.getConnectionDurationUs();
                        grpcReporter.reportConnectionDuration(
                                ip, serviceType.getOperationName(), connectionDuration);
                        Logger.warn("Connection broken for {}:{} {}, duration: {}us, recreating channel and retrying once, msg:{}",
                                ip, port, serviceType, connectionDuration, error.getMessage());
                        return invokeReactiveGrpcCall(channelPool.replace(channelKey, pooledChannel), request,
                                requestTimeoutMs, ip, serviceType, true, grpcCall);
                    });
        }).doOnError(error -> Logger.error("Exception during {} gRPC call for {}:{}",
                serviceType.getOperationName(), ip, port, error));
    }

    private <RequestT, ResponseT extends MessageLite> Mono<ResponseT> invokeReactiveGrpcCall(
            GrpcChannelPool.PooledChannel pooledChannel,
            RequestT request,
            long requestTimeoutMs,
            String ip,
            ServiceType serviceType,
            boolean retry,
            ReactiveGrpcCall<RequestT, ResponseT> grpcCall) {
        return Mono.defer(() -> {
            pooledChannel.markUsed();
            long startTime = System.nanoTime() / 1_000;
            AtomicReference<ClientCall<?, ?>> clientCall = new AtomicReference<>();
            ClientInterceptor captureCall = new ClientInterceptor() {
                @Override
                public <CallRequestT, CallResponseT> ClientCall<CallRequestT, CallResponseT> interceptCall(
                        MethodDescriptor<CallRequestT, CallResponseT> method,
                        CallOptions callOptions,
                        Channel next) {
                    ClientCall<CallRequestT, CallResponseT> call = next.newCall(method, callOptions);
                    clientCall.set(call);
                    return call;
                }
            };
            Channel cancellableChannel = ClientInterceptors.intercept(pooledChannel.getChannel(), captureCall);
            String service = serviceType.getOperationName();
            GrpcRuntimeMetrics.GrpcCallHandle callHandle =
                    grpcRuntimeMetrics.recordCallStarted("engine", service, requestTimeoutMs);
            return Mono.defer(() -> grpcCall.invoke(cancellableChannel, requestTimeoutMs, request))
                    .doOnNext(response -> grpcReporter.reportCallMetrics(
                            ip,
                            service,
                            System.nanoTime() / 1_000 - startTime,
                            response.getSerializedSize(),
                            retry))
                    .doOnCancel(() -> {
                        ClientCall<?, ?> call = clientCall.get();
                        if (call != null) {
                            call.cancel("subscriber cancelled", null);
                        }
                    })
                    .doFinally(signalType -> grpcRuntimeMetrics.recordCallCompleted(callHandle));
        });
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

    public EngineRpcService.WorkerStatusPB getWorkerStatus(
            String ip,
            int port,
            EngineRpcService.StatusVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCall(
                ip,
                port,
                stub -> stub.rpcServiceStub().getWorkerStatus(request),
                requestTimeoutMs,
                ServiceType.WORKER_STATUS);
    }

    /**
     * Queries KV cache group metadata through a cold, cancellable Reactor publisher.
     *
     * @return a publisher that applies the supplied deadline when subscribed
     */
    public Mono<EngineRpcService.KvCacheGroupListPB> getKvCacheGroupsMetadata(
            String ip,
            int port,
            EngineRpcService.KvCacheGroupsRequestPB request,
            long requestTimeoutMs) {
        return executeReactiveGrpcCall(ip, port, request, requestTimeoutMs, ServiceType.KV_CACHE_GROUP_METADATA,
                (channel, deadlineMs, grpcRequest) ->
                        ReactorRpcServiceGrpc.newReactorStub(channel)
                                .withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS)
                                .getKvCacheGroupsMetadata(Mono.just(grpcRequest)));
    }

    public EngineRpcService.CacheStatusPB getCacheStatus(
            String ip,
            int port,
            EngineRpcService.CacheVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCall(
                ip,
                port,
                stub -> stub.rpcServiceStub().getCacheStatus(request),
                requestTimeoutMs,
                ServiceType.CACHE_STATUS);
    }

    public EngineRpcService.WorkerStatusPB getMultimodalWorkerStatus(
            String ip,
            int port,
            EngineRpcService.StatusVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCall(
                ip,
                port,
                stub -> stub.multimodalRpcServiceStub().getWorkerStatus(request),
                requestTimeoutMs,
                ServiceType.MULTIMODAL_WORKER_STATUS);
    }

    public EngineRpcService.CacheStatusPB getMultimodalCacheStatus(
            String ip,
            int port,
            EngineRpcService.CacheVersionPB request,
            long requestTimeoutMs) {
        return executeGrpcCall(
                ip,
                port,
                stub -> stub.multimodalRpcServiceStub().getCacheStatus(request),
                requestTimeoutMs,
                ServiceType.MULTIMODAL_CACHE_STATUS);
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
        channelPool.shutdown();
    }

    private GrpcStubWrapper createStub(ManagedChannel channel) {
        return new GrpcStubWrapper(
                RpcServiceGrpc.newBlockingStub(channel),
                MultimodalRpcServiceGrpc.newBlockingStub(channel));
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

    @FunctionalInterface
    private interface ReactiveGrpcCall<RequestT, ResponseT> {

        Mono<ResponseT> invoke(Channel channel, long requestTimeoutMs, RequestT request);
    }

    private record GrpcStubWrapper(RpcServiceGrpc.RpcServiceBlockingStub rpcServiceStub,
                                   MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub multimodalRpcServiceStub) {

        GrpcStubWrapper withDeadlineAfter(long timeout, TimeUnit unit) {
            return new GrpcStubWrapper(
                    rpcServiceStub.withDeadlineAfter(timeout, unit),
                    multimodalRpcServiceStub.withDeadlineAfter(timeout, unit));
        }
    }

    private enum ServiceType {
        WORKER_STATUS("worker", "GetWorkerStatus"),
        KV_CACHE_GROUP_METADATA("kv_cache_group_metadata", "GetKvCacheGroupsMetadata"),
        CACHE_STATUS("cache", "GetCacheStatus"),
        MULTIMODAL_WORKER_STATUS("multimodal_worker", "GetWorkerStatus"),
        MULTIMODAL_CACHE_STATUS("multimodal_cache", "GetCacheStatus");

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
    }
}
