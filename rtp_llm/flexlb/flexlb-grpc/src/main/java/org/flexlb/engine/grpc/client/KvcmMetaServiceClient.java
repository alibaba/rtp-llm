package org.flexlb.engine.grpc.client;

import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ClientInterceptors;
import io.grpc.MethodDescriptor;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcChannelPool;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcRuntimeMetrics;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.flexlb.kvcm.grpc.ReactorMetaServiceGrpc;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.Collection;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Low-level KVCM MetaService gRPC client backed by a shared channel pool.
 */
@Component
public class KvcmMetaServiceClient {

    private final GrpcChannelPool<GrpcTarget> channelPool;
    private final GrpcRuntimeMetrics grpcRuntimeMetrics;

    public KvcmMetaServiceClient(GrpcChannelFactory channelFactory, GrpcRuntimeMetrics grpcRuntimeMetrics) {
        this.channelPool = new GrpcChannelPool<>(channelFactory::create);
        this.grpcRuntimeMetrics = grpcRuntimeMetrics;
    }

    public GetClusterInfoResponse getClusterInfo(
            GrpcTarget target,
            GetClusterInfoRequest request,
            long requestTimeoutMs) {
        GrpcChannelPool.PooledChannel pooledChannel = channelPool.getOrCreate(target);
        pooledChannel.markUsed();
        return MetaServiceGrpc.newBlockingStub(pooledChannel.getChannel())
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS)
                .getClusterInfo(request);
    }

    /**
     * Queries KVCM host cache state through a cold, cancellable Reactor publisher.
     *
     * @return a publisher that applies the supplied deadline when subscribed
     */
    public Mono<GetHostCacheStateResponse> getHostCacheState(
            GrpcTarget target,
            GetHostCacheStateRequest request,
            long requestTimeoutMs) {
        return executeReactiveGrpcCall(target, request, requestTimeoutMs, "GetHostCacheState",
                (channel, deadlineMs, grpcRequest) -> ReactorMetaServiceGrpc.newReactorStub(channel)
                        .withDeadlineAfter(deadlineMs, TimeUnit.MILLISECONDS)
                        .getHostCacheState(Mono.just(grpcRequest)));
    }

    public void removeStaleChannels(Collection<GrpcTarget> activeTargets) {
        channelPool.removeStaleChannels(activeTargets);
    }

    public void shutdown() {
        channelPool.shutdown();
    }

    private <RequestT, ResponseT> Mono<ResponseT> executeReactiveGrpcCall(
            GrpcTarget target,
            RequestT request,
            long requestTimeoutMs,
            String service,
            ReactiveGrpcCall<RequestT, ResponseT> grpcCall) {
        return Mono.defer(() -> {
            GrpcChannelPool.PooledChannel pooledChannel = channelPool.getOrCreate(target);
            pooledChannel.markUsed();
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
            GrpcRuntimeMetrics.GrpcCallHandle callHandle =
                    grpcRuntimeMetrics.recordCallStarted("kvcm", service, requestTimeoutMs);
            return Mono.defer(() -> grpcCall.invoke(cancellableChannel, requestTimeoutMs, request))
                    .doOnCancel(() -> {
                        ClientCall<?, ?> call = clientCall.get();
                        if (call != null) {
                            call.cancel("subscriber cancelled", null);
                        }
                    })
                    .doFinally(signalType -> grpcRuntimeMetrics.recordCallCompleted(callHandle));
        });
    }

    @FunctionalInterface
    private interface ReactiveGrpcCall<RequestT, ResponseT> {

        Mono<ResponseT> invoke(Channel channel, long requestTimeoutMs, RequestT request);
    }
}
