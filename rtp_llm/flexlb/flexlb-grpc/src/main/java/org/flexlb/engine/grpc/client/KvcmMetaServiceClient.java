package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcChannelPool;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.concurrent.TimeUnit;

/**
 * Low-level KVCM MetaService gRPC client backed by a shared channel pool.
 */
@Component
public class KvcmMetaServiceClient {

    private final GrpcChannelPool<GrpcTarget> channelPool;

    public KvcmMetaServiceClient(GrpcChannelFactory channelFactory) {
        this.channelPool = new GrpcChannelPool<>(channelFactory::create);
    }

    public GetClusterInfoResponse getClusterInfo(
            GrpcTarget target,
            GetClusterInfoRequest request,
            long requestTimeoutMs) {
        return MetaServiceGrpc.newBlockingStub(channelFor(target))
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS)
                .getClusterInfo(request);
    }

    public GetHostCacheStateResponse getHostCacheState(
            GrpcTarget target,
            GetHostCacheStateRequest request,
            long requestTimeoutMs) {
        return MetaServiceGrpc.newBlockingStub(channelFor(target))
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS)
                .getHostCacheState(request);
    }

    public void removeStaleChannels(Collection<GrpcTarget> activeTargets) {
        channelPool.removeStaleChannels(activeTargets);
    }

    public void shutdown() {
        channelPool.shutdown();
    }

    private ManagedChannel channelFor(GrpcTarget target) {
        return channelPool.getOrCreate(target).getChannel();
    }
}
