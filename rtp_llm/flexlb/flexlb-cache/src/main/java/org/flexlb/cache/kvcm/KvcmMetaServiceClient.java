package org.flexlb.cache.kvcm;

import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import org.flexlb.cache.kvcm.protocol.GetClusterInfoRequest;
import org.flexlb.cache.kvcm.protocol.GetClusterInfoResponse;
import org.flexlb.cache.kvcm.protocol.GetHostCacheStateRequest;
import org.flexlb.cache.kvcm.protocol.GetHostCacheStateResponse;
import org.flexlb.cache.kvcm.protocol.MetaServiceGrpc;
import org.flexlb.util.network.GrpcChannelFactory;
import org.flexlb.util.network.GrpcChannelPool;
import org.flexlb.util.network.GrpcTarget;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
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
            GrpcTarget target, GetClusterInfoRequest request, long requestTimeoutMs) {
        return MetaServiceGrpc.newBlockingStub(channelFor(target))
                .withDeadlineAfter(requestTimeoutMs, TimeUnit.MILLISECONDS)
                .getClusterInfo(request);
    }

    public GetHostCacheStateResponse getHostCacheState(
            GrpcTarget target, GetHostCacheStateRequest request, Deadline queryDeadline) {
        return MetaServiceGrpc.newBlockingStub(channelFor(target))
                .withDeadline(queryDeadline)
                .getHostCacheState(request);
    }

    public void removeStaleChannels(Collection<GrpcTarget> activeTargets) {
        channelPool.removeStaleChannels(activeTargets);
    }

    @PreDestroy
    public void shutdown() {
        channelPool.shutdown();
    }

    private ManagedChannel channelFor(GrpcTarget target) {
        return channelPool.getOrCreate(target).getChannel();
    }
}
