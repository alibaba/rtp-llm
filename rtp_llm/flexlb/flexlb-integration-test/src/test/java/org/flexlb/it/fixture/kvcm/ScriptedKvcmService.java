package org.flexlb.it.fixture.kvcm;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Scriptable KVCM metadata-service fake that records the keys FlexLB actually sends.
 *
 * <p>{@link KvcmIntegrationTestFixtures.CacheResponse#EMPTY} represents a valid cache miss and
 * must not trigger transport fallback. {@link KvcmIntegrationTestFixtures.CacheResponse#UNAVAILABLE}
 * models an RPC failure and drives the real KVCM failover manager.
 */
final class ScriptedKvcmService extends MetaServiceGrpc.MetaServiceImplBase {

    private static final int MATCH_ALL_REQUESTED_BLOCKS = -1;

    private final AtomicReference<KvcmIntegrationTestFixtures.CacheResponse> cacheResponse =
            new AtomicReference<>(KvcmIntegrationTestFixtures.CacheResponse.EMPTY);
    private final AtomicInteger clusterInfoCalls = new AtomicInteger();
    private final AtomicInteger cacheStateCalls = new AtomicInteger();
    private final AtomicInteger localMatchBlocks = new AtomicInteger(MATCH_ALL_REQUESTED_BLOCKS);
    private final AtomicReference<List<Long>> lastCacheBlockKeys = new AtomicReference<>(List.of());
    private volatile int leaderPort;
    private volatile String matchingWorkerHost;

    @Override
    public void getClusterInfo(
            GetClusterInfoRequest request,
            StreamObserver<GetClusterInfoResponse> responseObserver) {
        clusterInfoCalls.incrementAndGet();
        responseObserver.onNext(GetClusterInfoResponse.newBuilder()
                .setHeader(okHeader())
                .setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                        .setHost(IntegrationTestFixtures.WORKER_IP)
                        .setMetaRpcPort(leaderPort))
                .build());
        responseObserver.onCompleted();
    }

    @Override
    public void getHostCacheState(
            GetHostCacheStateRequest request,
            StreamObserver<GetHostCacheStateResponse> responseObserver) {
        cacheStateCalls.incrementAndGet();
        lastCacheBlockKeys.set(List.copyOf(request.getBlockCacheKeysList()));
        if (cacheResponse.get() == KvcmIntegrationTestFixtures.CacheResponse.UNAVAILABLE) {
            responseObserver.onError(Status.UNAVAILABLE
                    .withDescription("scripted KVCM cache query failure")
                    .asRuntimeException());
            return;
        }

        GetHostCacheStateResponse.Builder response = GetHostCacheStateResponse.newBuilder().setHeader(okHeader());
        if (cacheResponse.get() == KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH) {
            int requestedBlocks = request.getBlockCacheKeysCount();
            response.addHosts(HostCacheMatch.newBuilder()
                    .setHostIpPort(matchingWorkerHost)
                    .setLocal(localMatchBlocks(requestedBlocks)));
        }
        responseObserver.onNext(response.build());
        responseObserver.onCompleted();
    }

    void setLeaderPort(int leaderPort) {
        this.leaderPort = leaderPort;
    }

    void setMatchingWorkerHost(String matchingWorkerHost) {
        this.matchingWorkerHost = matchingWorkerHost;
    }

    void setCacheResponse(KvcmIntegrationTestFixtures.CacheResponse cacheResponse) {
        this.cacheResponse.set(cacheResponse);
    }

    /**
     * Limits a configured worker's local match to a number of request blocks.
     *
     * <p>Use {@value #MATCH_ALL_REQUESTED_BLOCKS} to restore the normal full-match response.
     */
    void setLocalMatchBlocks(int localMatchBlocks) {
        if (localMatchBlocks < MATCH_ALL_REQUESTED_BLOCKS) {
            throw new IllegalArgumentException("Local match blocks must be non-negative or -1 for all blocks");
        }
        this.localMatchBlocks.set(localMatchBlocks);
    }

    int cacheStateCalls() {
        return cacheStateCalls.get();
    }

    int clusterInfoCalls() {
        return clusterInfoCalls.get();
    }

    List<Long> lastCacheBlockKeys() {
        return lastCacheBlockKeys.get();
    }

    private int localMatchBlocks(int requestedBlocks) {
        int configuredBlocks = localMatchBlocks.get();
        return configuredBlocks == MATCH_ALL_REQUESTED_BLOCKS
                ? requestedBlocks
                : Math.min(configuredBlocks, requestedBlocks);
    }

    private CommonResponseHeader okHeader() {
        return CommonResponseHeader.newBuilder()
                .setStatus(org.flexlb.kvcm.grpc.Status.newBuilder().setCode(ErrorCode.OK))
                .build();
    }
}
