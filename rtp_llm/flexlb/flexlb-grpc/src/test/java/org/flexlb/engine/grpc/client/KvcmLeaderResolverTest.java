package org.flexlb.engine.grpc.client;

import io.grpc.Status;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmLeaderResolverTest {

    @Test
    void refreshesFromNextBootstrapTargetAfterFailure() {
        RoutingServiceDiscovery serviceDiscovery = Mockito.mock(RoutingServiceDiscovery.class);
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        GrpcTarget failedTarget = new GrpcTarget("10.0.0.1", 6381);
        GrpcTarget healthyTarget = new GrpcTarget("10.0.0.2", 6381);
        GrpcTarget leaderTarget = new GrpcTarget("10.0.0.3", 6382);
        AtomicReference<Set<GrpcTarget>> activeTargets = new AtomicReference<>();
        when(serviceDiscovery.getHosts(any())).thenReturn(List.of(
                WorkerHost.of(failedTarget.host(), 8080),
                WorkerHost.of(healthyTarget.host(), 8080)));
        when(metaServiceClient.getClusterInfo(any(GrpcTarget.class), any(GetClusterInfoRequest.class), anyLong()))
                .thenAnswer(invocation -> {
                    GrpcTarget target = invocation.getArgument(0);
                    if (target.equals(failedTarget)) {
                        throw Status.UNAVAILABLE.asRuntimeException();
                    }
                    return clusterInfo(leaderTarget);
                });
        Mockito.doAnswer(invocation -> {
                    activeTargets.set(Set.copyOf(invocation.getArgument(0)));
                    return null;
                })
                .when(metaServiceClient)
                .removeStaleChannels(any());
        KvcmLeaderResolver resolver = new KvcmLeaderResolver(
                enabledConfiguration(), serviceDiscovery, metaServiceClient);

        assertTrue(resolver.refresh());
        assertEquals(leaderTarget, resolver.resolve());

        assertEquals(Set.of(failedTarget, healthyTarget, leaderTarget), activeTargets.get());
    }

    @Test
    void returnsFalseWhenAllBootstrapTargetsFail() {
        RoutingServiceDiscovery serviceDiscovery = Mockito.mock(RoutingServiceDiscovery.class);
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        GrpcTarget firstTarget = new GrpcTarget("10.0.0.1", 6381);
        GrpcTarget secondTarget = new GrpcTarget("10.0.0.2", 6381);
        when(serviceDiscovery.getHosts(any())).thenReturn(List.of(
                WorkerHost.of(firstTarget.host(), 8080),
                WorkerHost.of(secondTarget.host(), 8080)));
        when(metaServiceClient.getClusterInfo(any(GrpcTarget.class), any(GetClusterInfoRequest.class), anyLong()))
                .thenThrow(Status.UNAVAILABLE.asRuntimeException());
        KvcmLeaderResolver resolver = new KvcmLeaderResolver(
                enabledConfiguration(), serviceDiscovery, metaServiceClient);

        assertFalse(resolver.refresh());
        assertNull(resolver.resolve());
        verify(metaServiceClient, times(2)).getClusterInfo(
                any(GrpcTarget.class), any(GetClusterInfoRequest.class), anyLong());
    }

    @Test
    void skipsBootstrapResponsesWithoutUsableLeader() {
        RoutingServiceDiscovery serviceDiscovery = Mockito.mock(RoutingServiceDiscovery.class);
        KvcmMetaServiceClient metaServiceClient = Mockito.mock(KvcmMetaServiceClient.class);
        GrpcTarget nonOkTarget = new GrpcTarget("10.0.0.1", 6381);
        GrpcTarget blankLeaderTarget = new GrpcTarget("10.0.0.2", 6381);
        GrpcTarget healthyTarget = new GrpcTarget("10.0.0.3", 6381);
        GrpcTarget leaderTarget = new GrpcTarget("10.0.0.4", 6382);
        when(serviceDiscovery.getHosts(any())).thenReturn(List.of(
                WorkerHost.of(nonOkTarget.host(), 8080),
                WorkerHost.of(blankLeaderTarget.host(), 8080),
                WorkerHost.of(healthyTarget.host(), 8080)));
        when(metaServiceClient.getClusterInfo(any(GrpcTarget.class), any(GetClusterInfoRequest.class), anyLong()))
                .thenAnswer(invocation -> {
                    GrpcTarget target = invocation.getArgument(0);
                    if (target.equals(nonOkTarget)) {
                        return clusterInfo(ErrorCode.INTERNAL_ERROR, leaderTarget);
                    }
                    if (target.equals(blankLeaderTarget)) {
                        return clusterInfo(ErrorCode.OK, new GrpcTarget("", leaderTarget.port()));
                    }
                    return clusterInfo(ErrorCode.OK, leaderTarget);
                });
        KvcmLeaderResolver resolver = new KvcmLeaderResolver(
                enabledConfiguration(), serviceDiscovery, metaServiceClient);

        assertTrue(resolver.refresh());
        assertEquals(leaderTarget, resolver.resolve());
    }

    private static GetClusterInfoResponse clusterInfo(GrpcTarget leaderTarget) {
        return clusterInfo(ErrorCode.OK, leaderTarget);
    }

    private static GetClusterInfoResponse clusterInfo(ErrorCode code, GrpcTarget leaderTarget) {
        return GetClusterInfoResponse.newBuilder()
                .setHeader(CommonResponseHeader.newBuilder()
                        .setStatus(org.flexlb.kvcm.grpc.Status.newBuilder().setCode(code)))
                .setLeaderEndpoint(MetaNodeEndpoint.newBuilder()
                        .setHost(leaderTarget.host())
                        .setMetaRpcPort(leaderTarget.port()))
                .build();
    }

    private static CacheMatchConfiguration enabledConfiguration() {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(ServiceDiscoveryType.DASHSCOPE);
        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setAddress("v-kvcm");
        kvcm.setPort(6381);
        kvcm.setDiscovery(discovery);
        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        return new CacheMatchConfiguration(modelMetaConfig);
    }
}
