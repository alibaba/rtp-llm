package org.flexlb.engine.grpc.client;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Discovers and caches the current KVCM leader endpoint.
 */
@Slf4j
@Component
public class KvcmLeaderResolver {

    private final boolean enabled;
    private final KvcmConfig config;
    private final Endpoint kvcmEndpoint;
    private final RoutingServiceDiscovery serviceDiscovery;
    private final KvcmMetaServiceClient metaServiceClient;
    private final AtomicReference<GrpcTarget> leader = new AtomicReference<>();

    public KvcmLeaderResolver(
            ModelMetaConfig modelMetaConfig,
            RoutingServiceDiscovery serviceDiscovery,
            KvcmMetaServiceClient metaServiceClient) {
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoutes().stream().findFirst().orElse(null);
        this.config = serviceRoute != null ? serviceRoute.getKvcm() : null;
        this.enabled = config != null && config.isEnabled();
        this.kvcmEndpoint = enabled ? config.toEndpoint() : null;
        this.serviceDiscovery = serviceDiscovery;
        this.metaServiceClient = metaServiceClient;
    }

    public GrpcTarget resolve() {
        return leader.get();
    }

    public void refresh() {
        if (!enabled) {
            return;
        }

        List<WorkerHost> discoveredHosts = serviceDiscovery.getHosts(kvcmEndpoint);
        Set<GrpcTarget> bootstrapTargets = new LinkedHashSet<>();
        for (WorkerHost discoveredHost : discoveredHosts) {
            // Discovery supplies candidate IPs; the configured port is used only for GetClusterInfo.
            bootstrapTargets.add(new GrpcTarget(discoveredHost.getIp(), config.getPort()));
        }
        for (GrpcTarget bootstrapTarget : bootstrapTargets) {
            try {
                GetClusterInfoResponse response = metaServiceClient.getClusterInfo(
                        bootstrapTarget,
                        GetClusterInfoRequest.newBuilder()
                                .setTraceId(IdUtils.fastUuid())
                                .build(),
                        config.getRequestTimeoutMs());
                ErrorCode code = response.getHeader().getStatus().getCode();
                if (code != ErrorCode.OK || !response.hasLeaderEndpoint()) {
                    log.warn("KVCM bootstrap target {} did not return a leader, code={}",
                            bootstrapTarget, code);
                    continue;
                }
                MetaNodeEndpoint endpoint = response.getLeaderEndpoint();
                if (StringUtils.isBlank(endpoint.getHost())) {
                    continue;
                }
                int leaderPort = endpoint.getMetaRpcPort();
                if (leaderPort <= 0) {
                    log.warn("KVCM bootstrap target {} returned an invalid leader meta RPC port: {}",
                            bootstrapTarget, leaderPort);
                    continue;
                }

                GrpcTarget newLeader = new GrpcTarget(endpoint.getHost(), leaderPort);
                GrpcTarget previousLeader = leader.get();
                if (!newLeader.equals(previousLeader)) {
                    leader.set(newLeader);
                    log.info("KVCM leader changed from {} to {}", previousLeader, newLeader);
                }
                Set<GrpcTarget> activeTargets = new HashSet<>(bootstrapTargets);
                activeTargets.add(newLeader);
                metaServiceClient.removeStaleChannels(activeTargets);
                return;
            } catch (Exception e) {
                log.warn("Failed to query KVCM cluster info from bootstrap target: {}", bootstrapTarget, e);
            }
        }
    }
}
