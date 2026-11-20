package org.flexlb.cache.kvcm;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.cache.kvcm.protocol.ErrorCode;
import org.flexlb.cache.kvcm.protocol.GetClusterInfoRequest;
import org.flexlb.cache.kvcm.protocol.GetClusterInfoResponse;
import org.flexlb.cache.kvcm.protocol.MetaNodeEndpoint;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.config.cache.KvcmConfig;
import org.flexlb.util.IdUtils;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.RoutingServiceDiscovery;
import org.flexlb.util.network.GrpcTarget;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Discovers and caches the current KVCM leader endpoint.
 *
 * <p>Resolve reads the currently known metadata-service target. Refresh consults service discovery
 * and metadata responses to update that target; callers can retry against the refreshed leader
 * without owning discovery state themselves.
 */
@Slf4j
@Component
public class KvcmLeaderResolver {

    private final boolean enabled;
    private final CacheMatchConfiguration configuration;
    private final KvcmConfig topologyConfig;
    private final DiscoveryEndpoint kvcmEndpoint;
    private final RoutingServiceDiscovery serviceDiscovery;
    private final KvcmMetaServiceClient metaServiceClient;
    private final AtomicReference<GrpcTarget> leader = new AtomicReference<>();

    public KvcmLeaderResolver(
            CacheMatchConfiguration configuration,
            RoutingServiceDiscovery serviceDiscovery,
            KvcmMetaServiceClient metaServiceClient) {
        this.configuration = configuration;
        this.topologyConfig = configuration.getKvcmConfig();
        this.enabled = configuration.isKvcmEnabled();
        this.kvcmEndpoint = enabled ? topologyConfig.toEndpoint() : null;
        this.serviceDiscovery = serviceDiscovery;
        this.metaServiceClient = metaServiceClient;
    }

    public GrpcTarget resolve() {
        return leader.get();
    }

    public boolean refresh() {
        if (!enabled) {
            return false;
        }

        List<DiscoveredHost> discoveredHosts = serviceDiscovery.getHosts(kvcmEndpoint);
        Set<GrpcTarget> bootstrapTargets = new LinkedHashSet<>();
        for (DiscoveredHost discoveredHost : discoveredHosts) {
            // Discovery supplies candidate IPs; the configured port is used only for
            // GetClusterInfo.
            bootstrapTargets.add(new GrpcTarget(discoveredHost.getIp(), topologyConfig.getPort()));
        }
        for (GrpcTarget bootstrapTarget : bootstrapTargets) {
            try {
                GetClusterInfoResponse response =
                        metaServiceClient.getClusterInfo(
                                bootstrapTarget,
                                GetClusterInfoRequest.newBuilder()
                                        .setTraceId(IdUtils.fastUuid())
                                        .build(),
                                configuration.getKvcmRuntimeConfig().getRequestTimeoutMs());
                ErrorCode code = response.getHeader().getStatus().getCode();
                if (code != ErrorCode.OK || !response.hasLeaderEndpoint()) {
                    log.warn(
                            "KVCM bootstrap target {} did not return a leader, code={}",
                            bootstrapTarget,
                            code);
                    continue;
                }
                MetaNodeEndpoint endpoint = response.getLeaderEndpoint();
                if (StringUtils.isBlank(endpoint.getHost())) {
                    continue;
                }
                int leaderPort = endpoint.getMetaRpcPort();
                if (leaderPort <= 0) {
                    log.warn(
                            "KVCM bootstrap target {} returned an invalid leader meta RPC port: {}",
                            bootstrapTarget,
                            leaderPort);
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
                return true;
            } catch (Exception e) {
                log.warn(
                        "Failed to query KVCM cluster info from bootstrap target: {}",
                        bootstrapTarget,
                        e);
            }
        }
        return false;
    }
}
