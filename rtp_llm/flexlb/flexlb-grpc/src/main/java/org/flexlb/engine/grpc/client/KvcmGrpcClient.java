package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcChannelPool;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetClusterInfoResponse;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.MetaNodeEndpoint;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * KVCM gRPC client with cached leader and role-specific cache namespaces.
 */
@Slf4j
@Component
public class KvcmGrpcClient {

    private final boolean enabled;
    private final ServiceRoute serviceRoute;
    private final KvcmConfig config;
    private final Endpoint kvcmEndpoint;
    private final RoutingServiceDiscovery serviceDiscovery;
    private final ScheduledExecutorService refreshExecutor;
    private final GrpcChannelPool<GrpcTarget> channelPool;
    private final AtomicReference<GrpcTarget> leader = new AtomicReference<>();
    private final ConcurrentMap<String, ConcurrentMap<RoleType, String>> namespaceByGroupAndRole =
            new ConcurrentHashMap<>();
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();

    public KvcmGrpcClient(
            ModelMetaConfig modelMetaConfig,
            RoutingServiceDiscovery serviceDiscovery,
            GrpcChannelFactory channelFactory) {
        this.serviceDiscovery = serviceDiscovery;
        this.channelPool = new GrpcChannelPool<>(channelFactory::create);
        this.serviceRoute = modelMetaConfig.getServiceRoutes().stream().findFirst().orElse(null);
        this.config = serviceRoute != null ? serviceRoute.getKvcm() : null;
        this.enabled = config != null && config.isEnabled();
        this.kvcmEndpoint = enabled ? config.toEndpoint() : null;

        if (!enabled) {
            this.refreshExecutor = null;
            return;
        }

        this.refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "kvcm-metadata-refresher");
            thread.setDaemon(true);
            return thread;
        });
        this.refreshExecutor.scheduleWithFixedDelay(
                this::refreshLeaderSafely,
                0,
                config.getLeaderRefreshIntervalMs(),
                TimeUnit.MILLISECONDS);
        log.info("Started KVCM client, address={}, bootstrapPort={}, leaderRefreshIntervalMs={}",
                config.getAddress(), config.getPort(), config.getLeaderRefreshIntervalMs());
    }

    public Map<String, Integer> findMatchingEngines(
            List<Long> blockCacheKeys,
            RoleType roleType,
            String group) {
        if (!enabled) {
            log.warn("Skipping KVCM cache query because the KVCM client is disabled, role={}, group={}",
                    roleType, group);
            return Collections.emptyMap();
        }
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        ConcurrentMap<RoleType, String> namespaceByRole =
                namespaceByGroupAndRole.get(StringUtils.defaultString(group));
        String namespace = namespaceByRole == null ? null : namespaceByRole.get(roleType);
        if (StringUtils.isBlank(namespace)) {
            log.warn("Skipping KVCM cache query because namespace is unavailable, role={}, group={}",
                    roleType, group);
            requestImmediateRefresh();
            return Collections.emptyMap();
        }
        GrpcTarget currentLeader = leader.get();
        if (currentLeader == null) {
            log.warn("Skipping KVCM cache query because leader is unavailable, role={}, group={}",
                    roleType, group);
            requestImmediateRefresh();
            return Collections.emptyMap();
        }

        GetHostCacheStateRequest request = GetHostCacheStateRequest.newBuilder()
                .setTraceId(IdUtils.fastUuid())
                // KVCM exposes the cache namespace as instance_id in its protocol.
                .setInstanceId(namespace)
                .addAllBlockCacheKeys(blockCacheKeys)
                .build();

        try {
            GetHostCacheStateResponse response = MetaServiceGrpc.newBlockingStub(channelFor(currentLeader))
                    .withDeadlineAfter(config.getRequestTimeoutMs(), TimeUnit.MILLISECONDS)
                    .getHostCacheState(request);
            ErrorCode code = response.getHeader().getStatus().getCode();
            if (code != ErrorCode.OK) {
                if (code == ErrorCode.SERVER_NOT_LEADER || code == ErrorCode.INSTANCE_NOT_EXIST) {
                    requestImmediateRefresh();
                }
                throw new IllegalStateException(
                        "KVCM GetHostCacheState failed, code=" + code
                                + ", message=" + response.getHeader().getStatus().getMessage());
            }
            return toPrefixMatchBlocksByHost(response.getHostsList());
        } catch (StatusRuntimeException e) {
            requestImmediateRefresh();
            throw e;
        }
    }

    private void refreshLeaderSafely() {
        try {
            refreshLeader();
        } catch (Exception e) {
            log.warn("Failed to refresh KVCM leader; keeping the last known leader", e);
        }
        try {
            refreshNamespaceMap();
        } catch (Exception e) {
            log.warn("Failed to refresh KVCM namespace map; keeping the last known values", e);
        }
    }

    private void refreshLeader() {
        List<WorkerHost> discoveredHosts = serviceDiscovery.getHosts(kvcmEndpoint);
        Set<GrpcTarget> bootstrapTargets = new HashSet<>();
        for (WorkerHost discoveredHost : discoveredHosts) {
            // Discovery supplies candidate IPs; the configured bootstrap port is used only for GetClusterInfo.
            bootstrapTargets.add(new GrpcTarget(discoveredHost.getIp(), config.getPort()));
        }
        for (GrpcTarget bootstrapTarget : bootstrapTargets) {
            try {
                GetClusterInfoResponse response = MetaServiceGrpc.newBlockingStub(channelFor(bootstrapTarget))
                        .withDeadlineAfter(config.getRequestTimeoutMs(), TimeUnit.MILLISECONDS)
                        .getClusterInfo(GetClusterInfoRequest.newBuilder()
                                .setTraceId(IdUtils.fastUuid())
                                .build());
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
                // ClusterInfo is authoritative for the leader's real gRPC host and port.
                GrpcTarget newLeader = new GrpcTarget(endpoint.getHost(), leaderPort);
                GrpcTarget previousLeader = leader.get();
                if (!newLeader.equals(previousLeader)) {
                    leader.set(newLeader);
                    log.info("KVCM leader changed from {} to {}", previousLeader, newLeader);
                }
                Set<GrpcTarget> activeChannelTargets = new HashSet<>(bootstrapTargets);
                activeChannelTargets.add(newLeader);
                channelPool.removeStaleChannels(activeChannelTargets);
                return;
            } catch (Exception e) {
                log.warn("Failed to query KVCM cluster info from bootstrap target: {}", bootstrapTarget, e);
            }
        }
    }

    private void refreshNamespaceMap() {
        boolean changed = false;

        for (RoleType roleType : serviceRoute.getAllRoleTypes()) {
            for (var endpointWithGroup : serviceRoute.getAllEndpointsWithGroup(roleType)) {
                Endpoint endpoint = endpointWithGroup.getRight();
                if (endpoint == null) {
                    continue;
                }
                List<WorkerHost> hosts = serviceDiscovery.getHosts(endpoint);
                if (hosts.isEmpty()) {
                    continue;
                }

                String namespace = hosts.getFirst().getDeploymentName();
                if (StringUtils.isBlank(namespace)) {
                    continue;
                }

                String group = StringUtils.defaultString(endpointWithGroup.getLeft());
                ConcurrentMap<RoleType, String> namespaceByRole =
                        namespaceByGroupAndRole.computeIfAbsent(group, ignored -> new ConcurrentHashMap<>());
                String currentNamespace = namespaceByRole.get(roleType);
                if (!namespace.equals(currentNamespace)) {
                    namespaceByRole.put(roleType, namespace);
                    changed = true;
                }
            }
        }

        if (changed) {
            log.info("Updated KVCM role namespaces: {}", namespaceByGroupAndRole);
        }
    }

    private Map<String, Integer> toPrefixMatchBlocksByHost(List<HostCacheMatch> matches) {
        Map<String, Integer> result = new HashMap<>();
        for (HostCacheMatch match : matches) {
            if (StringUtils.isBlank(match.getHostIpPort())) {
                continue;
            }
            int prefixMatchBlocks = Math.toIntExact(match.getPrefixMatchBlocks());
            result.merge(match.getHostIpPort(), prefixMatchBlocks, Math::max);
        }
        return result;
    }

    private ManagedChannel channelFor(GrpcTarget target) {
        return channelPool.getOrCreate(target).getChannel();
    }

    private void requestImmediateRefresh() {
        if (refreshExecutor == null || refreshExecutor.isShutdown()
                || !immediateRefreshQueued.compareAndSet(false, true)) {
            return;
        }
        try {
            refreshExecutor.execute(() -> {
                try {
                    refreshLeaderSafely();
                } finally {
                    immediateRefreshQueued.set(false);
                }
            });
        } catch (RejectedExecutionException e) {
            immediateRefreshQueued.set(false);
        }
    }

    @PreDestroy
    public void shutdown() {
        if (refreshExecutor != null) {
            refreshExecutor.shutdown();
        }
        channelPool.shutdown();
    }

}
