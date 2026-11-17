package org.flexlb.engine.grpc;

import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.kvcm.grpc.CommonResponseHeader;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.util.IdUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Refreshes the KVCM leader off the routing thread and queries it within one deadline per route.
 * DSV uses the configured namespace and local matches for its single-engine endpoints.
 */
@Slf4j
@Component
public final class KvcmGrpcClient implements AutoCloseable {
    private static final long MIN_REFRESH_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(1);

    private final KvcmCacheMatchingConfig options;
    private final KvcmConfig topology;
    private final ServiceDiscovery discovery;
    private final ScheduledExecutorService refreshExecutor;
    private final Map<String, ManagedChannel> channels = new LinkedHashMap<>();
    private final AtomicBoolean refreshQueued = new AtomicBoolean();
    private final AtomicLong lastRequestedRefresh = new AtomicLong(System.nanoTime());

    private volatile ManagedChannel leader;

    private boolean closed;
    private int heartbeatFailures;
    private int queryFailures;
    private int recoverySuccesses;
    private boolean standby;

    @Autowired
    public KvcmGrpcClient(ConfigService config, ModelMetaConfig model, ServiceDiscovery discovery) {
        this(config.loadBalanceConfig().getCacheMatching() instanceof KvcmCacheMatchingConfig kvcm
                ? kvcm : null, model.getServiceRoute().getKvcm(), discovery);
    }

    public KvcmGrpcClient(KvcmCacheMatchingConfig options, KvcmConfig topology, ServiceDiscovery discovery) {
        this.options = options;
        this.topology = topology;
        this.discovery = discovery;
        if (options == null) {
            refreshExecutor = null;
            return;
        }
        if (topology == null || topology.getAddress() == null || topology.getAddress().isBlank()
                || topology.getNamespace() == null || topology.getNamespace().isBlank()
                || topology.getPort() < 1 || topology.getPort() > 65535) {
            throw new IllegalArgumentException("KVCM requires MODEL_SERVICE_CONFIG.kvcm address, port and namespace");
        }
        refreshExecutor = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread thread = new Thread(task, "kvcm-service-state-refresher");
            thread.setDaemon(true);
            return thread;
        });
        refreshExecutor.scheduleWithFixedDelay(this::refreshLeader, 0,
                options.getLeaderRefreshIntervalMs(), TimeUnit.MILLISECONDS);
    }

    public synchronized boolean useStandby() {
        return standby;
    }

    public Map<String, Integer> findMatchingEngines(List<Long> keys, long blockSize, List<String> candidates) {
        if (!enabled()) {
            throw new IllegalStateException("KVCM is disabled");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("KVCM requires the routed cache block size");
        }
        var request = GetHostCacheStateRequest.newBuilder()
                .setTraceId(IdUtils.fastUuid())
                .setInstanceId(topology.getNamespace() + "_" + blockSize)
                .setQueryType(QueryType.QT_PREFIX_MATCH)
                .addAllBlockCacheKeys(keys)
                .addAllMedium(options.getMedium())
                .setGlobalKvsHostCount(Math.max(0, options.getGlobalKvsHostCount()))
                .setEnableP2P(options.isEnableP2p())
                .build();
        Deadline deadline = Deadline.after(options.getRequestTimeoutMs(), TimeUnit.MILLISECONDS);
        for (int attempt = 0; ; attempt++) {
            try {
                ManagedChannel target = leader;
                if (target == null || deadline.isExpired()) {
                    throw new IllegalStateException("KVCM leader unavailable within the query budget");
                }
                Map<String, Integer> matches = query(target, request, deadline, candidates);
                recordQuery(true);
                return matches;
            } catch (RuntimeException failure) {
                requestLeaderRefresh();
                if (attempt >= options.getMaxQueryRetryCount() || deadline.isExpired() || leader == null) {
                    recordQuery(false);
                    throw failure;
                }
            }
        }
    }

    public boolean enabled() {
        return options != null;
    }

    @Override
    @PreDestroy
    public void close() {
        if (refreshExecutor == null) {
            return;
        }
        refreshExecutor.shutdownNow();
        synchronized (this) {
            closed = true;
            leader = null;
            channels.values().forEach(ManagedChannel::shutdownNow);
            channels.clear();
        }
    }

    private Map<String, Integer> query(ManagedChannel target, GetHostCacheStateRequest request,
                                       Deadline deadline, List<String> candidates) {
        var response = MetaServiceGrpc.newBlockingStub(target).withDeadline(deadline).getHostCacheState(request);
        requireOk(response.getHeader());
        Set<String> live = new HashSet<>(candidates);
        Map<String, Integer> matches = new HashMap<>();
        for (var host : response.getHostsList()) {
            String address = host.getHostIpPort();
            if (!live.contains(address) && address.endsWith("@0")) {
                address = address.substring(0, address.length() - 2);
            }
            if (live.contains(address)) {
                long count = host.getLocal();
                if (count < 0 || count > request.getBlockCacheKeysCount()) {
                    throw new IllegalStateException("Invalid KVCM local match count");
                }
                matches.merge(address, Math.toIntExact(count), Math::max);
            }
        }
        return Map.copyOf(matches);
    }

    private static void requireOk(CommonResponseHeader header) {
        if (header.getStatus().getCode() != ErrorCode.OK) {
            throw new IllegalStateException("KVCM " + header.getStatus().getCode() + ": " + header.getStatus().getMessage());
        }
    }

    private void requestLeaderRefresh() {
        long now = System.nanoTime();
        long previous = lastRequestedRefresh.get();
        if (now - previous < MIN_REFRESH_INTERVAL_NANOS
                || !lastRequestedRefresh.compareAndSet(previous, now)) {
            return;
        }
        try {
            if (refreshQueued.compareAndSet(false, true)) {
                refreshExecutor.execute(() -> {
                    try {
                        refreshLeader();
                    } finally {
                        refreshQueued.set(false);
                    }
                });
            }
        } catch (RejectedExecutionException ignored) {
            refreshQueued.set(false);
            // The client's close path has already stopped the refresher.
        }
    }

    private void refreshLeader() {
        try {
            var hosts = discovery.getHosts(topology.getAddress());
            Set<String> bootstrapTargets = new LinkedHashSet<>();
            for (var host : hosts) {
                bootstrapTargets.add(target(host.getIp(), topology.getPort()));
            }
            if (bootstrapTargets.isEmpty()) {
                bootstrapTargets.add(target(topology.getAddress(), topology.getPort()));
            }
            String discovered = null;
            for (String bootstrap : bootstrapTargets) {
                try {
                    var cluster = MetaServiceGrpc.newBlockingStub(channel(bootstrap))
                            .withDeadlineAfter(options.getRequestTimeoutMs(), TimeUnit.MILLISECONDS)
                            .getClusterInfo(GetClusterInfoRequest.newBuilder().setTraceId(IdUtils.fastUuid()).build());
                    requireOk(cluster.getHeader());
                    var endpoint = cluster.getLeaderEndpoint();
                    if (!endpoint.getHost().isBlank() && endpoint.getMetaRpcPort() > 0
                            && endpoint.getMetaRpcPort() <= 65535) {
                        discovered = target(endpoint.getHost(), endpoint.getMetaRpcPort());
                        break;
                    }
                } catch (RuntimeException failure) {
                    log.debug("KVCM bootstrap failed at {}", bootstrap, failure);
                }
            }
            publishLeader(bootstrapTargets, discovered);
        } catch (RuntimeException failure) {
            recordHeartbeat(false);
            log.warn("KVCM leader refresh failed", failure);
        }
    }

    private static String target(String host, int port) {
        return (host.contains(":") && !host.startsWith("[") ? "[" + host + "]" : host) + ":" + port;
    }

    private synchronized ManagedChannel channel(String address) {
        if (closed) {
            throw new IllegalStateException("KVCM client is closed");
        }
        return channels.computeIfAbsent(address,
                key -> ManagedChannelBuilder.forTarget(key).usePlaintext().build());
    }

    private synchronized void publishLeader(Set<String> bootstrapTargets, String discovered) {
        if (closed) {
            return;
        }
        if (discovered != null) {
            leader = channel(discovered);
        }
        channels.entrySet().removeIf(entry -> {
            if (bootstrapTargets.contains(entry.getKey()) || entry.getValue() == leader) {
                return false;
            }
            entry.getValue().shutdown();
            return true;
        });
        recordHeartbeat(discovered != null);
    }

    private synchronized void recordHeartbeat(boolean success) {
        if (success) {
            heartbeatFailures = 0;
            if (standby && ++recoverySuccesses >= options.getRecoverySuccessThreshold()) {
                standby = false;
                queryFailures = 0;
                recoverySuccesses = 0;
            }
        } else {
            recoverySuccesses = 0;
            if (++heartbeatFailures >= options.getHeartbeatFailureThreshold()) {
                standby = true;
            }
        }
    }

    private synchronized void recordQuery(boolean success) {
        if (success) {
            queryFailures = 0;
        } else if (++queryFailures >= options.getQueryFailureThreshold()) {
            standby = true;
            recoverySuccesses = 0;
        }
    }
}
