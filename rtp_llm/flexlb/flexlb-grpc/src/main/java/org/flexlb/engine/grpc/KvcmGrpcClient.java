package org.flexlb.engine.grpc;

import io.grpc.Deadline;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.flexlb.config.ConfigService;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.kvcm.grpc.*;
import org.springframework.stereotype.Component;
import javax.annotation.PreDestroy;
import java.util.*;
import java.util.concurrent.TimeUnit;

/** MU MetaService contract, consumed by the existing DSV cache lookup boundary. */
@Component
public final class KvcmGrpcClient implements AutoCloseable {
    private final KvcmCacheMatchingConfig options;
    private final KvcmConfig topology;
    private final ServiceDiscovery discovery;
    private final Map<String, ManagedChannel> channels = new LinkedHashMap<>();
    private volatile String leader;
    private volatile long refreshAt;

    @org.springframework.beans.factory.annotation.Autowired
    public KvcmGrpcClient(ConfigService config, ModelMetaConfig model, ServiceDiscovery discovery) {
        this(config.loadBalanceConfig().getCacheMatching() instanceof KvcmCacheMatchingConfig kvcm ? kvcm : null,
                model.getServiceRoute().getKvcm(), discovery);
    }

    public KvcmGrpcClient(KvcmCacheMatchingConfig options, KvcmConfig topology, ServiceDiscovery discovery) {
        this.options = options;
        this.topology = topology;
        this.discovery = discovery;
        if (options != null && (topology == null || topology.getAddress() == null || topology.getAddress().isBlank()
                || topology.getNamespace() == null || topology.getNamespace().isBlank()
                || topology.getPort() < 1 || topology.getPort() > 65535)) {
            throw new IllegalArgumentException("KVCM requires MODEL_SERVICE_CONFIG.kvcm address, port and namespace");
        }
    }

    public boolean enabled() { return options != null; }

    public Map<String, Integer> findMatchingEngines(List<Long> keys, long blockSize, List<String> candidates) {
        if (!enabled()) throw new IllegalStateException("KVCM is disabled");
        if (blockSize <= 0) throw new IllegalArgumentException("KVCM requires the routed cache block size");
        String instance = topology.getNamespace() + "_" + blockSize;
        Deadline deadline = Deadline.after(options.getRequestTimeoutMs(), TimeUnit.MILLISECONDS);
        try {
            String target = leader;
            if (target == null || System.nanoTime() >= refreshAt) target = refresh(instance, deadline);
            var response = MetaServiceGrpc.newBlockingStub(channel(target)).withDeadline(deadline)
                    .getHostCacheState(GetHostCacheStateRequest.newBuilder().setInstanceId(instance)
                            .setQueryType(QueryType.QT_PREFIX_MATCH).addAllBlockCacheKeys(keys)
                            .addAllMedium(options.getMedium()).build());
            requireOk(response.getHeader());
            Set<String> live = new HashSet<>(candidates);
            Map<String, Integer> matches = new HashMap<>();
            for (var host : response.getHostsList()) {
                String address = host.getHostIpPort();
                // This DSV branch exposes one engine per endpoint, so only @0 is unambiguous.
                if (!live.contains(address) && address.endsWith("@0")) address = address.substring(0, address.length() - 2);
                if (live.contains(address)) {
                    long count = host.getLocal();
                    if (count < 0 || count > keys.size()) throw new IllegalStateException("Invalid KVCM local match count");
                    matches.merge(address, Math.toIntExact(count), Math::max);
                }
            }
            return Map.copyOf(matches);
        } catch (RuntimeException failure) {
            refreshAt = 0;
            throw failure;
        }
    }

    private synchronized String refresh(String instance, Deadline deadline) {
        if (leader != null && System.nanoTime() < refreshAt) return leader;
        var hosts = discovery.getHosts(topology.getAddress());
        String host = hosts.isEmpty() ? topology.getAddress() : hosts.get(0).getIp();
        String bootstrap = host + ":" + topology.getPort();
        var cluster = MetaServiceGrpc.newBlockingStub(channel(bootstrap)).withDeadline(deadline)
                .getClusterInfo(GetClusterInfoRequest.newBuilder().setInstanceId(instance).build());
        requireOk(cluster.getHeader());
        var endpoint = cluster.getLeaderEndpoint();
        if (endpoint.getHost().isBlank() || endpoint.getMetaRpcPort() < 1 || endpoint.getMetaRpcPort() > 65535)
            throw new IllegalStateException("KVCM returned no usable leader");
        leader = endpoint.getHost() + ":" + endpoint.getMetaRpcPort();
        refreshAt = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(options.getLeaderRefreshIntervalMs());
        // Keep only bootstrap/current leader channels; repeated failovers cannot grow the pool.
        channels.entrySet().removeIf(entry -> {
            if (entry.getKey().equals(bootstrap) || entry.getKey().equals(leader)) return false;
            entry.getValue().shutdown(); return true;
        });
        return leader;
    }

    private static void requireOk(CommonResponseHeader header) {
        if (header.getStatus().getCode() != ErrorCode.OK)
            throw new IllegalStateException("KVCM " + header.getStatus().getCode() + ": " + header.getStatus().getMessage());
    }

    private synchronized ManagedChannel channel(String address) {
        return channels.computeIfAbsent(address, key -> ManagedChannelBuilder.forTarget(key).usePlaintext().build());
    }

    @Override @PreDestroy public synchronized void close() {
        channels.values().forEach(ManagedChannel::shutdownNow); channels.clear();
    }
}
