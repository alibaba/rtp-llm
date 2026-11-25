package org.flexlb.service.monitor;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/**
 * Discovers shared DashServing gRPC listeners without blocking the timing collector.
 * ModelServiceConfiguration initializes each provider, including FRONTEND, before this bean is used.
 * Configured service entries are read once at construction; their live instance lists continue refreshing.
 * No private listener is installed in the shared discovery provider.
 */
@Component
public class RequestTimingDiscovery {
    private static final Logger log = LoggerFactory.getLogger(RequestTimingDiscovery.class);
    private static final long REFRESH_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(2);
    private static final List<RoleType> TIMING_ROLES = List.of(RoleType.FRONTEND, RoleType.PREFILL, RoleType.PDFUSION);

    private final List<EndpointRole> configuredEndpoints;
    private final ServiceDiscovery serviceDiscovery;
    private final ExecutorService executor;
    private final LongSupplier nanoTime;
    // The lock protects only endpoint snapshots and task references; provider IO stays outside it.
    private final Map<EndpointRole, EndpointRefresh> endpoints = new LinkedHashMap<>();
    private boolean closed;

    @Autowired
    public RequestTimingDiscovery(ModelMetaConfig modelMetaConfig, ServiceDiscovery serviceDiscovery) {
        this(modelMetaConfig, serviceDiscovery,
                Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("request-timing-discovery-", 0).factory()),
                System::nanoTime);
    }

    RequestTimingDiscovery(ModelMetaConfig modelMetaConfig,
                           ServiceDiscovery serviceDiscovery,
                           ExecutorService executor,
                           LongSupplier nanoTime) {
        this.serviceDiscovery = serviceDiscovery;
        this.executor = executor;
        this.nanoTime = nanoTime;
        Set<EndpointRole> configured = new LinkedHashSet<>();
        for (RoleType role : TIMING_ROLES) {
            for (var endpoint : modelMetaConfig.endpointsWithGroup(role)) {
                if (endpoint.getRight() != null) {
                    configured.add(
                            new EndpointRole(endpoint.getRight(), role == RoleType.FRONTEND ? "frontend" : "prefill"));
                }
            }
        }
        this.configuredEndpoints = List.copyOf(configured);
    }

    /**
     * Returns the latest per-endpoint snapshots and schedules due refreshes without waiting for IO.
     * A successful empty discovery removes that endpoint's hosts; a failed refresh keeps its last snapshot.
     */
    public synchronized Set<RequestTimingSettings.Source> sources() {
        if (closed) {
            return Set.of();
        }
        long now = nanoTime.getAsLong();
        Set<RequestTimingSettings.Source> sources = new LinkedHashSet<>();
        for (EndpointRole key : configuredEndpoints) {
            EndpointRefresh refresh = endpoints.computeIfAbsent(key, ignored -> new EndpointRefresh());
            sources.addAll(refresh.sources);
            if (refresh.task == null && now >= refresh.nextRefreshNanos) {
                refresh.nextRefreshNanos = now + REFRESH_INTERVAL_NANOS;
                try {
                    refresh.task = executor.submit(() -> refresh(key, refresh));
                } catch (RejectedExecutionException failure) {
                    log.warn("Timing discovery refresh rejected, address={}", key.endpoint().getAddress());
                }
            }
        }
        return Set.copyOf(sources);
    }

    private void refresh(EndpointRole key, EndpointRefresh refresh) {
        Set<RequestTimingSettings.Source> sources = null;
        try {
            List<WorkerHost> hosts = serviceDiscovery.getHosts(key.endpoint());
            // Null means the provider has not supplied a snapshot, not a successful removal.
            if (hosts != null) {
                Set<RequestTimingSettings.Source> resolved = new LinkedHashSet<>();
                for (WorkerHost host : hosts) {
                    if (host == null || host.getIp() == null || host.getIp().isBlank() || host.getGrpcPort() < 1
                            || host.getGrpcPort() > 65535) {
                        continue;
                    }
                    // All engine indices behind a daemon share grpcPort. workerStatusPort is per-engine.
                    resolved.add(new RequestTimingSettings.Source(new GrpcTarget(host.getIp(), host.getGrpcPort()),
                            key.role()));
                }
                if (hosts.isEmpty() || !resolved.isEmpty()) {
                    sources = Set.copyOf(resolved);
                }
            }
        } catch (Exception failure) {
            log.warn("Timing discovery refresh failed, address={}", key.endpoint().getAddress(), failure);
        } finally {
            complete(key, refresh, sources);
        }
    }

    private synchronized void complete(EndpointRole key,
                                       EndpointRefresh refresh,
                                       Set<RequestTimingSettings.Source> sources) {
        if (closed || endpoints.get(key) != refresh) {
            return;
        }
        if (sources == null) {
            refresh.emptySnapshots = 0;
        } else if (sources.isEmpty() && !refresh.sources.isEmpty() && key.endpoint().getDiscovery() != null
                && key.endpoint().getDiscovery().getType() == ServiceDiscoveryType.VIPSERVER) {
            // VipServer maps lookup failures to an empty list. Require two real
            // refresh rounds before treating it as scale-to-zero, keeping cursors through one failure.
            refresh.emptySnapshots++;
            if (refresh.emptySnapshots >= 2) {
                refresh.sources = sources;
            }
        } else {
            refresh.emptySnapshots = 0;
            refresh.sources = sources;
        }
        refresh.task = null;
    }

    /**
     * Drops snapshots on leadership or configuration changes; old tasks cannot repopulate them.
     */
    public synchronized void reset() {
        endpoints.values().forEach(RequestTimingDiscovery::cancel);
        endpoints.clear();
    }

    @PreDestroy
    public synchronized void shutdown() {
        closed = true;
        reset();
        executor.shutdownNow();
    }

    private static void cancel(EndpointRefresh refresh) {
        if (refresh.task != null) {
            refresh.task.cancel(true);
        }
    }

    private record EndpointRole(Endpoint endpoint, String role) {
    }

    private static final class EndpointRefresh {
        private Set<RequestTimingSettings.Source> sources = Set.of();
        private Future<?> task;
        private long nextRefreshNanos = Long.MIN_VALUE;
        private int emptySnapshots;
    }
}
