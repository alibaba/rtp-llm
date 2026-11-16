package org.flexlb.worker.status;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.config.service.WorkerDiscoveryEndpoint;
import org.flexlb.util.discovery.ServiceDiscovery;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;

/**
 * Resolves grouped worker hosts and caches the last nonempty discovery result. Empty results or
 * discovery failures do not silently discard that fallback.
 */
@Service("workerAddressService")
public class WorkerAddressService {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final long EMPTY_WORKER_WARNING_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);

    private final ServiceMetricsReporter serviceMetricsReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ServiceDiscovery serviceDiscovery;
    // Cache the last non-empty worker list per endpoint; its key fields stay unchanged after
    // initialization.
    private final ConcurrentMap<WorkerDiscoveryEndpoint, List<WorkerHost>>
            lastNonEmptyHostsByEndpoint = new ConcurrentHashMap<>();
    private final ConcurrentMap<WorkerDiscoveryEndpoint, WorkerAvailabilityLogState>
            workerAvailabilityByEndpoint = new ConcurrentHashMap<>();

    /**
     * Service discovery request thread pool
     */
    private final ThreadPoolExecutor serviceDiscoveryExecutor;

    public WorkerAddressService(
            ServiceMetricsReporter serviceMetricsReporter,
            ModelMetaConfig modelMetaConfig,
            ServiceDiscovery serviceDiscovery,
            ConfigService configService) {

        this.serviceMetricsReporter = serviceMetricsReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.serviceDiscovery = serviceDiscovery;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.serviceDiscoveryExecutor =
                new ThreadPoolExecutor(
                        10,
                        config.getInternalRuntime().getServiceDiscoveryMaxThreads(),
                        60L,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(1000),
                        new NamedThreadFactory("service-discovery-executor"),
                        new ThreadPoolExecutor.AbortPolicy());
    }

    @PreDestroy
    public void destroy() {
        serviceDiscoveryExecutor.shutdown();
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Pair<String, WorkerDiscoveryEndpoint>> endpoints =
                modelMetaConfig.endpointsWithGroup(modelName, modelEndpointType);
        if (endpoints.isEmpty()) {
            logger.info(
                    "modelName={} role={} service route not found", modelName, modelEndpointType);
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportRawServiceDiscoveryHostCount(
                                    modelName, modelEndpointType, 0));
            return workerHosts;
        }
        int discoveredHostCount = 0;
        boolean allEndpointsSucceeded = true;
        for (Pair<String, WorkerDiscoveryEndpoint> endpointTuple : endpoints) {
            WorkerDiscoveryEndpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info(
                        "modelName={} endpoint is null, endpointType={}",
                        modelName,
                        modelEndpointType);
                allEndpointsSucceeded = false;
                continue;
            }
            ServiceHosts result = getServiceHosts(modelName, endpoint);
            workerHosts.addAll(result.discoveredOrCachedHostsForStatusCheck());
            if (result.discoveredHostCount() == null) {
                allEndpointsSucceeded = false;
            } else {
                discoveredHostCount += result.discoveredHostCount();
            }
        }
        if (allEndpointsSucceeded) {
            int rawHostCount = discoveredHostCount;
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportRawServiceDiscoveryHostCount(
                                    modelName, modelEndpointType, rawHostCount));
        }
        if (!allEndpointsSucceeded && workerHosts.isEmpty()) {
            throw new IllegalStateException(
                    "Service discovery failed without a cached worker snapshot");
        }
        return workerHosts;
    }

    public static List<WorkerHost> expandWorkerHosts(
            List<org.flexlb.util.discovery.DiscoveredHost> discoveredHosts,
            WorkerDiscoveryEndpoint endpoint) {
        if (discoveredHosts == null || discoveredHosts.isEmpty()) {
            return List.of();
        }

        boolean grpcEndpoint =
                org.flexlb.util.network.BackendServiceProtocolEnum.GRPC
                        .getName()
                        .equals(endpoint.getProtocol());
        int multiEngineNum = endpoint.getMultiEngineNum();
        List<WorkerHost> normalizedHosts = new ArrayList<>(discoveredHosts.size() * multiEngineNum);
        for (org.flexlb.util.discovery.DiscoveredHost host : discoveredHosts) {
            int discoveredPort = host.getPort();
            int httpPort = grpcEndpoint ? discoveredPort - 1 : discoveredPort;
            int grpcPort = grpcEndpoint ? discoveredPort : discoveredPort + 1;
            int workerStatusBasePort =
                    endpoint.getWorkerStatusPort() == null
                            ? grpcPort
                            : endpoint.getWorkerStatusPort();
            requireValidPort(httpPort, "httpPort");
            requireValidPort(grpcPort, "grpcPort");
            requireValidPort(httpPort + 5, "httpServerPort");
            requireValidPort(workerStatusBasePort, "workerStatusPort");
            requireValidPort(
                    (long) workerStatusBasePort + multiEngineNum - 1, "lastWorkerStatusPort");
            for (int engineIndex = 0; engineIndex < multiEngineNum; engineIndex++) {
                normalizedHosts.add(
                        new WorkerHost(
                                host.getIp(),
                                httpPort,
                                grpcPort,
                                httpPort + 5,
                                workerStatusBasePort + engineIndex,
                                host.getSite(),
                                endpoint.getGroup(),
                                host.getDeploymentName(),
                                engineIndex,
                                multiEngineNum));
            }
        }
        return normalizedHosts;
    }

    @Scheduled(fixedRate = 2000)
    private void reportExecutorMetrics() {
        try {
            serviceMetricsReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "serviceDiscoveryExecutor",
                    serviceDiscoveryExecutor.getActiveCount(),
                    serviceDiscoveryExecutor.getQueue().size(),
                    serviceDiscoveryExecutor.getCorePoolSize(),
                    serviceDiscoveryExecutor.getPoolSize());
        } catch (Throwable failure) {
            logger.warn("Failed to report service discovery executor metrics", failure);
        }
    }

    /**
     * Failed and empty discovery results retain the endpoint's last non-empty snapshot.
     * Cached workers continue through the regular worker health checks.
     */
    private ServiceHosts getServiceHosts(String modelName, WorkerDiscoveryEndpoint endpoint) {
        String address = endpoint.getAddress();
        Future<List<WorkerHost>> future;
        try {
            future =
                    serviceDiscoveryExecutor.submit(
                            () ->
                                    expandWorkerHosts(
                                            java.util.Objects.requireNonNull(
                                                    serviceDiscovery.getHosts(endpoint),
                                                    "discovery hosts"),
                                            endpoint));
        } catch (RejectedExecutionException e) {
            logger.error(
                    "query service discovery rejected, model={}, address={}, msg:{}",
                    modelName,
                    address,
                    e.getMessage());
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
            return new ServiceHosts(
                    lastNonEmptyHostsByEndpoint.getOrDefault(endpoint, List.of()), null);
        }
        try {
            List<WorkerHost> hosts = future.get(500, TimeUnit.MILLISECONDS);
            reportWorkerAvailability(modelName, endpoint, hosts);
            if (!hosts.isEmpty()) {
                // Update the cache only after a successful wait;
                lastNonEmptyHostsByEndpoint.put(endpoint, hosts);
                return new ServiceHosts(hosts, hosts.size());
            }
            return new ServiceHosts(
                    lastNonEmptyHostsByEndpoint.getOrDefault(endpoint, List.of()), 0);
        } catch (TimeoutException e) {
            future.cancel(true);
            logger.error(
                    "query service discovery timeout, model={}, address={}", modelName, address);
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null));
        } catch (InterruptedException e) {
            future.cancel(true);
            Thread.currentThread().interrupt();
            logger.warn(
                    "query service discovery interrupted, model={}, address={}",
                    modelName,
                    address);
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
        } catch (ExecutionException e) {
            logger.error(
                    "query service discovery error, model={}, address={}",
                    modelName,
                    address,
                    e.getCause());
            org.flexlb.util.metrics.SafeFlexMonitor.run(
                    "discovery",
                    () ->
                            serviceMetricsReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
        }
        return new ServiceHosts(
                lastNonEmptyHostsByEndpoint.getOrDefault(endpoint, List.of()), null);
    }

    private void reportWorkerAvailability(
            String modelName, WorkerDiscoveryEndpoint endpoint, List<WorkerHost> hosts) {
        WorkerAvailabilityLogState state =
                workerAvailabilityByEndpoint.computeIfAbsent(
                        endpoint, ignored -> new WorkerAvailabilityLogState());
        if (hosts.isEmpty()) {
            if (state.shouldWarnForEmptyWorkers()) {
                logger.warn(
                        "No workers discovered, model={}, address={}, group={}; "
                                + "retaining last non-empty worker list (warning limited to once per minute)",
                        modelName,
                        endpoint.getAddress(),
                        endpoint.getGroup());
            }
            return;
        }

        if (state.markAvailable()) {
            logger.info(
                    "Worker discovery recovered, model={}, address={}, group={}, worker_count={}",
                    modelName,
                    endpoint.getAddress(),
                    endpoint.getGroup(),
                    hosts.size());
        }
    }

    private static void requireValidPort(long port, String name) {
        if (port <= 0 || port > 65535) {
            throw new IllegalArgumentException(name + " must be in [1, 65535], received " + port);
        }
    }

    private record ServiceHosts(
            List<WorkerHost> discoveredOrCachedHostsForStatusCheck, Integer discoveredHostCount) {}

    /**
     * Limits repeated empty-Worker warnings and resets the deadline when Workers return.
     */
    private static final class WorkerAvailabilityLogState {

        // Zero means workers are available; otherwise this is the next empty-list warning deadline.
        private final AtomicLong emptyWarningDeadlineNanos = new AtomicLong();

        private boolean shouldWarnForEmptyWorkers() {
            long now = System.nanoTime();
            long warningDeadline = emptyWarningDeadlineNanos.get();
            if (warningDeadline != 0 && now < warningDeadline) {
                return false;
            }
            return emptyWarningDeadlineNanos.compareAndSet(
                    warningDeadline, now + EMPTY_WORKER_WARNING_INTERVAL_NANOS);
        }

        private boolean markAvailable() {
            long warningDeadline = emptyWarningDeadlineNanos.get();
            return warningDeadline != 0
                    && emptyWarningDeadlineNanos.compareAndSet(warningDeadline, 0);
        }
    }
}
