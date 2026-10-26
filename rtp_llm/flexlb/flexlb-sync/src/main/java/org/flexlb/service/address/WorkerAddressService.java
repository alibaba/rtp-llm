package org.flexlb.service.address;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
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

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;

@Service("workerAddressService")
public class WorkerAddressService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final long EMPTY_WORKER_WARNING_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);

    private final EngineHealthReporter engineHealthReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ServiceDiscovery serviceDiscovery;
    // Cache the last non-empty worker list per endpoint; its key fields stay unchanged after
    // initialization.
    private final ConcurrentMap<Endpoint, List<WorkerHost>> lastNonEmptyHostsByEndpoint =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<Endpoint, WorkerAvailabilityLogState> workerAvailabilityByEndpoint =
            new ConcurrentHashMap<>();

    /**
     * Service discovery request thread pool
     */
    private final ThreadPoolExecutor serviceDiscoveryExecutor;

    public WorkerAddressService(
            EngineHealthReporter engineHealthReporter,
            ModelMetaConfig modelMetaConfig,
            ServiceDiscovery serviceDiscovery,
            ConfigService configService) {

        this.engineHealthReporter = engineHealthReporter;
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

    @Scheduled(fixedRate = 2000)
    private void reportExecutorMetrics() {
        try {
            engineHealthReporter.reportThreadPoolInfo(
                    ENGINE_BALANCING_THREAD_POOL_INFO,
                    "serviceDiscoveryExecutor",
                    serviceDiscoveryExecutor);
        } catch (Throwable failure) {
            logger.warn("Failed to report service discovery executor metrics", failure);
        }
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Pair<String, Endpoint>> endpoints =
                modelMetaConfig.endpointsWithGroup(modelName, modelEndpointType);
        if (endpoints.isEmpty()) {
            logger.info(
                    "modelName={} role={} service route not found", modelName, modelEndpointType);
            reportDiscoveredHostCount(modelName, modelEndpointType, 0);
            return workerHosts;
        }
        int discoveredHostCount = 0;
        boolean allEndpointsSucceeded = true;
        for (Pair<String, Endpoint> endpointTuple : endpoints) {
            Endpoint endpoint = endpointTuple.getRight();
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
            reportDiscoveredHostCount(modelName, modelEndpointType, discoveredHostCount);
        }
        return workerHosts;
    }

    private void reportDiscoveredHostCount(String modelName, RoleType role, int count) {
        try {
            engineHealthReporter.reportRawServiceDiscoveryHostCount(modelName, role, count);
        } catch (Throwable failure) {
            logger.warn(
                    "Failed to report discovered worker count for {} {}", modelName, role, failure);
        }
    }

    /**
     * Failed and empty discovery results retain the endpoint's last non-empty snapshot.
     * Cached workers continue through the regular worker health checks.
     */
    private ServiceHosts getServiceHosts(String modelName, Endpoint endpoint) {
        String address = endpoint.getAddress();
        Future<List<WorkerHost>> future;
        try {
            future =
                    serviceDiscoveryExecutor.submit(
                            () ->
                                    java.util.Objects.requireNonNull(
                                            serviceDiscovery.getHosts(endpoint),
                                            "discovery hosts"));
        } catch (RejectedExecutionException e) {
            logger.error(
                    "query service discovery rejected, model={}, address={}, msg:{}",
                    modelName,
                    address,
                    e.getMessage());
            org.flexlb.metric.SafeFlexMonitor.run(
                    "serviceDiscoveryFailure",
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
            return new ServiceHosts(cachedHostsOrThrow(endpoint, e), null);
        }
        Throwable discoveryFailure;
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
            discoveryFailure = e;
            future.cancel(true);
            logger.error(
                    "query service discovery timeout, model={}, address={}", modelName, address);
            org.flexlb.metric.SafeFlexMonitor.run(
                    "serviceDiscoveryFailure",
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null));
        } catch (InterruptedException e) {
            discoveryFailure = e;
            future.cancel(true);
            Thread.currentThread().interrupt();
            logger.warn(
                    "query service discovery interrupted, model={}, address={}",
                    modelName,
                    address);
            org.flexlb.metric.SafeFlexMonitor.run(
                    "serviceDiscoveryFailure",
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
        } catch (ExecutionException e) {
            discoveryFailure = e.getCause();
            logger.error(
                    "query service discovery error, model={}, address={}",
                    modelName,
                    address,
                    e.getCause());
            org.flexlb.metric.SafeFlexMonitor.run(
                    "serviceDiscoveryFailure",
                    () ->
                            engineHealthReporter.reportStatusCheckerFail(
                                    BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null));
        }
        return new ServiceHosts(cachedHostsOrThrow(endpoint, discoveryFailure), null);
    }

    private List<WorkerHost> cachedHostsOrThrow(Endpoint endpoint, Throwable failure) {
        List<WorkerHost> cached = lastNonEmptyHostsByEndpoint.get(endpoint);
        if (cached != null && !cached.isEmpty()) {
            return cached;
        }
        throw new IllegalStateException(
                "Worker discovery failed without a cached snapshot: " + endpoint.getAddress(),
                failure);
    }

    private record ServiceHosts(
            List<WorkerHost> discoveredOrCachedHostsForStatusCheck, Integer discoveredHostCount) {}

    private void reportWorkerAvailability(
            String modelName, Endpoint endpoint, List<WorkerHost> hosts) {
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

    private static final class WorkerAvailabilityLogState {

        // Zero means workers are available; otherwise this is the next empty-list warning deadline.
        private final AtomicLong emptyWarningDeadlineNanos = new AtomicLong();

        boolean shouldWarnForEmptyWorkers() {
            long now = System.nanoTime();
            long warningDeadline = emptyWarningDeadlineNanos.get();
            if (warningDeadline != 0 && now < warningDeadline) {
                return false;
            }
            return emptyWarningDeadlineNanos.compareAndSet(
                    warningDeadline, now + EMPTY_WORKER_WARNING_INTERVAL_NANOS);
        }

        boolean markAvailable() {
            long warningDeadline = emptyWarningDeadlineNanos.get();
            return warningDeadline != 0
                    && emptyWarningDeadlineNanos.compareAndSet(warningDeadline, 0);
        }
    }
}
