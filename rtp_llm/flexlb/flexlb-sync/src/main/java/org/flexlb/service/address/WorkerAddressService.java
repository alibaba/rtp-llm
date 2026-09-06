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
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.enums.BalanceStatusEnum;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
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
    private final ConcurrentMap<String, WorkerAvailabilityLogState> workerAvailabilityByAddress =
            new ConcurrentHashMap<>();

    /**
     * Service discovery request thread pool
     */
    private final ThreadPoolExecutor serviceDiscoveryExecutor;

    public WorkerAddressService(EngineHealthReporter engineHealthReporter,
                                ModelMetaConfig modelMetaConfig,
                                ServiceDiscovery serviceDiscovery,
                                ConfigService configService) {

        this.engineHealthReporter = engineHealthReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.serviceDiscovery = serviceDiscovery;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.serviceDiscoveryExecutor = new ThreadPoolExecutor(
                10,
                config.getInternalRuntime().getServiceDiscoveryMaxThreads(),
                60L,
                TimeUnit.SECONDS, new LinkedBlockingQueue<>(1000),
                new NamedThreadFactory("service-discovery-executor"),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
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
                    "serviceDiscoveryExecutor", serviceDiscoveryExecutor);
        } catch (Throwable failure) {
            logger.warn("Failed to report service discovery executor metrics", failure);
        }
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Pair<String, Endpoint>> endpoints =
                modelMetaConfig.endpointsWithGroup(
                        modelName, modelEndpointType);
        if (endpoints.isEmpty()) {
            logger.info("modelName={} role={} service route not found",
                    modelName, modelEndpointType);
            return workerHosts;
        }
        for (Pair<String, Endpoint> endpointTuple : endpoints) {
            Endpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}", modelName, modelEndpointType);
                continue;
            }
            workerHosts.addAll(getServiceHosts(modelName, endpoint));
        }
        return workerHosts;
    }

    private List<WorkerHost> getServiceHosts(String modelName, Endpoint endpoint) {
        Future<List<WorkerHost>> future = serviceDiscoveryExecutor.submit(
                () -> queryServiceHosts(modelName, endpoint));
        try {
            // Prevent a slow service discovery request from blocking status synchronization.
            List<WorkerHost> hosts = future.get(500, TimeUnit.MILLISECONDS);
            reportWorkerAvailability(modelName, endpoint, hosts);
            return hosts;
        } catch (Exception e) {
            String address = endpoint.getAddress();
            if (e instanceof TimeoutException) {
                logger.error("query service discovery timeout, model={}, address={}, msg:{}",
                        modelName, address, "timeout");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
            } else {
                logger.error("query service discovery error, model={}, address={}, msg:{}",
                        modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
            }
            future.cancel(true);
            return new ArrayList<>();
        }
    }

    private void reportWorkerAvailability(String modelName, Endpoint endpoint, List<WorkerHost> hosts) {
        WorkerAvailabilityLogState state = workerAvailabilityByAddress.computeIfAbsent(
                endpoint.getAddress(), ignored -> new WorkerAvailabilityLogState());
        if (hosts == null || hosts.isEmpty()) {
            if (state.shouldWarnForEmptyWorkers()) {
                logger.warn("No workers discovered, model={}, address={}, group={}; "
                                + "worker list is empty (warning limited to once per minute)",
                        modelName, endpoint.getAddress(), endpoint.getGroup());
            }
            return;
        }

        if (state.markAvailable()) {
            logger.info("Worker discovery recovered, model={}, address={}, group={}, worker_count={}",
                    modelName, endpoint.getAddress(), endpoint.getGroup(), hosts.size());
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

    private List<WorkerHost> queryServiceHosts(
            String modelName, Endpoint endpoint) {
        long startNanos = System.nanoTime();
        try {
            return serviceDiscovery.getHosts(endpoint);
        } catch (Throwable failure) {
            logger.error("query service discovery exception, cost={}ms, model={}, address={}, msg:{}",
                    TimeUnit.NANOSECONDS.toMillis(
                            System.nanoTime() - startNanos),
                    modelName, endpoint.getAddress(), failure.getMessage());
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
            return new ArrayList<>();
        }
    }
}
