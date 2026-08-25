package org.flexlb.service.address;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

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
    public static final ExecutorService serviceDiscoveryExecutor = new ThreadPoolExecutor(
            10,
            1000,
            60L,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1000),
            new NamedThreadFactory("service-discovery-executor"),
            new ThreadPoolExecutor.CallerRunsPolicy());

    public WorkerAddressService(EngineHealthReporter engineHealthReporter,
                                ModelMetaConfig modelMetaConfig,
                                ServiceDiscovery serviceDiscovery) {
        this.engineHealthReporter = engineHealthReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.serviceDiscovery = serviceDiscovery;
    }

    @PreDestroy
    public void destroy() {
        serviceDiscoveryExecutor.shutdown();
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return new ArrayList<>();
        }

        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Pair<String, Endpoint>> endpoints = serviceRoute.getAllEndpointsWithGroup(modelEndpointType);
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

    public List<WorkerHost> getServiceHosts(String modelName, Endpoint endpoint) {
        ServiceDiscoveryRunner serviceDiscoveryRunner =
                new ServiceDiscoveryRunner(modelName, endpoint, engineHealthReporter, serviceDiscovery);
        Future<List<WorkerHost>> future = serviceDiscoveryExecutor.submit(serviceDiscoveryRunner);
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
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null, null);
            } else {
                logger.error("query service discovery error, model={}, address={}, msg:{}",
                        modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
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

    public static class ServiceDiscoveryRunner implements Callable<List<WorkerHost>> {

        private static final Logger logger = LoggerFactory.getLogger("syncLogger");

        private final String modelName;
        private final Endpoint endpoint;
        private final EngineHealthReporter engineHealthReporter;
        private final ServiceDiscovery serviceDiscovery;

        public ServiceDiscoveryRunner(String modelName,
                                      Endpoint endpoint,
                                      EngineHealthReporter engineHealthReporter,
                                      ServiceDiscovery serviceDiscovery) {
            this.endpoint = endpoint;
            this.engineHealthReporter = engineHealthReporter;
            this.modelName = modelName;
            this.serviceDiscovery = serviceDiscovery;
        }

        @Override
        public List<WorkerHost> call() {
            long start = System.nanoTime() / 1000;
            try {
                return serviceDiscovery.getHosts(endpoint);
            } catch (Throwable e) {
                logger.error("query service discovery exception, cost={}ms, model={}, address={}, msg:{}",
                        System.nanoTime() / 1000 - start, modelName, endpoint.getAddress(), e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
                return new ArrayList<>();
            }
        }
    }
}
