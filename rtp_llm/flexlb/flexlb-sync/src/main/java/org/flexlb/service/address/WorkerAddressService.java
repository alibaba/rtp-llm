package org.flexlb.service.address;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

@Service("workerAddressService")
public class WorkerAddressService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final EngineHealthReporter engineHealthReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ServiceDiscovery serviceDiscovery;
    private final ExecutorService serviceDiscoveryExecutorRef;
    /**
     * Service discovery request thread pool
     */
    public static final ExecutorService serviceDiscoveryExecutor = new ThreadPoolExecutor(
            10,
            1000,
            60L,
            TimeUnit.SECONDS, new LinkedBlockingQueue<>(1000),
            new NamedThreadFactory("service-discovery-executor"),
            new ThreadPoolExecutor.AbortPolicy()
    );

    @Autowired
    public WorkerAddressService(EngineHealthReporter engineHealthReporter,
                                ModelMetaConfig modelMetaConfig,
                                ServiceDiscovery serviceDiscovery) {
        this(engineHealthReporter, modelMetaConfig, serviceDiscovery, serviceDiscoveryExecutor);
    }

    public WorkerAddressService(EngineHealthReporter engineHealthReporter,
                                ModelMetaConfig modelMetaConfig,
                                ServiceDiscovery serviceDiscovery,
                                ExecutorService serviceDiscoveryExecutorRef) {
        this.engineHealthReporter = engineHealthReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.serviceDiscovery = serviceDiscovery;
        this.serviceDiscoveryExecutorRef = serviceDiscoveryExecutorRef;
    }

    @PreDestroy
    public void destroy() {
        serviceDiscoveryExecutorRef.shutdown();
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        return getEngineWorkerDiscoveryResult(modelName, modelEndpointType).hosts();
    }

    public WorkerDiscoveryResult getEngineWorkerDiscoveryResult(String modelName, RoleType modelEndpointType) {
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return WorkerDiscoveryResult.success(new ArrayList<>());
        }
        List<WorkerHost> workerHosts = new ArrayList<>();
        boolean reliable = true;
        List<Pair<String, Endpoint>> endpoints = serviceRoute.getAllEndpointsWithGroup(modelEndpointType);
        for (Pair<String, Endpoint> endpointTuple : endpoints) {
            String groupName = endpointTuple.getLeft();
            Endpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}", modelName, modelEndpointType);
                continue;
            }
            String address = endpoint.getAddress();
            WorkerDiscoveryResult discoveryResult = getServiceHostsResult(modelName, address);
            reliable = reliable && discoveryResult.reliable();
            workerHosts.addAll(convertServiceDiscoveryHosts(discoveryResult.hosts(), endpoint.getProtocol(), groupName));
        }
        return new WorkerDiscoveryResult(workerHosts, reliable);
    }

    public List<WorkerHost> getServiceHosts(String modelName, String address) {
        return getServiceHostsResult(modelName, address).hosts();
    }

    public WorkerDiscoveryResult getServiceHostsResult(String modelName, String address) {
        // Use all machines mounted on the first service discovery address in ServiceRoute
        ServiceDiscoveryRunner serviceDiscoveryRunner = new ServiceDiscoveryRunner(modelName, address, engineHealthReporter, serviceDiscovery);
        Future<List<WorkerHost>> future;
        try {
            future = serviceDiscoveryExecutorRef.submit(serviceDiscoveryRunner);
        } catch (RejectedExecutionException e) {
            logger.error("query service discovery rejected, model={}, address={}, msg:{}", modelName, address, e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            return WorkerDiscoveryResult.failure();
        }
        try {
            // Set timeout to prevent blocking threads when service discovery has no machines and takes long to return
            return WorkerDiscoveryResult.success(future.get(500, TimeUnit.MILLISECONDS));
        } catch (Exception e) {
            if (e instanceof TimeoutException) {
                logger.error("query service discovery timeout, model={}, address={}, msg:{}", modelName, address, "timeout");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null, null);
            } else if (e instanceof ExecutionException) {
                logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address,
                        e.getCause() == null ? e.getMessage() : e.getCause().getMessage());
            } else {
                logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            }
            if (future != null) {
                future.cancel(true);
            }
            return WorkerDiscoveryResult.failure();
        }
    }

    public List<WorkerHost> convertServiceDiscoveryHosts(List<WorkerHost> hosts, String protocol, String groupName) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        for (WorkerHost host : hosts) {
            if (BackendServiceProtocolEnum.GRPC.getName().equals(protocol)) {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort() - 1, host.getPort(), host.getPort() + 4, host.getSite(), groupName));
            } else {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort(), host.getPort() + 1, host.getPort() + 5, host.getSite(), groupName));
            }
        }
        return workerHosts;
    }

    public static class ServiceDiscoveryRunner implements Callable<List<WorkerHost>> {

        private static final Logger logger = LoggerFactory.getLogger("syncLogger");

        private final String modelName;

        private final String address;

        private final EngineHealthReporter engineHealthReporter;

        private final ServiceDiscovery serviceDiscovery;

        public ServiceDiscoveryRunner(String modelName,
                                      String address,
                                      EngineHealthReporter engineHealthReporter,
                                      ServiceDiscovery serviceDiscovery) {
            this.address = address;
            this.engineHealthReporter = engineHealthReporter;
            this.modelName = modelName;
            this.serviceDiscovery = serviceDiscovery;
        }

        @Override
        public List<WorkerHost> call() {
            long start = System.nanoTime() / 1000;
            try {
                return serviceDiscovery.getHosts(address);
            } catch (Throwable e) {
                logger.error("query service discovery exception, cost={}ms, model={}, address={}, msg:{}",
                        System.nanoTime() / 1000 - start, modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
                throw new RuntimeException(e);
            }
        }
    }

    public record WorkerDiscoveryResult(List<WorkerHost> hosts, boolean reliable) {
        public static WorkerDiscoveryResult success(List<WorkerHost> hosts) {
            return new WorkerDiscoveryResult(hosts == null ? new ArrayList<>() : hosts, true);
        }

        public static WorkerDiscoveryResult failure() {
            return new WorkerDiscoveryResult(new ArrayList<>(), false);
        }
    }
}
