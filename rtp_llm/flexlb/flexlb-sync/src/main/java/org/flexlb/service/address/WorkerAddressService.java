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
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

@Service("workerAddressService")
public class WorkerAddressService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final EngineHealthReporter engineHealthReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ServiceDiscovery serviceDiscovery;
    /**
     * Service discovery request thread pool
     */
    public static final ExecutorService serviceDiscoveryExecutor = new ThreadPoolExecutor(
            10,
            1000,
            60L,
            TimeUnit.SECONDS, new LinkedBlockingQueue<>(1000),
            new NamedThreadFactory("service-discovery-executor"),
            new ThreadPoolExecutor.CallerRunsPolicy()
    );

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
        return getEngineWorkers(modelName, modelEndpointType).getWorkerHosts();
    }

    public EngineWorkerList getEngineWorkers(String modelName, RoleType modelEndpointType) {
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return new EngineWorkerList(new ArrayList<>(), new HashSet<>());
        }
        List<WorkerHost> workerHosts = new ArrayList<>();
        Set<String> unavailableGroups = new HashSet<>();
        Set<String> discoveryFailedGroups = new HashSet<>();
        List<Pair<String, Endpoint>> endpoints = serviceRoute.getAllEndpointsWithGroup(modelEndpointType);
        for (Pair<String, Endpoint> endpointTuple : endpoints) {
            String groupName = normalizeGroupName(endpointTuple.getLeft());
            Endpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}", modelName, modelEndpointType);
                unavailableGroups.add(groupName);
                continue;
            }
            String address = endpoint.getAddress();
            ServiceHostResult serviceHostResult = getServiceHostResult(modelName, address);
            if (!serviceHostResult.isSuccess()) {
                logger.warn("modelName={} endpoint discovery failed, keep cached workers, endpointType={}, group={}, address={}",
                        modelName, modelEndpointType, groupName, address);
                discoveryFailedGroups.add(groupName);
                continue;
            }
            List<WorkerHost> groupWorkers = convertServiceDiscoveryHosts(serviceHostResult.getHosts(), endpoint.getProtocol(), groupName);
            if (groupWorkers.isEmpty()) {
                logger.warn("modelName={} endpoint discovery is empty, endpointType={}, group={}, address={}",
                        modelName, modelEndpointType, groupName, address);
                unavailableGroups.add(groupName);
                continue;
            }
            workerHosts.addAll(groupWorkers);
        }
        return new EngineWorkerList(workerHosts, unavailableGroups, discoveryFailedGroups);
    }

    public List<WorkerHost> getServiceHosts(String modelName, String address) {
        return getServiceHostResult(modelName, address).getHosts();
    }

    public ServiceHostResult getServiceHostResult(String modelName, String address) {
        // Use all machines mounted on the first service discovery address in ServiceRoute
        ServiceDiscoveryRunner serviceDiscoveryRunner = new ServiceDiscoveryRunner(modelName, address, serviceDiscovery);
        Future<List<WorkerHost>> future = serviceDiscoveryExecutor.submit(serviceDiscoveryRunner);
        try {
            // Set timeout to prevent blocking threads when service discovery has no machines and takes long to return
            return ServiceHostResult.success(future.get(500, TimeUnit.MILLISECONDS));
        } catch (TimeoutException e) {
            logger.error("query service discovery timeout, model={}, address={}, msg:{}", modelName, address, "timeout");
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null, null);
            future.cancel(true);
            return ServiceHostResult.failure();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("query service discovery interrupted, model={}, address={}, msg:{}", modelName, address, e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            future.cancel(true);
            return ServiceHostResult.failure();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address, cause.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            future.cancel(true);
            return ServiceHostResult.failure();
        } catch (Exception e) {
            logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address, e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            future.cancel(true);
            return ServiceHostResult.failure();
        }
    }

    public List<WorkerHost> convertServiceDiscoveryHosts(List<WorkerHost> hosts, String protocol, String groupName) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        if (hosts == null) {
            return workerHosts;
        }
        String normalizedGroupName = normalizeGroupName(groupName);
        for (WorkerHost host : hosts) {
            if (BackendServiceProtocolEnum.GRPC.getName().equals(protocol)) {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort() - 1, host.getPort(), host.getPort() + 4, host.getSite(), normalizedGroupName));
            } else {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort(), host.getPort() + 1, host.getPort() + 5, host.getSite(), normalizedGroupName));
            }
        }
        return workerHosts;
    }

    private String normalizeGroupName(String groupName) {
        return groupName == null ? "" : groupName;
    }

    public static class EngineWorkerList {

        private final List<WorkerHost> workerHosts;

        private final Set<String> unavailableGroups;

        private final Set<String> discoveryFailedGroups;

        public EngineWorkerList(List<WorkerHost> workerHosts, Set<String> unavailableGroups) {
            this(workerHosts, unavailableGroups, new HashSet<>());
        }

        public EngineWorkerList(List<WorkerHost> workerHosts, Set<String> unavailableGroups, Set<String> discoveryFailedGroups) {
            this.workerHosts = workerHosts == null ? new ArrayList<>() : workerHosts;
            this.unavailableGroups = unavailableGroups == null ? new HashSet<>() : unavailableGroups;
            this.discoveryFailedGroups = discoveryFailedGroups == null ? new HashSet<>() : discoveryFailedGroups;
        }

        public List<WorkerHost> getWorkerHosts() {
            return workerHosts;
        }

        public Set<String> getUnavailableGroups() {
            return unavailableGroups;
        }

        public Set<String> getDiscoveryFailedGroups() {
            return discoveryFailedGroups;
        }
    }

    public static class ServiceHostResult {

        private final List<WorkerHost> hosts;

        private final boolean success;

        private ServiceHostResult(List<WorkerHost> hosts, boolean success) {
            this.hosts = hosts == null ? new ArrayList<>() : hosts;
            this.success = success;
        }

        public static ServiceHostResult success(List<WorkerHost> hosts) {
            return new ServiceHostResult(hosts, true);
        }

        public static ServiceHostResult failure() {
            return new ServiceHostResult(new ArrayList<>(), false);
        }

        public List<WorkerHost> getHosts() {
            return hosts;
        }

        public boolean isSuccess() {
            return success;
        }
    }

    public static class ServiceDiscoveryRunner implements Callable<List<WorkerHost>> {

        private static final Logger logger = LoggerFactory.getLogger("syncLogger");

        private final String modelName;

        private final String address;

        private final ServiceDiscovery serviceDiscovery;

        public ServiceDiscoveryRunner(String modelName,
                                      String address,
                                      ServiceDiscovery serviceDiscovery) {
            this.address = address;
            this.modelName = modelName;
            this.serviceDiscovery = serviceDiscovery;
        }

        @Override
        public List<WorkerHost> call() {
            long start = System.nanoTime() / 1000;
            List<WorkerHost> hosts = serviceDiscovery.getHosts(address);
            logger.debug("query service discovery finished, cost={}us, model={}, address={}, size={}",
                    System.nanoTime() / 1000 - start, modelName, address, hosts == null ? 0 : hosts.size());
            return hosts;
        }
    }
}
