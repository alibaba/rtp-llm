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
import org.flexlb.exception.ServiceDiscoveryException;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
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
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return new ArrayList<>();
        }
        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Pair<String, Endpoint>> endpoints = serviceRoute.getAllEndpointsWithGroup(modelEndpointType);
        for (Pair<String, Endpoint> endpointTuple : endpoints) {
            String groupName = endpointTuple.getLeft();
            Endpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}", modelName, modelEndpointType);
                continue;
            }
            String address = endpoint.getAddress();
            workerHosts.addAll(convertServiceDiscoveryHosts(getServiceHosts(modelName, address), endpoint.getProtocol(), groupName));
        }
        return workerHosts;
    }

    /**
     * Resolves the hosts behind one discovery address. A successful lookup may legitimately
     * return an empty list (the service has no hosts); a failed or timed-out lookup throws
     * {@link ServiceDiscoveryException} so callers never mistake an outage for an empty fleet.
     */
    public List<WorkerHost> getServiceHosts(String modelName, String address) {
        // Use all machines mounted on the first service discovery address in ServiceRoute
        ServiceDiscoveryRunner serviceDiscoveryRunner = new ServiceDiscoveryRunner(address, serviceDiscovery);
        Future<List<WorkerHost>> future = serviceDiscoveryExecutor.submit(serviceDiscoveryRunner);
        try {
            // Set timeout to prevent blocking threads when service discovery has no machines and takes long to return
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            logger.error("query service discovery timeout, model={}, address={}, msg:{}", modelName, address, "timeout");
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null, null);
            throw new ServiceDiscoveryException(BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT,
                    "service discovery timeout, model=" + modelName + ", address=" + address, e);
        } catch (Exception e) {
            future.cancel(true);
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            Throwable cause = e instanceof ExecutionException && e.getCause() != null ? e.getCause() : e;
            logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address, cause.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null, null);
            throw new ServiceDiscoveryException(BalanceStatusEnum.SERVICE_DISCOVERY_ERROR,
                    "service discovery failed, model=" + modelName + ", address=" + address + ", msg=" + cause.getMessage(), cause);
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

        private final String address;

        private final ServiceDiscovery serviceDiscovery;

        public ServiceDiscoveryRunner(String address, ServiceDiscovery serviceDiscovery) {
            this.address = address;
            this.serviceDiscovery = serviceDiscovery;
        }

        @Override
        public List<WorkerHost> call() throws Exception {
            return serviceDiscovery.getHosts(address);
        }
    }
}
