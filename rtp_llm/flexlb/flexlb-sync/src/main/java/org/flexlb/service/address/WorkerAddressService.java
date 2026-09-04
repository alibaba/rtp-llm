package org.flexlb.service.address;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.apache.commons.lang3.tuple.Pair;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
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

    /** Service-discovery request pool shared by all model/role lookups. */
    public static ExecutorService serviceDiscoveryExecutor;

    public WorkerAddressService(
            EngineHealthReporter engineHealthReporter,
            ModelMetaConfig modelMetaConfig,
            ServiceDiscovery serviceDiscovery,
            ConfigService configService) {
        this.engineHealthReporter = engineHealthReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.serviceDiscovery = serviceDiscovery;
        FlexlbConfig config = configService.loadBalanceConfig();
        serviceDiscoveryExecutor = new ThreadPoolExecutor(
                10,
                config.getServiceDiscoveryMaxSize(),
                60L,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(1000),
                new NamedThreadFactory("service-discovery-executor"),
                new ThreadPoolExecutor.CallerRunsPolicy());
    }

    @PreDestroy
    public void destroy() {
        serviceDiscoveryExecutor.shutdown();
    }

    /**
     * Resolve every worker for a role across all discovery groups.
     * A failure in any group aborts the snapshot so callers can retain the previous complete view.
     */
    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType roleType) {
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(
                IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return new ArrayList<>();
        }

        List<WorkerHost> workerHosts = new ArrayList<>();
        for (Pair<String, Endpoint> endpointTuple
                : serviceRoute.getAllEndpointsWithGroup(roleType)) {
            String groupName = endpointTuple.getLeft();
            Endpoint endpoint = endpointTuple.getRight();
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}",
                        modelName, roleType);
                continue;
            }
            workerHosts.addAll(convertServiceDiscoveryHosts(
                    getServiceHosts(modelName, endpoint.getAddress()),
                    endpoint.getProtocol(),
                    groupName));
        }
        return workerHosts;
    }

    /**
     * Resolve one discovery address. An empty list is a successful empty fleet; timeout or failure
     * throws so the caller never interprets an outage as authoritative membership removal.
     */
    public List<WorkerHost> getServiceHosts(String modelName, String address) {
        Future<List<WorkerHost>> future =
                serviceDiscoveryExecutor.submit(() -> serviceDiscovery.getHosts(address));
        try {
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (TimeoutException error) {
            future.cancel(true);
            logger.error("query service discovery timeout, model={}, address={}",
                    modelName, address);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
            throw new ServiceDiscoveryException(
                    BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT,
                    "service discovery timeout, model=" + modelName + ", address=" + address,
                    error);
        } catch (InterruptedException error) {
            future.cancel(true);
            Thread.currentThread().interrupt();
            throw discoveryFailure(modelName, address, error);
        } catch (ExecutionException error) {
            future.cancel(true);
            Throwable cause = error.getCause() == null ? error : error.getCause();
            throw discoveryFailure(modelName, address, cause);
        }
    }

    private ServiceDiscoveryException discoveryFailure(
            String modelName, String address, Throwable cause) {
        logger.error("query service discovery error, model={}, address={}, msg:{}",
                modelName, address, cause.getMessage());
        engineHealthReporter.reportStatusCheckerFail(
                modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
        return new ServiceDiscoveryException(
                BalanceStatusEnum.SERVICE_DISCOVERY_ERROR,
                "service discovery failed, model=" + modelName
                        + ", address=" + address + ", msg=" + cause.getMessage(),
                cause);
    }

    public List<WorkerHost> convertServiceDiscoveryHosts(
            List<WorkerHost> hosts, String protocol, String groupName) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        for (WorkerHost host : hosts) {
            if (BackendServiceProtocolEnum.GRPC.getName().equals(protocol)) {
                workerHosts.add(new WorkerHost(
                        host.getIp(), host.getPort() - 1, host.getPort(),
                        host.getPort() + 4, host.getSite(), groupName));
            } else {
                workerHosts.add(new WorkerHost(
                        host.getIp(), host.getPort(), host.getPort() + 1,
                        host.getPort() + 5, host.getSite(), groupName));
            }
        }
        return workerHosts;
    }
}
