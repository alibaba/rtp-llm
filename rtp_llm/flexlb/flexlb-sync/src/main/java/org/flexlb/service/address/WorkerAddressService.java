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
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;

@Service("workerAddressService")
public class WorkerAddressService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final EngineHealthReporter engineHealthReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ServiceDiscovery serviceDiscovery;
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

    private List<WorkerHost> getServiceHosts(String modelName, String address) {
        Future<List<WorkerHost>> future = serviceDiscoveryExecutor.submit(
                () -> queryServiceHosts(modelName, address));
        try {
            // Set timeout to prevent blocking threads when service discovery has no machines and takes long to return
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            if (e instanceof TimeoutException) {
                logger.error("query service discovery timeout, model={}, address={}, msg:{}", modelName, address, "timeout");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_TIMEOUT, null);
            } else {
                logger.error("query service discovery error, model={}, address={}, msg:{}", modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
            }
            future.cancel(true);
            return new ArrayList<>();
        }
    }

    private static List<WorkerHost> convertServiceDiscoveryHosts(
            List<WorkerHost> hosts, String protocol, String groupName) {
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

    private List<WorkerHost> queryServiceHosts(
            String modelName, String address) {
        long startNanos = System.nanoTime();
        try {
            return serviceDiscovery.getHosts(address);
        } catch (Throwable failure) {
            logger.error("query service discovery exception, cost={}ms, model={}, address={}, msg:{}",
                    TimeUnit.NANOSECONDS.toMillis(
                            System.nanoTime() - startNanos),
                    modelName, address, failure.getMessage());
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.SERVICE_DISCOVERY_ERROR, null);
            return new ArrayList<>();
        }
    }
}
