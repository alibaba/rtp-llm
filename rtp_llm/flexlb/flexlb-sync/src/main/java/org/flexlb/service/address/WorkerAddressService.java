package org.flexlb.service.address;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.annotation.PreDestroy;

import com.alibaba.csp.sentinel.util.function.Tuple2;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.TypeReference;

import com.taobao.vipserver.client.core.Host;
import com.taobao.vipserver.client.core.VIPClient;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.domain.worker.WorkerHost;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service("workerAddressService")
public class WorkerAddressService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final EngineHealthReporter engineHealthReporter;
    private final ModelMetaConfig modelMetaConfig;
    /**
     * vipserver请求线程池
     */
    public static ExecutorService vipServerExecutor = new ThreadPoolExecutor(10, 1000, 60L,
            TimeUnit.SECONDS, new LinkedBlockingQueue<>(1000),
            new NamedThreadFactory("vipserver-query-executor"),
            new ThreadPoolExecutor.CallerRunsPolicy());

    public WorkerAddressService(EngineHealthReporter engineHealthReporter,
                                ModelMetaConfig modelMetaConfig) {

        this.engineHealthReporter = engineHealthReporter;
        this.modelMetaConfig = modelMetaConfig;
    }

    @PreDestroy
    public void destroy() {
        vipServerExecutor.shutdown();
    }

    public List<WorkerHost> getEngineWorkerList(String modelName, RoleType modelEndpointType) {
        logger.info("get engine worker list, modelName={}, modelEndpointType={}", modelName, modelEndpointType);
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoute(IdUtils.getServiceIdByModelName(modelName));
        if (serviceRoute == null) {
            logger.info("modelName={} service route not found", modelName);
            return new ArrayList<>();
        }
        List<WorkerHost> workerHosts = new ArrayList<>();
        List<Tuple2<String, Endpoint>> endpoints = serviceRoute.getAllEndpointsWithGroup(modelEndpointType);
        for (Tuple2<String, Endpoint> endpointTuple : endpoints) {
            String groupName = endpointTuple.r1;
            Endpoint endpoint = endpointTuple.r2;
            if (endpoint == null) {
                logger.info("modelName={} endpoint is null, endpointType={}", modelName, modelEndpointType);
                continue;
            }
            String type = endpoint.getType();
            if (LoadBalanceStrategyEnum.VIPSERVER.getName().equals(type)) {
                String address = endpoint.getAddress();
                workerHosts.addAll(convertVipserverHosts(agetVIPHosts(modelName, address), endpoint.getProtocol(), groupName));
            } else if (LoadBalanceStrategyEnum.SPECIFIED_IP_PORT_LIST.getName().equals(type)) {
                String address = endpoint.getAddress();
                List<String> ipPortList = JSON.parseObject(address, new TypeReference<List<String>>() {
                });
                ipPortList.forEach(ipPort -> {
                    String ip = ipPort.split(":")[0];
                    int port = Integer.parseInt(ipPort.split(":")[1]);
                    if (BackendServiceProtocolEnum.GRPC.getName().equals(endpoint.getProtocol())) {
                        workerHosts.add(new WorkerHost(ip, port - 1, port,  port + 4, "", groupName));
                    } else {
                        workerHosts.add(new WorkerHost(ip, port, port + 1,  port + 5, "", groupName));
                    }
                });
            }
            logger.info("modelName={}, endpoint type is not support, type={}", modelName, type);
        }
        return workerHosts;
    }

    public List<Host> agetVIPHosts(String modelName, String address) {
        // 使用ServiceRoute里面第一个vipServer地址挂载的所有机器
        VipserverRunner vipserverRunner = new VipserverRunner(modelName, address, engineHealthReporter);
        Future<List<Host>> future = vipServerExecutor.submit(vipserverRunner);
        try {
            // 设置超时时间，防止部分vipserver没有机器，返回时间很长阻塞线程
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            if (e instanceof TimeoutException) {
                logger.error("query vipserver timeout, model={}, address={}, msg:{}", modelName, address, "timeout");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.VIPSERVER_TIMEOUT);
            } else {
                logger.error("query vipserver error, model={}, address={}, msg:{}", modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.VIPSERVER_ERROR);
            }
            future.cancel(true);
            return new ArrayList<>();
        }
    }

    public List<WorkerHost> convertVipserverHosts(List<Host> hosts, String protocol, String groupName) {
        List<WorkerHost> workerHosts = new ArrayList<>();
        for (Host host : hosts) {
            if (BackendServiceProtocolEnum.GRPC.getName().equals(protocol)) {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort() - 1, host.getPort(), host.getPort() + 4, host.getSite(), groupName));
            } else {
                workerHosts.add(new WorkerHost(host.getIp(), host.getPort(), host.getPort() + 1, host.getPort() + 5, host.getSite(), groupName));
            }
        }
        return workerHosts;
    }

    public static class VipserverRunner implements Callable<List<Host>> {

        private static final Logger logger = LoggerFactory.getLogger("syncLogger");

        private final String modelName;

        private final String address;

        private final EngineHealthReporter engineHealthReporter;

        public VipserverRunner(String modelName, String address, EngineHealthReporter engineHealthReporter) {
            this.address = address;
            this.engineHealthReporter = engineHealthReporter;
            this.modelName = modelName;
        }

        @Override
        public List<Host> call() {
            long start = System.currentTimeMillis();
            try {
                return VIPClient.srvHosts(address);
            } catch (Throwable e) {
                logger.error("query vipserver exception, cost={}ms, model={}, address={}, msg:{}",
                        System.currentTimeMillis() - start, modelName, address, e.getMessage());
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.VIPSERVER_ERROR);
                return new ArrayList<>();
            }
        }
    }
}
