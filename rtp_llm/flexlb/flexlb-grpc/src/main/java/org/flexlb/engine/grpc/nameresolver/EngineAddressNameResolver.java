package org.flexlb.engine.grpc.nameresolver;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * @author saichen.sm
 * date: 2025/9/19
 */
@Slf4j
@Component
public class EngineAddressNameResolver implements CustomNameResolver {

    private final Map<String/*address*/, List<String/*ip:port*/>> domainHostsMap = new ConcurrentHashMap<>();
    private final ServiceDiscovery serviceDiscovery;
    private Listener listener;
    private List<String/*ip:port*/> allIpPortList = new ArrayList<>();
    private final List<String> serviceAddressList;

    public EngineAddressNameResolver(ServiceDiscovery serviceDiscovery) {
        String modelConfig = System.getenv("MODEL_SERVICE_CONFIG");
        this.serviceDiscovery = serviceDiscovery;
        this.serviceAddressList = initServiceAddressList(modelConfig);
        log.info("EngineAddressNameResolver start subscribe clusters:{} ", serviceAddressList);
        fetchAllDomainsHosts();
        setupListeners(serviceDiscovery, serviceAddressList);
    }

    @Scheduled(fixedDelay = 30000) // 每30秒执行一次
    public void periodicHostUpdate() {
        Logger.info("EngineAddressNameResolver performing periodic host update for domains: {}", serviceAddressList);
        fetchAllDomainsHosts();
    }

    private void setupListeners(ServiceDiscovery serviceDiscovery, List<String> serviceAddressList) {
        // 为每个服务地址创建独立的监听器
        for (String serviceAddress : serviceAddressList) {
            if (serviceAddress == null) {
                Logger.warn("Skipping null serviceAddress");
                continue;
            }
            ServiceHostListener addressListener = hosts -> updateDomainHosts(serviceAddress, hosts);
            serviceDiscovery.listen(serviceAddress, addressListener);
        }
    }

    private void fetchAllDomainsHosts() {
        for (String serverAddress : serviceAddressList) {
            if (serverAddress == null) {
                Logger.warn("Skipping null serverAddress during fetch");
                continue;
            }

            try {
                List<WorkerHost> hosts = serviceDiscovery.getHosts(serverAddress);
                Logger.info("Fetched {} hosts for domain: {}", hosts != null ? hosts.size() : 0, serverAddress);
                updateDomainHosts(serverAddress, hosts);
            } catch (Exception e) {
                Logger.error("Failed to fetch hosts for domain: {}, error: {}", serverAddress, e.getMessage(), e);
            }
        }
    }

    private List<String> initServiceAddressList(String modelConfigJson) {
        return Optional.ofNullable(modelConfigJson)
                .filter(StringUtils::isNotBlank)
                .map(json -> JsonUtils.toObject(modelConfigJson, ServiceRoute.class))
                .map(serviceRoute -> serviceRoute.getAllEndpoints().stream()
                        .map(Endpoint::getAddress)
                        .collect(Collectors.toList()))
                .filter(CollectionUtils::isNotEmpty)
                .orElseThrow(() -> new IllegalArgumentException("serviceAddressList cannot be null, please config 'MODEL_SERVICE_CONFIG' environment variable, modelConfigJson=" + modelConfigJson));
    }

    @Override
    public void start(Listener listener) {
        this.listener = listener;
        listener.onAddressUpdate(allIpPortList);
    }

    /**
     * 更新指定地址的机器列表，并聚合所有地址的机器列表
     *
     * @param address  服务地址
     * @param hostList 主机列表
     */
    private void updateDomainHosts(String address, List<WorkerHost> hostList) {
        if (hostList == null || hostList.isEmpty()) {
            domainHostsMap.remove(address);
        } else {
            List<String/*ip:port*/> ipPortList = new ArrayList<>(hostList.size());
            for (WorkerHost host : hostList) {
                ipPortList.add(host.getIp() + ":" + host.getPort());
            }
            domainHostsMap.put(address, ipPortList);
        }
        // 聚合所有地址的机器列表
        List<String/*ip:port*/> aggregatedHosts = new ArrayList<>();
        for (List<String/*ip:port*/> hosts : domainHostsMap.values()) {
            aggregatedHosts.addAll(hosts);
        }
        Logger.info("Address {} hosts updated, total aggregated hosts: {}", address, aggregatedHosts.size());
        // 更新全局机器列表并通知监听器
        this.allIpPortList = aggregatedHosts;
        if (this.listener != null) {
            this.listener.onAddressUpdate(allIpPortList);
        }
    }

    @PreDestroy
    public void destroy() {
        if (serviceDiscovery != null) {
            serviceDiscovery.shutdown();
        }
    }
}