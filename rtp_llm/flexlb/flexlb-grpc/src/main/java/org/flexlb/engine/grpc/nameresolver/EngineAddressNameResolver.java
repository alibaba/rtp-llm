package org.flexlb.engine.grpc.nameresolver;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashSet;
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
    private final Map<String/*address*/, String/*protocol*/> addressProtocolMap = new ConcurrentHashMap<>();

    public EngineAddressNameResolver(
            ServiceDiscovery serviceDiscovery,
            @Value("${MODEL_SERVICE_CONFIG:}") String modelConfig) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceAddressList = initServiceAddressList(modelConfig);
        log.info("EngineAddressNameResolver start subscribe clusters:{} ", serviceAddressList);
        fetchAllDomainsHosts();
        setupListeners(serviceDiscovery, serviceAddressList);
    }

    @Scheduled(fixedDelay = 30000) // Execute every 30 seconds
    public void periodicHostUpdate() {
        fetchAllDomainsHosts();
    }

    private void setupListeners(ServiceDiscovery serviceDiscovery, List<String> serviceAddressList) {
        // Create independent listener for each service address
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
                        .map(endpoint -> {
                            // Keep address -> protocol mapping for port correction in updateDomainHosts
                            if (endpoint.getAddress() != null && endpoint.getProtocol() != null) {
                                addressProtocolMap.put(endpoint.getAddress(), endpoint.getProtocol());
                            }
                            return endpoint.getAddress();
                        })
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
     * Update host list for specified address and aggregate all address host lists
     *
     * @param address  Service address
     * @param hostList Host list
     */
    private synchronized void updateDomainHosts(String address, List<WorkerHost> hostList) {
        if (hostList == null || hostList.isEmpty()) {
            domainHostsMap.remove(address);
        } else {
            // VipServer registers the gRPC port (not httpPort) for GRPC-protocol deployments.
            // Downstream AbstractGrpcClient expects "ip:httpPort" and applies toGrpcPort(+1),
            // so correct the port back to httpPort semantics here (aligned with the GRPC branch
            // of WorkerAddressService.convertServiceDiscoveryHosts on the sync path).
            String protocol = addressProtocolMap.get(address);
            boolean isGrpcProtocol = BackendServiceProtocolEnum.GRPC.getName().equalsIgnoreCase(protocol);
            List<String/*ip:port*/> ipPortList = new ArrayList<>(hostList.size());
            for (WorkerHost host : hostList) {
                int port = isGrpcProtocol ? host.getPort() - 1 : host.getPort();
                ipPortList.add(host.getIp() + ":" + port);
            }
            domainHostsMap.put(address, ipPortList);
        }
        // Aggregate host lists from all addresses
        List<String/*ip:port*/> aggregatedHosts = new ArrayList<>();
        for (List<String/*ip:port*/> hosts : domainHostsMap.values()) {
            aggregatedHosts.addAll(hosts);
        }

        // Service discovery polls even when membership is unchanged. Keep the
        // update path and its logs edge-triggered so a steady cluster is quiet.
        if (new HashSet<>(allIpPortList).equals(new HashSet<>(aggregatedHosts))) {
            return;
        }

        Logger.info("Engine hosts changed: domain={}, domainHosts={}, totalHosts={}",
                address, hostList == null ? 0 : hostList.size(), aggregatedHosts.size());
        // Update global host list and notify listener
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
