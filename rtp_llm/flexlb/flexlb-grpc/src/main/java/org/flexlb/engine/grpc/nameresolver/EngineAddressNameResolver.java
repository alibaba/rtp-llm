package org.flexlb.engine.grpc.nameresolver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.enums.BackendServiceProtocolEnum;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
    private final Map<String/*address*/, String/*protocol*/> addressProtocols;

    public EngineAddressNameResolver(
            ServiceDiscovery serviceDiscovery,
            ModelMetaConfig modelConfig) {
        this.serviceDiscovery = serviceDiscovery;
        this.addressProtocols = addressProtocols(modelConfig.getServiceRoute());
        log.info("EngineAddressNameResolver start subscribe clusters:{} ", addressProtocols.keySet());
        fetchAllDomainsHosts();
        setupListeners();
    }

    @Scheduled(fixedDelay = 30000) // Execute every 30 seconds
    public void periodicHostUpdate() {
        fetchAllDomainsHosts();
    }

    private void setupListeners() {
        // Create independent listener for each service address
        for (String serviceAddress : addressProtocols.keySet()) {
            ServiceHostListener addressListener = hosts -> updateDomainHosts(serviceAddress, hosts);
            serviceDiscovery.listen(serviceAddress, addressListener);
        }
    }

    private void fetchAllDomainsHosts() {
        for (String serverAddress : addressProtocols.keySet()) {
            try {
                List<WorkerHost> hosts = serviceDiscovery.getHosts(serverAddress);
                updateDomainHosts(serverAddress, hosts);
            } catch (Exception e) {
                Logger.error("Failed to fetch hosts for domain: {}, error: {}", serverAddress, e.getMessage(), e);
            }
        }
    }

    private static Map<String, String> addressProtocols(ServiceRoute serviceRoute) {
        Map<String, String> protocols = new LinkedHashMap<>();
        for (var endpoint : serviceRoute.getAllEndpoints()) {
            if (endpoint.getAddress() != null) {
                protocols.putIfAbsent(endpoint.getAddress(), null);
                if (endpoint.getProtocol() != null) {
                    protocols.put(endpoint.getAddress(), endpoint.getProtocol());
                }
            }
        }
        return Collections.unmodifiableMap(protocols);
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
            String protocol = addressProtocols.get(address);
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
