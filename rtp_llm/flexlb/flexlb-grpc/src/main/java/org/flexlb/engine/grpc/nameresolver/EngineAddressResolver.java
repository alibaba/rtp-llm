package org.flexlb.engine.grpc.nameresolver;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Resolves and publishes the complete set of engine addresses.
 *
 * @author saichen.sm
 * date: 2025/9/19
 */
@Slf4j
@Component
public class EngineAddressResolver {

    private final Map<Endpoint, List<WorkerHost>> domainHostsMap = new ConcurrentHashMap<>();
    private final ServiceDiscovery serviceDiscovery;
    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();
    private volatile List<WorkerHost> allHosts = List.of();
    private final List<Endpoint> serviceEndpoints;

    public EngineAddressResolver(ServiceDiscovery serviceDiscovery, ModelMetaConfig modelMetaConfig) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceEndpoints = initServiceEndpoints(modelMetaConfig);
        log.info("EngineAddressResolver start subscribe endpoints:{} ", serviceEndpoints);
        fetchAllDomainsHosts();
        setupListeners(serviceDiscovery, serviceEndpoints);
    }

    @Scheduled(fixedDelay = 30000) // Execute every 30 seconds
    public void periodicHostUpdate() {
        Logger.info("EngineAddressResolver performing periodic host update for endpoints: {}", serviceEndpoints);
        fetchAllDomainsHosts();
    }

    private void setupListeners(ServiceDiscovery serviceDiscovery, List<Endpoint> endpoints) {
        for (Endpoint endpoint : endpoints) {
            ServiceHostListener addressListener = hosts -> updateEndpointHosts(endpoint, hosts);
            serviceDiscovery.listen(endpoint, addressListener);
        }
    }

    private void fetchAllDomainsHosts() {
        for (Endpoint endpoint : serviceEndpoints) {
            try {
                List<WorkerHost> hosts = serviceDiscovery.getHosts(endpoint);
                Logger.info("Fetched {} hosts for address: {}",
                        hosts != null ? hosts.size() : 0, endpoint.getAddress());
                updateEndpointHosts(endpoint, hosts);
            } catch (Exception e) {
                Logger.error("Failed to fetch hosts for address: {}, error: {}",
                        endpoint.getAddress(), e.getMessage(), e);
            }
        }
    }

    private List<Endpoint> initServiceEndpoints(ModelMetaConfig modelMetaConfig) {
        List<Endpoint> endpoints = modelMetaConfig.getServiceRoutes().stream()
                .flatMap(serviceRoute -> serviceRoute.getAllEndpoints().stream())
                .distinct()
                .toList();
        if (CollectionUtils.isEmpty(endpoints)) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG must contain at least one role endpoint");
        }
        return endpoints;
    }

    public void subscribe(Listener listener) {
        if (listener == null || !listeners.addIfAbsent(listener)) {
            return;
        }
        notifyListener(listener, allHosts);
    }

    /**
     * Update host list for specified address and aggregate all address host lists
     *
     * @param endpoint Service endpoint
     * @param hostList Host list
     */
    private void updateEndpointHosts(Endpoint endpoint, List<WorkerHost> hostList) {
        if (hostList == null || hostList.isEmpty()) {
            domainHostsMap.remove(endpoint);
        } else {
            domainHostsMap.put(endpoint, List.copyOf(hostList));
        }
        // Aggregate host lists from all addresses
        List<WorkerHost> aggregatedHosts = new ArrayList<>();
        for (List<WorkerHost> hosts : domainHostsMap.values()) {
            aggregatedHosts.addAll(hosts);
        }
        Logger.info("Address {} hosts updated, total aggregated hosts: {}",
                endpoint.getAddress(), aggregatedHosts.size());
        // Update global host list and notify listener
        this.allHosts = List.copyOf(aggregatedHosts);
        for (Listener listener : listeners) {
            notifyListener(listener, allHosts);
        }
    }

    private void notifyListener(Listener listener, List<WorkerHost> hosts) {
        try {
            listener.onAddressUpdate(hosts);
        } catch (Exception e) {
            Logger.error("Failed to notify engine address listener", e);
        }
    }

    public interface Listener {

        void onAddressUpdate(List<WorkerHost> hosts);
    }
}
