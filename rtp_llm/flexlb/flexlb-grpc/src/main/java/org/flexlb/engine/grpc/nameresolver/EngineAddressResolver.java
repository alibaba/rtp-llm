package org.flexlb.engine.grpc.nameresolver;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
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
        initializeDomainHosts();
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

    private void initializeDomainHosts() {
        Map<Endpoint, List<WorkerHost>> initialHosts = new LinkedHashMap<>();
        for (Endpoint endpoint : serviceEndpoints) {
            List<WorkerHost> hosts;
            try {
                hosts = serviceDiscovery.getHosts(endpoint);
            } catch (Exception e) {
                throw new IllegalStateException(
                        "Failed to fetch initial engine hosts for endpoint: " + endpoint.getAddress(), e);
            }
            if (hosts == null || hosts.isEmpty()) {
                throw new IllegalStateException(
                        "No initial engine hosts discovered for endpoint: " + endpoint.getAddress());
            }
            initialHosts.put(endpoint, List.copyOf(hosts));
        }
        initialHosts.forEach(this::updateEndpointHosts);
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
        Map<AddressSource, Endpoint> uniqueSources = new LinkedHashMap<>();
        modelMetaConfig.getServiceRoutes().stream()
                .flatMap(serviceRoute -> serviceRoute.getAllEndpoints().stream())
                .forEach(endpoint -> uniqueSources.putIfAbsent(AddressSource.from(endpoint), endpoint));
        List<Endpoint> endpoints = List.copyOf(uniqueSources.values());
        if (CollectionUtils.isEmpty(endpoints)) {
            throw new IllegalArgumentException("MODEL_SERVICE_CONFIG must contain at least one role endpoint");
        }
        return endpoints;
    }

    private record DiscoverySource(ServiceDiscoveryType type, String baseUrl, List<String> hosts) {
        private static DiscoverySource from(DiscoveryConfig discovery) {
            return discovery == null ? null : new DiscoverySource(
                    discovery.getType(), discovery.getBaseUrl(),
                    discovery.getHosts() == null ? List.of() : List.copyOf(discovery.getHosts()));
        }
    }

    private record AddressSource(String address, String protocol, String path,
                                 Integer workerStatusPort, int multiEngineNum, DiscoverySource discovery) {
        private static AddressSource from(Endpoint endpoint) {
            return new AddressSource(endpoint.getAddress(), endpoint.getProtocol(), endpoint.getPath(),
                    endpoint.getWorkerStatusPort(), endpoint.getMultiEngineNum(),
                    DiscoverySource.from(endpoint.getDiscovery()));
        }
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
            // Empty discovery must not retire existing channels or worker caches.
            return;
        }
        domainHostsMap.put(endpoint, List.copyOf(hostList));
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
