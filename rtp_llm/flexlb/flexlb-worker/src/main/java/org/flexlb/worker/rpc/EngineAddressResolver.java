package org.flexlb.worker.rpc;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.util.Logger;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoverySettings;
import org.flexlb.util.discovery.ServiceDiscovery;
import org.flexlb.util.discovery.ServiceDiscoveryType;
import org.flexlb.util.discovery.ServiceHostListener;
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
 *
 * <p>Initialization resolves configured service endpoints and installs discovery listeners.
 * Periodic refresh updates the domain hosts, then subscriptions receive immutable host lists
 * through the resolver callbacks.
 */
@Slf4j
@Component
public class EngineAddressResolver {
    private final Map<DiscoveryEndpoint, List<DiscoveredHost>> domainHostsMap =
            new ConcurrentHashMap<>();
    private final ServiceDiscovery serviceDiscovery;
    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();

    private volatile List<DiscoveredHost> allHosts = List.of();

    private final List<DiscoveryEndpoint> serviceEndpoints;

    public EngineAddressResolver(
            ServiceDiscovery serviceDiscovery, ModelMetaConfig modelMetaConfig) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceEndpoints = initServiceEndpoints(modelMetaConfig);
        log.info("EngineAddressResolver start subscribe endpoints:{} ", serviceEndpoints);
        initializeDomainHosts();
        setupListeners(serviceDiscovery, serviceEndpoints);
    }

    @Scheduled(fixedDelay = 30000) // Execute every 30 seconds
    public void periodicHostUpdate() {
        Logger.info(
                "EngineAddressResolver performing periodic host update for endpoints: {}",
                serviceEndpoints);
        fetchAllDomainsHosts();
    }

    public void subscribe(Listener listener) {
        if (listener == null || !listeners.addIfAbsent(listener)) {
            return;
        }
        notifyListener(listener, allHosts);
    }

    private void setupListeners(
            ServiceDiscovery serviceDiscovery, List<DiscoveryEndpoint> endpoints) {
        for (DiscoveryEndpoint endpoint : endpoints) {
            ServiceHostListener addressListener = hosts -> updateEndpointHosts(endpoint, hosts);
            serviceDiscovery.listen(endpoint, addressListener);
        }
    }

    /**
     * Update host list for specified address and aggregate all address host lists
     *
     * @param endpoint Service endpoint
     * @param hostList Host list
     */
    private void updateEndpointHosts(DiscoveryEndpoint endpoint, List<DiscoveredHost> hostList) {
        if (hostList == null || hostList.isEmpty()) {
            // Empty discovery must not retire existing channels or worker caches.
            return;
        }
        domainHostsMap.put(endpoint, List.copyOf(hostList));
        // Aggregate host lists from all addresses
        List<DiscoveredHost> aggregatedHosts = new ArrayList<>();
        for (List<DiscoveredHost> hosts : domainHostsMap.values()) {
            aggregatedHosts.addAll(hosts);
        }
        Logger.info(
                "Address {} hosts updated, total aggregated hosts: {}",
                endpoint.getAddress(),
                aggregatedHosts.size());
        // Update global host list and notify listener
        this.allHosts = List.copyOf(aggregatedHosts);
        for (Listener listener : listeners) {
            notifyListener(listener, allHosts);
        }
    }

    private void notifyListener(Listener listener, List<DiscoveredHost> hosts) {
        try {
            listener.onAddressUpdate(hosts);
        } catch (Exception e) {
            Logger.error("Failed to notify engine address listener", e);
        }
    }

    private void initializeDomainHosts() {
        Map<DiscoveryEndpoint, List<DiscoveredHost>> initialHosts = new LinkedHashMap<>();
        for (DiscoveryEndpoint endpoint : serviceEndpoints) {
            List<DiscoveredHost> hosts;
            try {
                hosts = serviceDiscovery.getHosts(endpoint);
            } catch (Exception e) {
                throw new IllegalStateException(
                        "Failed to fetch initial engine hosts for endpoint: "
                                + endpoint.getAddress(),
                        e);
            }
            if (hosts == null || hosts.isEmpty()) {
                throw new IllegalStateException(
                        "No initial engine hosts discovered for endpoint: "
                                + endpoint.getAddress());
            }
            initialHosts.put(endpoint, List.copyOf(hosts));
        }
        initialHosts.forEach(this::updateEndpointHosts);
    }

    private void fetchAllDomainsHosts() {
        for (DiscoveryEndpoint endpoint : serviceEndpoints) {
            try {
                List<DiscoveredHost> hosts = serviceDiscovery.getHosts(endpoint);
                Logger.info(
                        "Fetched {} hosts for address: {}",
                        hosts != null ? hosts.size() : 0,
                        endpoint.getAddress());
                updateEndpointHosts(endpoint, hosts);
            } catch (Exception e) {
                Logger.error(
                        "Failed to fetch hosts for address: {}, error: {}",
                        endpoint.getAddress(),
                        e.getMessage(),
                        e);
            }
        }
    }

    private List<DiscoveryEndpoint> initServiceEndpoints(ModelMetaConfig modelMetaConfig) {
        Map<AddressSource, DiscoveryEndpoint> uniqueSources = new LinkedHashMap<>();
        modelMetaConfig.getServiceRoutes().stream()
                .flatMap(serviceRoute -> serviceRoute.getAllEndpoints().stream())
                .forEach(
                        endpoint ->
                                uniqueSources.putIfAbsent(AddressSource.from(endpoint), endpoint));
        List<DiscoveryEndpoint> endpoints = List.copyOf(uniqueSources.values());
        if (CollectionUtils.isEmpty(endpoints)) {
            throw new IllegalArgumentException(
                    "MODEL_SERVICE_CONFIG must contain at least one role endpoint");
        }
        return endpoints;
    }

    /**
     * Freezes discovery-provider inputs so equivalent sources share endpoint resolution.
     */
    private record DiscoverySource(ServiceDiscoveryType type, String baseUrl, List<String> hosts) {
        private static DiscoverySource from(DiscoverySettings discovery) {
            return discovery == null
                    ? null
                    : new DiscoverySource(
                            discovery.getType(),
                            discovery.getBaseUrl(),
                            discovery.getHosts() == null
                                    ? List.of()
                                    : List.copyOf(discovery.getHosts()));
        }
    }

    /**
     * Defines the endpoint fields that distinguish independent discovery subscriptions.
     */
    private record AddressSource(
            String address,
            String protocol,
            String path,
            Integer workerStatusPort,
            int multiEngineNum,
            DiscoverySource discovery) {
        private static AddressSource from(DiscoveryEndpoint endpoint) {
            return new AddressSource(
                    endpoint.getAddress(),
                    endpoint.getProtocol(),
                    endpoint.getPath(),
                    endpoint.getWorkerStatusPort(),
                    endpoint.getMultiEngineNum(),
                    DiscoverySource.from(endpoint.getDiscovery()));
        }
    }

    /**
     * Receives a complete current host list after discovery changes.
     */
    public interface Listener {

        void onAddressUpdate(List<DiscoveredHost> hosts);
    }
}
