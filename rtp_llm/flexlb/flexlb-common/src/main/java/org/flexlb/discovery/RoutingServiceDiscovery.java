package org.flexlb.discovery;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Routes each endpoint to the provider selected by its discovery configuration.
 */
@Slf4j
public class RoutingServiceDiscovery implements ServiceDiscovery {

    private final Map<ServiceDiscoveryType, ServiceDiscoveryProvider> providers;

    public RoutingServiceDiscovery(List<ServiceDiscoveryProvider> discoveryProviders) {
        this.providers = new EnumMap<>(ServiceDiscoveryType.class);
        for (ServiceDiscoveryProvider provider : discoveryProviders) {
            ServiceDiscoveryProvider previous = providers.putIfAbsent(provider.getType(), provider);
            if (previous != null) {
                throw new IllegalStateException(
                        "Multiple service discovery providers registered for type: " + provider.getType().getValue());
            }
        }
        log.info("Registered service discovery providers: {}", providers.keySet());
    }

    public void validate(Endpoint endpoint) {
        validatedProviderFor(endpoint).validate(endpoint);
    }

    @Override
    public List<WorkerHost> getHosts(Endpoint endpoint) {
        return providers.get(endpoint.getDiscovery().getType()).getHosts(endpoint);
    }

    @Override
    public void listen(Endpoint endpoint, ServiceHostListener listener) {
        providers.get(endpoint.getDiscovery().getType()).listen(endpoint, listener);
    }

    @Override
    public void shutdown() {
        for (ServiceDiscoveryProvider provider : providers.values()) {
            try {
                provider.shutdown();
            } catch (Exception e) {
                log.error("Failed to shut down service discovery provider: {}", provider.getType().getValue(), e);
            }
        }
    }

    private ServiceDiscoveryProvider validatedProviderFor(Endpoint endpoint) {
        if (endpoint == null) {
            throw new IllegalArgumentException("endpoint must not be null");
        }
        if (StringUtils.isBlank(endpoint.getAddress())) {
            throw new IllegalArgumentException("endpoint address must not be blank");
        }
        DiscoveryConfig discovery = endpoint.getDiscovery();
        if (discovery == null) {
            throw new IllegalArgumentException(
                    "endpoint discovery must be configured for address: " + endpoint.getAddress());
        }
        if (discovery.getType() == null) {
            throw new IllegalArgumentException(
                    "endpoint discovery type must be configured for address: " + endpoint.getAddress());
        }
        ServiceDiscoveryProvider provider = providers.get(discovery.getType());
        if (provider == null) {
            throw new IllegalArgumentException(
                    "No service discovery provider available for type: " + discovery.getType().getValue()
                            + ", address: " + endpoint.getAddress());
        }
        return provider;
    }
}
