package org.flexlb.util.discovery;

import java.util.List;

/**
 * ServiceDiscovery - Service discovery interface
 *
 * @author saichen.sm
 */
public interface ServiceDiscovery {

    /**
     * Validate an endpoint and initialize provider-specific resources.
     *
     * @param endpoint DiscoveryEndpoint and its discovery configuration
     */
    void validate(DiscoveryEndpoint endpoint);

    /**
     * Synchronously get the host list for an endpoint.
     *
     * @param endpoint DiscoveryEndpoint and its discovery configuration
     * @return Host list
     */
    List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint);

    default List<DiscoveredHost> getHosts(String address) {
        throw new UnsupportedOperationException("address-only discovery is not supported");
    }

    /**
     * Listen for host changes for an endpoint.
     *
     * @param endpoint DiscoveryEndpoint and its discovery configuration
     * @param listener Host change listener
     */
    void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener);

    default void listen(String address, ServiceHostListener listener) {
        throw new UnsupportedOperationException("address-only discovery is not supported");
    }

    /**
     * Stop all listeners
     */
    void shutdown();
}
