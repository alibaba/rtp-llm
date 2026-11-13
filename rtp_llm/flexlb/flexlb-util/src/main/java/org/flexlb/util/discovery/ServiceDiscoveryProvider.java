package org.flexlb.util.discovery;

import java.util.List;

/**
 * Provider for one service discovery type.
 */
public interface ServiceDiscoveryProvider {

    ServiceDiscoveryType getType();

    void validate(DiscoveryEndpoint endpoint);

    List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint);

    void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener);

    void shutdown();
}
