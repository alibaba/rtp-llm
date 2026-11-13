package org.flexlb.util.discovery;

import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Service discovery backed by the static host list in an endpoint configuration.
 */
public final class StaticServiceDiscoveryProvider implements ServiceDiscoveryProvider {
    private final ConcurrentMap<DiscoveryEndpoint, List<DiscoveredHost>> hostsByEndpoint =
            new ConcurrentHashMap<>();

    @Override
    public ServiceDiscoveryType getType() {
        return ServiceDiscoveryType.STATIC_ENV;
    }

    @Override
    public void validate(DiscoveryEndpoint endpoint) {
        validateEndpoint(endpoint);
        configuredHosts(endpoint);
    }

    @Override
    public void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener) {
        if (listener != null) {
            listener.onHostsChanged(getHosts(endpoint));
        }
    }

    @Override
    public List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint) {
        return hostsByEndpoint.get(endpoint);
    }

    @Override
    public void shutdown() {
        // Static discovery owns no external resources.
    }

    private void validateEndpoint(DiscoveryEndpoint endpoint) {
        if (endpoint == null || StringUtils.isBlank(endpoint.getAddress())) {
            throw new IllegalArgumentException("static-env discovery address must not be blank");
        }
        if (endpoint.getDiscovery() == null) {
            throw new IllegalArgumentException(
                    "static-env discovery must be configured for address: "
                            + endpoint.getAddress());
        }
    }

    private List<DiscoveredHost> configuredHosts(DiscoveryEndpoint endpoint) {
        return hostsByEndpoint.computeIfAbsent(endpoint, this::parseConfiguredHosts);
    }

    private List<DiscoveredHost> parseConfiguredHosts(DiscoveryEndpoint endpoint) {
        List<String> hosts = endpoint.getDiscovery().getHosts();
        if (hosts == null || hosts.isEmpty()) {
            throw new IllegalArgumentException(
                    "static-env discovery hosts must be configured for address: "
                            + endpoint.getAddress());
        }
        return parseHosts(hosts);
    }

    private List<DiscoveredHost> parseHosts(List<String> configuredHosts) {
        return configuredHosts.stream().map(this::parseHost).toList();
    }

    private DiscoveredHost parseHost(String hostConfig) {
        if (StringUtils.isBlank(hostConfig)) {
            throw new IllegalArgumentException("static-env discovery host must not be blank");
        }
        int separator = hostConfig.lastIndexOf(':');
        if (separator <= 0 || separator == hostConfig.length() - 1) {
            throw new IllegalArgumentException(
                    "Invalid static-env host: " + hostConfig + ", expected host:port");
        }

        String host = hostConfig.substring(0, separator).trim();
        int port = Integer.parseInt(hostConfig.substring(separator + 1).trim());
        if (StringUtils.isBlank(host)) {
            throw new IllegalArgumentException("Invalid static-env host: " + hostConfig);
        }
        return DiscoveredHost.of(host, port);
    }
}
