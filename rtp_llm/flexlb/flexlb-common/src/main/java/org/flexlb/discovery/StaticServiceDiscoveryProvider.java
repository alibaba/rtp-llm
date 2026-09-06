package org.flexlb.discovery;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Service discovery backed by the static host list in an endpoint configuration.
 */
public final class StaticServiceDiscoveryProvider implements ServiceDiscoveryProvider {

    private final ConcurrentMap<Endpoint, List<WorkerHost>> hostsByEndpoint = new ConcurrentHashMap<>();

    @Override
    public ServiceDiscoveryType getType() {
        return ServiceDiscoveryType.STATIC_ENV;
    }

    @Override
    public void validate(Endpoint endpoint) {
        validateEndpoint(endpoint);
        configuredHosts(endpoint);
    }

    @Override
    public List<WorkerHost> getHosts(Endpoint endpoint) {
        return hostsByEndpoint.get(endpoint);
    }

    private void validateEndpoint(Endpoint endpoint) {
        if (endpoint == null || StringUtils.isBlank(endpoint.getAddress())) {
            throw new IllegalArgumentException("static-env discovery address must not be blank");
        }
        if (endpoint.getDiscovery() == null) {
            throw new IllegalArgumentException(
                    "static-env discovery must be configured for address: " + endpoint.getAddress());
        }
    }

    @Override
    public void listen(Endpoint endpoint, ServiceHostListener listener) {
        if (listener != null) {
            listener.onHostsChanged(getHosts(endpoint));
        }
    }

    @Override
    public void shutdown() {
        // Static discovery owns no external resources.
    }

    private List<WorkerHost> configuredHosts(Endpoint endpoint) {
        return hostsByEndpoint.computeIfAbsent(endpoint, this::parseConfiguredHosts);
    }

    private List<WorkerHost> parseConfiguredHosts(Endpoint endpoint) {
        List<String> hosts = endpoint.getDiscovery().getHosts();
        if (hosts == null || hosts.isEmpty()) {
            throw new IllegalArgumentException(
                    "static-env discovery hosts must be configured for address: " + endpoint.getAddress());
        }
        return parseHosts(hosts);
    }

    private List<WorkerHost> parseHosts(List<String> configuredHosts) {
        return configuredHosts.stream()
                .map(this::parseHost)
                .toList();
    }

    private WorkerHost parseHost(String hostConfig) {
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
        return WorkerHost.of(host, port);
    }
}
