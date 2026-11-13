package org.flexlb.util.discovery;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.util.network.BackendServiceProtocolEnum;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Routes each endpoint to the provider selected by its discovery configuration.
 */
@Slf4j
public class RoutingServiceDiscovery implements ServiceDiscovery {

    private final Map<ServiceDiscoveryType, ServiceDiscoveryProvider> providers;
    private final Supplier<DiscoveryClientSettings> runtimeConfigSupplier;

    public RoutingServiceDiscovery(List<ServiceDiscoveryProvider> discoveryProviders) {
        this(discoveryProviders, DiscoveryClientSettings::new);
    }

    public RoutingServiceDiscovery(
            List<ServiceDiscoveryProvider> discoveryProviders,
            Supplier<DiscoveryClientSettings> runtimeConfigSupplier) {
        this.providers = new EnumMap<>(ServiceDiscoveryType.class);
        this.runtimeConfigSupplier = Objects.requireNonNull(runtimeConfigSupplier);
        for (ServiceDiscoveryProvider provider : discoveryProviders) {
            ServiceDiscoveryProvider previous = providers.putIfAbsent(provider.getType(), provider);
            if (previous != null) {
                throw new IllegalStateException(
                        "Multiple service discovery providers registered for type: "
                                + provider.getType().getValue());
            }
        }
        log.info("Registered service discovery providers: {}", providers.keySet());
    }

    @Override
    public void validate(DiscoveryEndpoint endpoint) {
        bindRuntimeConfig(endpoint);
        validatedProviderFor(endpoint).validate(endpoint);
    }

    @Override
    public List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint) {
        bindRuntimeConfig(endpoint);
        List<DiscoveredHost> discoveredHosts = validatedProviderFor(endpoint).getHosts(endpoint);
        return normalizeHosts(discoveredHosts, endpoint);
    }

    @Override
    public void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener) {
        bindRuntimeConfig(endpoint);
        ServiceHostListener normalizedListener =
                listener == null
                        ? null
                        : hosts -> listener.onHostsChanged(normalizeHosts(hosts, endpoint));
        validatedProviderFor(endpoint).listen(endpoint, normalizedListener);
    }

    @Override
    public void shutdown() {
        for (ServiceDiscoveryProvider provider : providers.values()) {
            try {
                provider.shutdown();
            } catch (Exception e) {
                log.error(
                        "Failed to shut down service discovery provider: {}",
                        provider.getType().getValue(),
                        e);
            }
        }
    }

    private ServiceDiscoveryProvider validatedProviderFor(DiscoveryEndpoint endpoint) {
        if (endpoint == null) {
            throw new IllegalArgumentException("endpoint must not be null");
        }
        if (StringUtils.isBlank(endpoint.getAddress())) {
            throw new IllegalArgumentException("endpoint address must not be blank");
        }
        DiscoverySettings discovery = endpoint.getDiscovery();
        if (discovery == null) {
            throw new IllegalArgumentException(
                    "endpoint discovery must be configured for address: " + endpoint.getAddress());
        }
        if (discovery.getType() == null) {
            throw new IllegalArgumentException(
                    "endpoint discovery type must be configured for address: "
                            + endpoint.getAddress());
        }
        ServiceDiscoveryProvider provider = providers.get(discovery.getType());
        if (provider == null) {
            throw new IllegalArgumentException(
                    "No service discovery provider available for type: "
                            + discovery.getType().getValue()
                            + ", address: "
                            + endpoint.getAddress());
        }
        return provider;
    }

    private void bindRuntimeConfig(DiscoveryEndpoint endpoint) {
        if (endpoint != null && endpoint.getDiscovery() != null) {
            endpoint.getDiscovery().bindRuntimeConfig(runtimeConfigSupplier);
        }
    }

    private List<DiscoveredHost> normalizeHosts(
            List<DiscoveredHost> discoveredHosts, DiscoveryEndpoint endpoint) {
        if (discoveredHosts == null || discoveredHosts.isEmpty()) {
            return List.of();
        }

        boolean grpcEndpoint =
                BackendServiceProtocolEnum.GRPC.getName().equals(endpoint.getProtocol());
        int multiEngineNum = endpoint.getMultiEngineNum();
        List<DiscoveredHost> normalizedHosts =
                new ArrayList<>(discoveredHosts.size() * multiEngineNum);
        for (DiscoveredHost host : discoveredHosts) {
            int discoveredPort = host.getPort();
            int httpPort = grpcEndpoint ? discoveredPort - 1 : discoveredPort;
            int grpcPort = grpcEndpoint ? discoveredPort : discoveredPort + 1;
            int workerStatusBasePort =
                    endpoint.getWorkerStatusPort() == null
                            ? grpcPort
                            : endpoint.getWorkerStatusPort();
            requireValidPort(httpPort, "httpPort");
            requireValidPort(grpcPort, "grpcPort");
            requireValidPort(httpPort + 5, "httpServerPort");
            requireValidPort(workerStatusBasePort, "workerStatusPort");
            requireValidPort(
                    (long) workerStatusBasePort + multiEngineNum - 1, "lastWorkerStatusPort");
            for (int engineIndex = 0; engineIndex < multiEngineNum; engineIndex++) {
                normalizedHosts.add(
                        new DiscoveredHost(
                                host.getIp(),
                                httpPort,
                                grpcPort,
                                httpPort + 5,
                                workerStatusBasePort + engineIndex,
                                host.getSite(),
                                endpoint.getGroup(),
                                host.getDeploymentName(),
                                engineIndex,
                                multiEngineNum));
            }
        }
        return normalizedHosts;
    }

    private static void requireValidPort(long port, String name) {
        if (port <= 0 || port > 65535) {
            throw new IllegalArgumentException(name + " must be in [1, 65535], received " + port);
        }
    }
}
