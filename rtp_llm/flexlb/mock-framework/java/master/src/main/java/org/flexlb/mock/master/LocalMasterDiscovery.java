package org.flexlb.mock.master;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads runner-owned discovery files on each native EngineSyncRunner poll.
 */
@Configuration
public class LocalMasterDiscovery {
    @Bean(destroyMethod = "shutdown")
    @Primary
    public RoutingServiceDiscovery localFileDiscovery(
            ConfigService config, List<ServiceDiscoveryProvider> providers) {
        var selected = new ArrayList<ServiceDiscoveryProvider>();
        for (var provider : providers) {
            if (provider.getType() != ServiceDiscoveryType.STATIC_ENV) selected.add(provider);
        }
        selected.add(new FileHosts(Path.of(config.modelServiceConfig().getDiscoveryFile())));
        return new RoutingServiceDiscovery(selected);
    }

    private static final class FileHosts implements ServiceDiscoveryProvider {
        private final Path path;
        private final ObjectMapper json = new ObjectMapper();
        private final StaticServiceDiscoveryProvider configured =
                new StaticServiceDiscoveryProvider();

        FileHosts(Path path) {
            this.path = path;
        }

        @Override
        public ServiceDiscoveryType getType() {
            return ServiceDiscoveryType.STATIC_ENV;
        }

        @Override
        public void validate(Endpoint endpoint) {
            getHosts(endpoint);
        }

        private JsonNode addresses(Endpoint endpoint) {
            try {
                return json.readTree(path.toFile()).path(endpoint.getAddress());
            } catch (java.io.IOException error) {
                throw new IllegalStateException("Cannot read discovery file " + path, error);
            }
        }

        @Override
        public List<WorkerHost> getHosts(Endpoint endpoint) {
            JsonNode addresses = addresses(endpoint);
            if (addresses.isMissingNode()) {
                configured.validate(endpoint);
                return configured.getHosts(endpoint);
            }
            if (!addresses.isArray())
                throw new IllegalArgumentException(
                        "Expected address list for " + endpoint.getAddress());
            var hosts = new ArrayList<WorkerHost>();
            for (var value : addresses) {
                String address = value.asText();
                int separator = address.lastIndexOf(':');
                if (separator <= 0)
                    throw new IllegalArgumentException("Invalid Engine address " + address);
                hosts.add(
                        WorkerHost.of(
                                address.substring(0, separator),
                                Integer.parseInt(address.substring(separator + 1))));
            }
            return List.copyOf(hosts);
        }

        @Override
        public void listen(Endpoint endpoint, ServiceHostListener listener) {
            if (listener != null) listener.onHostsChanged(getHosts(endpoint));
        }

        @Override
        public void shutdown() {
            configured.shutdown();
        }
    }
}
