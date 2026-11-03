package org.flexlb.mock.master;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.*;
import org.springframework.stereotype.Component;

/** Local topology adapter for the native endpoint discovery mechanism. */
@Component
public final class FileDiscovery implements ServiceDiscoveryProvider {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final Path file;
    private volatile Map<String,List<WorkerHost>> snapshot;

    public FileDiscovery(ConfigService config) {
        file = Path.of(Objects.requireNonNull(config.modelServiceConfig().getDiscoveryFile(), "discovery_file"));
    }
    @Override public ServiceDiscoveryType getType() { return ServiceDiscoveryType.FILE; }
    @Override public void validate(Endpoint endpoint) { getHosts(endpoint); }
    @Override public List<WorkerHost> getHosts(Endpoint endpoint) {
        try {
            var root = JSON.readTree(file.toFile());
            if (root == null || !root.isObject()) throw new IllegalArgumentException("Discovery must be a domain map");
            var next = new HashMap<String,List<WorkerHost>>();
            var entries = root.fields();
            while (entries.hasNext()) {
                var entry = entries.next();
                if (!entry.getValue().isArray()) throw new IllegalArgumentException("Discovery hosts must be an array");
                var hosts = new ArrayList<WorkerHost>();
                for (var value : entry.getValue()) {
                    if (!value.isTextual()) throw new IllegalArgumentException("Discovery host must be text");
                    var address = java.net.URI.create("tcp://" + value.asText());
                    if (address.getHost() == null || address.getPort() < 2 || address.getPort() > 65531
                        || address.getUserInfo() != null || !address.getPath().isEmpty())
                        throw new IllegalArgumentException("Invalid discovery host: " + value.asText());
                    hosts.add(WorkerHost.of(address.getHost(), address.getPort(), ""));
                }
                next.put(entry.getKey(), List.copyOf(hosts));
            }
            snapshot = Map.copyOf(next);
        } catch (Exception error) {
            if (snapshot == null) throw new IllegalStateException("Cannot read discovery file: " + file, error);
        }
        return snapshot.getOrDefault(endpoint.getAddress(), List.of());
    }
    @Override public void listen(Endpoint endpoint, ServiceHostListener listener) {
        if (listener != null) listener.onHostsChanged(getHosts(endpoint));
    }
    @Override public void shutdown() { }
}
