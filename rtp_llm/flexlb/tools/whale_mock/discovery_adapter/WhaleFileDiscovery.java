package org.flexlb.mockdiscovery;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Test bundle extension, compiled against the pinned master. No production discovery replacement. */
@Component
@Primary
@ConditionalOnProperty(name = "mock.discovery.file")
public final class WhaleFileDiscovery implements ServiceDiscovery {
    private final Path file;
    private final ObjectMapper mapper = new ObjectMapper();
    private volatile Map<String, List<WorkerHost>> snapshot;

    public WhaleFileDiscovery(@Value("${mock.discovery.file}") String path) {
        file = Path.of(path);
    }

    @Override
    public synchronized List<WorkerHost> getHosts(String address) {
        try {
            JsonNode root = mapper.readTree(Files.readString(file));
            if (root == null || !root.isObject()) throw new IllegalArgumentException("Expected domain map");
            Map<String, List<WorkerHost>> next = new LinkedHashMap<>();
            var fields = root.fields();
            while (fields.hasNext()) {
                var entry = fields.next();
                if (!entry.getValue().isArray()) throw new IllegalArgumentException("Expected host array");
                List<WorkerHost> hosts = new ArrayList<>();
                for (JsonNode value : entry.getValue()) {
                    if (!value.isTextual()) throw new IllegalArgumentException("Expected host string");
                    String[] parts = value.asText().split(":", -1);
                    if (parts.length != 2 || parts[0].isBlank()) throw new IllegalArgumentException("Invalid host");
                    int port = Integer.parseInt(parts[1]);
                    if (port < 1 || port > 65535) throw new IllegalArgumentException("Invalid port");
                    hosts.add(WorkerHost.of(parts[0], port));
                }
                next.put(entry.getKey(), List.copyOf(hosts));
            }
            snapshot = Map.copyOf(next);
        } catch (Exception error) {
            // A partial/bad write must not remove every live worker. First load fails closed.
            if (snapshot == null) throw new IllegalStateException("Cannot load mock discovery: " + file, error);
        }
        return snapshot.getOrDefault(address, List.of());
    }

    @Override
    public void listen(String address, ServiceHostListener listener) {
        if (listener != null) listener.onHostsChanged(getHosts(address));
        // The unchanged master's WorkerAddressService periodically calls getHosts.
    }

    @Override public void shutdown() { }
}
