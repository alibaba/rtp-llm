package org.flexlb.mock.master;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.net.HostAndPort;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.ServiceDiscoveryProvider;
import org.flexlb.util.discovery.ServiceDiscoveryType;
import org.flexlb.util.discovery.ServiceHostListener;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Provides Engine endpoints from the Mock Runner's local discovery file.
 * Delivers the initial snapshot once and reloads the file on each subsequent lookup.
 */
@Slf4j
@Component
public final class LocalFileServiceDiscoveryProvider implements ServiceDiscoveryProvider {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Rate limit for the repeated fallback debug log (per instance). */
    private static final long FALLBACK_LOG_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(5);

    private final Path file;

    /** Last fully-parsed snapshot; replaced wholesale on every successful read. */
    private volatile Map<String, List<DiscoveredHost>> lastGoodSnapshot;

    private final AtomicLong lastFallbackLogNanos = new AtomicLong();

    public LocalFileServiceDiscoveryProvider(ConfigService configService) {
        file = Path.of(configService.modelServiceConfig().getDiscoveryFile());
    }

    @Override
    public ServiceDiscoveryType getType() {
        return ServiceDiscoveryType.LOCAL_FILE;
    }

    @Override
    public void validate(DiscoveryEndpoint endpoint) {
        getHosts(endpoint);
    }

    @Override
    public List<DiscoveredHost> getHosts(DiscoveryEndpoint endpoint) {
        String address = endpoint.getAddress();
        if (StringUtils.isBlank(address)) {
            return List.of();
        }
        try {
            lastGoodSnapshot = parseHosts(MAPPER.readTree(Files.readString(file)));
        } catch (Exception e) {
            if (lastGoodSnapshot == null) {
                throw new IllegalStateException(
                        "Failed to read discovery file with no previous snapshot: " + file, e);
            }
            logFallbackOnce(e);
        }
        return lastGoodSnapshot.getOrDefault(address, List.of());
    }

    @Override
    public void listen(DiscoveryEndpoint endpoint, ServiceHostListener listener) {
        // Deliver the initial snapshot. Subsequent updates are read through getHosts().
        if (listener != null) {
            listener.onHostsChanged(getHosts(endpoint));
        }
    }

    @Override
    public void shutdown() {}

    /**
     * Parse the whole file into a domain → hosts map. Any failure (I/O,
     * malformed JSON, wrong shape, invalid host entry) aborts the ENTIRE parse
     * so callers can never observe a half list.
     */
    private static Map<String, List<DiscoveredHost>> parseHosts(JsonNode root) {
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("discovery hosts must be an object");
        }
        Map<String, List<DiscoveredHost>> result = new LinkedHashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            JsonNode array = entry.getValue();
            if (array == null || !array.isArray()) {
                throw new IllegalArgumentException(
                        String.format(
                                "discovery entry for domain '%s' must be an array of ip:port strings",
                                entry.getKey()));
            }
            List<DiscoveredHost> hosts = new ArrayList<>(array.size());
            for (JsonNode node : array) {
                if (node == null || !node.isTextual()) {
                    throw new IllegalArgumentException(
                            String.format(
                                    "discovery entry for domain '%s' contains a non-string element",
                                    entry.getKey()));
                }
                hosts.add(parseHost(node.asText()));
            }
            result.put(entry.getKey(), List.copyOf(hosts));
        }
        return Map.copyOf(result);
    }

    /** Port interpretation follows the endpoint protocol. */
    private static DiscoveredHost parseHost(String hostStr) {
        HostAndPort address = HostAndPort.fromString(hostStr.trim());
        if (address.getHost().isBlank() || !address.hasPort() || address.getPort() == 0) {
            throw new IllegalArgumentException("Invalid host:port: " + hostStr);
        }
        return DiscoveredHost.of(address.getHost(), address.getPort());
    }

    /** Debug-log a read fallback at most once per interval (avoids log flooding at 20ms poll cadence). */
    private void logFallbackOnce(Exception cause) {
        long now = System.nanoTime();
        long last = lastFallbackLogNanos.get();
        if (now - last >= FALLBACK_LOG_INTERVAL_NANOS
                && lastFallbackLogNanos.compareAndSet(last, now)) {
            log.debug(
                    "Local file discovery read failed, serving last good snapshot, file={}, cause={}",
                    file,
                    cause.getMessage());
        }
    }
}
