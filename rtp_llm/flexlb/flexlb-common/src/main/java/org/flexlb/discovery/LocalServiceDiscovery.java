package org.flexlb.discovery;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/** Local domain-to-host discovery from a configured mapping or a reloadable JSON file. */
@Slf4j
public final class LocalServiceDiscovery implements ServiceDiscovery {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** Rate limit for the repeated fallback debug log (per instance). */
    private static final long FALLBACK_LOG_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(5);

    private final Path file;
    /** Last fully-parsed snapshot; replaced wholesale on every successful read. */
    private volatile Map<String, List<WorkerHost>> lastGoodSnapshot;
    private final AtomicLong lastFallbackLogNanos = new AtomicLong();

    public LocalServiceDiscovery(String filePath) {
        this(Path.of(filePath));
    }

    public LocalServiceDiscovery(Path file) {
        this.file = Objects.requireNonNull(file, "discovery file");
    }

    public LocalServiceDiscovery(Map<String, List<String>> hosts) {
        this.file = null;
        this.lastGoodSnapshot = parseHosts(MAPPER.valueToTree(hosts));
    }

    @Override
    public List<WorkerHost> getHosts(String address) {
        if (StringUtils.isBlank(address)) {
            return List.of();
        }
        if (file != null) {
            try {
                lastGoodSnapshot = parseHosts(MAPPER.readTree(Files.readString(file)));
            } catch (Exception e) {
                if (lastGoodSnapshot == null) {
                    throw new IllegalStateException(
                            "Failed to read discovery file with no previous snapshot: " + file, e);
                }
                logFallbackOnce(e);
            }
        }
        return lastGoodSnapshot.getOrDefault(address, List.of());
    }

    @Override
    public void listen(String address, ServiceHostListener listener) {
        // Notify the listener once with the current view.
        if (listener != null) {
            listener.onHostsChanged(getHosts(address));
        }
    }

    @Override
    public void shutdown() {
        log.info("LocalServiceDiscovery shutdown (file={})", file);
        // No background resources — nothing to release.
    }

    /**
     * Parse the whole file into a domain → hosts map. Any failure (I/O,
     * malformed JSON, wrong shape, invalid host entry) aborts the ENTIRE parse
     * so callers can never observe a half list.
     */
    private static Map<String, List<WorkerHost>> parseHosts(JsonNode root) {
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("discovery hosts must be an object");
        }
        Map<String, List<WorkerHost>> result = new LinkedHashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = root.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            JsonNode array = entry.getValue();
            if (array == null || !array.isArray()) {
                throw new IllegalArgumentException(String.format(
                        "discovery entry for domain '%s' must be an array of ip:port strings", entry.getKey()));
            }
            List<WorkerHost> hosts = new ArrayList<>(array.size());
            for (JsonNode node : array) {
                if (node == null || !node.isTextual()) {
                    throw new IllegalArgumentException(String.format(
                            "discovery entry for domain '%s' contains a non-string element", entry.getKey()));
                }
                hosts.add(parseHost(node.asText()));
            }
            result.put(entry.getKey(), List.copyOf(hosts));
        }
        return Map.copyOf(result);
    }

    /** Port interpretation follows the endpoint protocol. */
    private static WorkerHost parseHost(String hostStr) {
        String[] parts = hostStr.split(":");
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid host format: " + hostStr + ", expected ip:port");
        }
        String ip = parts[0].trim();
        int port = Integer.parseInt(parts[1].trim());
        if (ip.isEmpty() || port < 1 || port > 65535) {
            throw new IllegalArgumentException("Invalid host:port: " + hostStr);
        }
        return WorkerHost.of(ip, port);
    }

    /** Debug-log a read fallback at most once per interval (avoids log flooding at 20ms poll cadence). */
    private void logFallbackOnce(Exception cause) {
        long now = System.nanoTime();
        long last = lastFallbackLogNanos.get();
        if (now - last >= FALLBACK_LOG_INTERVAL_NANOS
                && lastFallbackLogNanos.compareAndSet(last, now)) {
            log.debug("LocalServiceDiscovery read failed, serving last good snapshot, file={}, cause={}",
                    file, cause.getMessage());
        }
    }
}
