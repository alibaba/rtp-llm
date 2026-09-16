package org.flexlb.discovery;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.master.WorkerHost;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * FileServiceDiscovery - File-based service discovery with dynamic reload.
 *
 * <p>Replaces the env-var-only {@link NoOpServiceDiscovery} for mock/test
 * deployments: the domain → hosts mapping lives in a small JSON file that an
 * external orchestrator (or the mock engine control plane) rewrites atomically
 * (tmp file + rename). Every {@link #getHosts(String)} call re-reads the file
 * so changes are picked up by the master's periodic sync loop without restart.
 *
 * <p>File format (compact, aligned with the endpoints.json semantics — entries
 * carry the HTTP port, grpc = http + 1):
 * <pre>{@code
 * {
 *   "mock.prefill.hosts.address": ["127.0.0.1:55150", "127.0.0.1:55151"],
 *   "mock.decode.hosts.address":  ["127.0.0.1:55160"]
 * }
 * }</pre>
 *
 * <p>Read-side policy: a parse failure (missing file, corrupted JSON, partial
 * write window, invalid entry) NEVER returns a half list — the last fully
 * parsed snapshot is served instead (debug-logged, rate-limited). If the very
 * first read fails there is no fallback, so an {@link IllegalStateException}
 * is thrown to surface the misconfiguration instead of silently pretending
 * the service has no workers.
 */
@Slf4j
public final class FileServiceDiscovery implements ServiceDiscovery {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** Rate limit for the repeated fallback debug log (per instance). */
    private static final long FALLBACK_LOG_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(5);

    private final Path file;
    /** Last fully-parsed snapshot; replaced wholesale on every successful read. */
    private volatile Map<String, List<WorkerHost>> lastGoodSnapshot = Map.of();
    private final AtomicLong lastFallbackLogNanos = new AtomicLong();

    public FileServiceDiscovery(String filePath) {
        this(filePath == null ? null : Path.of(filePath));
    }

    public FileServiceDiscovery(Path file) {
        this.file = file;
    }

    @Override
    public List<WorkerHost> getHosts(String address) {
        if (StringUtils.isBlank(address)) {
            log.warn("Service address is blank, returning empty host list");
            return Collections.emptyList();
        }
        Map<String, List<WorkerHost>> snapshot;
        try {
            snapshot = parseFile();
            lastGoodSnapshot = snapshot;
        } catch (Exception e) {
            snapshot = lastGoodSnapshot;
            if (snapshot.isEmpty()) {
                throw new IllegalStateException(String.format(
                        "FileServiceDiscovery failed to read discovery file and no previous snapshot "
                                + "exists for fallback, address=%s, file=%s, cause=%s",
                        address, file, e.getMessage()), e);
            }
            logFallbackOnce(e);
        }
        List<WorkerHost> hosts = snapshot.get(address);
        if (hosts == null) {
            log.warn("No hosts entry found for address: {} in discovery file: {}", address, file);
            return Collections.emptyList();
        }
        return hosts;
    }

    @Override
    public void listen(String address, ServiceHostListener listener) {
        log.info("FileServiceDiscovery relies on per-call re-read for address: {} (no push listener)", address);
        // Same contract as NoOpServiceDiscovery: no dynamic push, but trigger the
        // listener once with the current view so initial wiring completes.
        if (listener != null) {
            listener.onHostsChanged(getHosts(address));
        }
    }

    @Override
    public void shutdown() {
        log.info("FileServiceDiscovery shutdown (file={})", file);
        // No background resources — nothing to release.
    }

    /**
     * Parse the whole file into a domain → hosts map. Any failure (I/O,
     * malformed JSON, wrong shape, invalid host entry) aborts the ENTIRE parse
     * so callers can never observe a half list.
     */
    private Map<String, List<WorkerHost>> parseFile() throws Exception {
        String content = Files.readString(file);
        JsonNode root = MAPPER.readTree(content);
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException("discovery file root must be a JSON object: " + file);
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
        return result;
    }

    /** Parse an {@code ip:port} string (HTTP port — grpc port = http + 1). */
    private static WorkerHost parseHost(String hostStr) {
        String[] parts = hostStr.split(":");
        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid host format: " + hostStr + ", expected ip:port");
        }
        String ip = parts[0].trim();
        int port = Integer.parseInt(parts[1].trim());
        return WorkerHost.of(ip, port);
    }

    /** Debug-log a read fallback at most once per interval (avoids log flooding at 20ms poll cadence). */
    private void logFallbackOnce(Exception cause) {
        long now = System.nanoTime();
        long last = lastFallbackLogNanos.get();
        if (now - last >= FALLBACK_LOG_INTERVAL_NANOS
                && lastFallbackLogNanos.compareAndSet(last, now)) {
            log.debug("FileServiceDiscovery read failed, serving last good snapshot, file={}, cause={}",
                    file, cause.getMessage());
        }
    }
}
