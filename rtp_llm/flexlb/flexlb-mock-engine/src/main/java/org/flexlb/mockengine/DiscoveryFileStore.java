package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Writer-side owner of the file-based service discovery mapping consumed by
 * {@code org.flexlb.discovery.FileServiceDiscovery} on the master side.
 *
 * <p>The file is derived wholesale from the live services map on every
 * mutation (add/remove engine), so its content can never drift from the set
 * of engines actually hosted by this cluster process:
 *
 * <pre>{@code
 * {
 *   "mock.prefill.hosts.address": ["127.0.0.1:60999", "127.0.0.1:61000"],
 *   "mock.decode.hosts.address":  ["127.0.0.1:61001"]
 * }
 * }</pre>
 *
 * <p>Entries carry the HTTP port (grpc port − 1), matching the legacy
 * DOMAIN_ADDRESS env values produced by {@code writeDiscoveryFiles} and the
 * http→grpc (+1) conversion in {@code WorkerAddressService}.
 *
 * <p>Atomicity: write {@code <path>.tmp} in the same directory, then
 * {@code Files.move(ATOMIC_MOVE, REPLACE_EXISTING)}. A single in-process lock
 * serializes writers (add/remove engine hold it across the whole
 * mutate-services + rewrite-file critical section). Readers re-parse the file
 * on every poll; if they ever catch a torn window they fall back to their last
 * good snapshot, and ATOMIC_MOVE makes even that virtually impossible on the
 * same filesystem.
 */
final class DiscoveryFileStore {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** How many times a rewrite is retried before surfacing the failure to the caller. */
    private static final int WRITE_ATTEMPTS = 3;

    private final Path file;
    private final String prefillDomain;
    private final String decodeDomain;
    /** Serializes all writers within this process (cluster-single lock). */
    private final Object writeLock = new Object();
    private volatile boolean warnedNonAtomic = false;

    DiscoveryFileStore(String filePath, String prefillDomain, String decodeDomain) {
        this.file = Path.of(filePath);
        this.prefillDomain = prefillDomain;
        this.decodeDomain = decodeDomain;
    }

    Path getFile() {
        return file;
    }

    /**
     * Rebuild the discovery file from the current services map. Called from
     * the add/remove critical section, so the services map is stable while it
     * runs (the inner lock additionally serializes any other writers).
     */
    void rewrite(Map<Integer, JavaMockEngineCluster.FastRpcService> services) throws IOException {
        synchronized (writeLock) {
            Map<String, List<String>> payload = new LinkedHashMap<>();
            payload.put(prefillDomain, addressList(services, "PREFILL"));
            payload.put(decodeDomain, addressList(services, "DECODE"));
            atomicWriteWithRetry(payload);
        }
    }

    /** All engines of one role as {@code ip:httpPort} strings, ordered by gRPC port. */
    private List<String> addressList(Map<Integer, JavaMockEngineCluster.FastRpcService> services, String role) {
        List<JavaMockEngineCluster.FastRpcService> engines = new ArrayList<>();
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            if (role.equals(service.getRoleName())) {
                engines.add(service);
            }
        }
        engines.sort(Comparator.comparingInt(JavaMockEngineCluster.FastRpcService::getGrpcPort));
        List<String> addresses = new ArrayList<>(engines.size());
        for (JavaMockEngineCluster.FastRpcService service : engines) {
            addresses.add(service.getHost() + ":" + (service.getGrpcPort() - 1));
        }
        return addresses;
    }

    private void atomicWriteWithRetry(Map<String, List<String>> payload) throws IOException {
        IOException last = null;
        for (int attempt = 1; attempt <= WRITE_ATTEMPTS; attempt++) {
            try {
                atomicWrite(payload);
                return;
            } catch (IOException e) {
                last = e;
            }
        }
        throw last;
    }

    private void atomicWrite(Map<String, List<String>> payload) throws IOException {
        Path parent = file.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(tmp.toFile(), payload);
        try {
            Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            if (!warnedNonAtomic) {
                warnedNonAtomic = true;
                System.out.println("WARNING: atomic move not supported for discovery file " + file
                        + ", falling back to non-atomic replace: " + e.getMessage());
            }
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
