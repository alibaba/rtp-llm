package org.flexlb.mockengine;

import io.grpc.Server;
import io.netty.channel.EventLoopGroup;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Runtime scale-out/scale-in for the Java mock engine cluster, backing the
 * HTTP control endpoints {@code /add_engine} and {@code /remove_engine}.
 *
 * <p>Semantics vs the pre-existing {@code /stop_engine} + {@code /start_engine}
 * pair (which address a FIXED engine set):
 * <ul>
 *   <li>{@code addEngine} creates a brand-new {@code FastRpcService} on a fresh
 *       gRPC port (explicit or auto = current max + 1), registers it in the
 *       services map, starts the gRPC server, and appends it to the discovery
 *       file so a master running {@code FileServiceDiscovery} picks it up
 *       within one sync interval.</li>
 *   <li>{@code removeEngine} PERMANENTLY detaches the engine: graceful-stop
 *       semantics identical to {@code /stop_engine} (stopped flag + server
 *       shutdownNow, cutting in-flight RPC streams — master-side 3-consecutive-
 *       failure / TTL logic handles the fallout), plus removal from the
 *       services map AND the discovery file. stop/start keeps the discovery
 *       entry (engine expected to come back on the same port); remove never
 *       leaves residue.</li>
 * </ul>
 *
 * <p>All mutations run inside a single cluster-wide lock covering
 * "read services → allocate port/name → start server → rewrite discovery
 * file", so concurrent add/remove calls (and the file rewrite they trigger)
 * can never interleave and leave the file inconsistent with the services map.
 */
final class DynamicEngineManager {

    /** Outcome of a successful add — the fields the HTTP response reports. */
    record AddedEngine(String engineName, int grpcPort) {
    }

    /** Outcome of a successful remove — includes the counters at removal time. */
    record RemovedEngine(String engineName, int grpcPort, int runningAtRemoval, int waitingAtRemoval) {
    }

    /** Thrown for caller-input problems (bad role, port conflict, unknown engine). */
    static final class EngineOperationException extends Exception {
        final int status;

        EngineOperationException(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    private final JavaMockEngineCluster.Config config;
    private final MockPerformanceModel performance;
    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private final Map<Integer, Server> serversByPort;
    private final EventLoopGroup bossGroup;
    private final EventLoopGroup workerGroup;
    private final ScheduledExecutorService scheduler;
    private final JavaMockEngineCluster.ClusterStats stats;
    /** Null when the cluster runs without --discovery-file (add/remove still work, file not maintained). */
    private final DiscoveryFileStore discoveryFileStore;
    /** Cluster-shared engine_events.jsonl writer (null = event stream disabled); wired onto every dynamically added engine. */
    private final JavaMockEngineCluster.EngineEventLog engineEventLog;
    /** Cluster-single mutation lock: serializes add/remove including the file rewrite. */
    private final Object mutationLock = new Object();
    /** Monotonic id for dynamically added engine names, appended to the role index. */
    private final AtomicInteger dynamicId = new AtomicInteger();
    /**
     * Monotonic GLOBAL engine index for unique advertisement IPs
     * ({@code --unique-engine-ips}): initialized to the initial engine count
     * (prefill + decode) and never decremented — a removed engine's index is
     * never reused, so dynamically added engines always get a fresh unique
     * 127.x.y.z advertisement IP (see
     * {@link JavaMockEngineCluster#declaredHost}).
     */
    private final AtomicInteger nextEngineIndex;

    DynamicEngineManager(JavaMockEngineCluster.Config config,
                         MockPerformanceModel performance,
                         Map<Integer, JavaMockEngineCluster.FastRpcService> services,
                         Map<Integer, Server> serversByPort,
                         EventLoopGroup bossGroup,
                         EventLoopGroup workerGroup,
                         ScheduledExecutorService scheduler,
                         JavaMockEngineCluster.ClusterStats stats,
                         DiscoveryFileStore discoveryFileStore,
                         JavaMockEngineCluster.EngineEventLog engineEventLog) {
        this.config = config;
        this.performance = performance;
        this.services = services;
        this.serversByPort = serversByPort;
        this.bossGroup = bossGroup;
        this.workerGroup = workerGroup;
        this.scheduler = scheduler;
        this.stats = stats;
        this.discoveryFileStore = discoveryFileStore;
        this.engineEventLog = engineEventLog;
        // The manager is constructed right after the initial roles started, so
        // services.size() == nPrefill + nDecode — the first dynamic engine takes
        // the next global index.
        this.nextEngineIndex = new AtomicInteger(services.size());
    }

    boolean isFileDiscoveryEnabled() {
        return discoveryFileStore != null;
    }

    /**
     * Create and start a new engine of the given role.
     *
     * @param role         "prefill" or "decode" (case-insensitive)
     * @param explicitPort requested gRPC port, or null to auto-allocate max+1
     */
    AddedEngine addEngine(String role, Integer explicitPort) throws EngineOperationException, IOException {
        String roleName = normalizeRole(role);
        synchronized (mutationLock) {
            int grpcPort = resolvePort(explicitPort);
            String engineName = nextEngineName(roleName);
            JavaMockEngineCluster.FastRpcService service = JavaMockEngineCluster.startEngine(
                    config, performance, serversByPort, bossGroup, workerGroup,
                    services, scheduler, stats, roleName, engineName, grpcPort,
                    nextEngineIndex.getAndIncrement());
            service.setEngineEventLog(engineEventLog);
            try {
                rewriteDiscoveryFileLocked();
            } catch (IOException e) {
                // Engine is up but the file update failed: roll the engine back so
                // the discovery view stays consistent with the hosted engine set.
                rollbackEngine(grpcPort, service);
                throw new IOException("engine " + engineName + " started on port " + grpcPort
                        + " but discovery file rewrite failed, engine rolled back: " + e.getMessage(), e);
            }
            return new AddedEngine(engineName, grpcPort);
        }
    }

    /** Permanently remove an engine (resolved by gRPC port or engine name). */
    RemovedEngine removeEngine(JavaMockEngineCluster.FastRpcService service)
            throws EngineOperationException, IOException {
        synchronized (mutationLock) {
            int port = service.getGrpcPort();
            int runningAtRemoval = service.getRunningCount();
            int waitingAtRemoval = "DECODE".equals(service.getRoleName())
                    ? service.getDecodePendingQueueDepth()
                    : service.getWaitingCount();
            // Same rejection semantics as /stop_engine first, then drain bookkeeping
            // so in-flight counters net out (no leak report for a removed engine)…
            service.setStopped(true);
            service.drainAndShutdown();
            // …then cut the RPC streams for in-flight requests.
            Server server = serversByPort.remove(port);
            if (server != null) {
                server.shutdownNow();
            }
            service.shutdown();
            services.remove(port, service);
            try {
                rewriteDiscoveryFileLocked();
            } catch (IOException e) {
                // Engine is already gone; report the failure so the operator can
                // retry (a stale entry would keep the master retrying a dead port).
                throw new IOException("engine " + service.getEngineName() + " removed from port " + port
                        + " but discovery file rewrite failed: " + e.getMessage(), e);
            }
            return new RemovedEngine(service.getEngineName(), port, runningAtRemoval, waitingAtRemoval);
        }
    }

    // ────────────────── Internals (mutationLock held) ──────────────────

    private String normalizeRole(String role) throws EngineOperationException {
        if (role == null) {
            throw new EngineOperationException(400, "request must contain 'role' (prefill|decode)");
        }
        return switch (role.toLowerCase()) {
            case "prefill" -> "prefill";
            case "decode" -> "decode";
            default -> throw new EngineOperationException(400,
                    "unknown role '" + role + "', expected prefill|decode");
        };
    }

    private int resolvePort(Integer explicitPort) throws EngineOperationException {
        int grpcPort;
        if (explicitPort != null) {
            if (explicitPort <= 0 || explicitPort > 65_535) {
                throw new EngineOperationException(400, "port must be in [1, 65535]: " + explicitPort);
            }
            grpcPort = explicitPort;
        } else {
            grpcPort = services.keySet().stream().mapToInt(Integer::intValue).max().orElse(1024) + 1;
        }
        if (services.containsKey(grpcPort)) {
            throw new EngineOperationException(409,
                    "port " + grpcPort + " already in use by engine " + services.get(grpcPort).getEngineName());
        }
        return grpcPort;
    }

    /** Role-indexed name that is unique among currently hosted engines (e.g. "prefill-3"). */
    private String nextEngineName(String roleName) {
        int maxIndex = -1;
        String prefix = roleName + "-";
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            String name = service.getEngineName();
            if (name.startsWith(prefix)) {
                try {
                    maxIndex = Math.max(maxIndex, Integer.parseInt(name.substring(prefix.length())));
                } catch (NumberFormatException ignored) {
                    // Non-indexed name (e.g. "prefill-<port>" from the test constructor) — skip.
                }
            }
        }
        int index = maxIndex + 1;
        // Defensive: with concurrent re-adds after removes the index can collide
        // with a leftover name only if parsing failed above; keep it unique anyway.
        String candidate = prefix + index;
        while (engineNameTaken(candidate)) {
            candidate = prefix + (index + dynamicId.incrementAndGet() + 1_000_000);
        }
        return candidate;
    }

    private boolean engineNameTaken(String name) {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            if (name.equals(service.getEngineName())) {
                return true;
            }
        }
        return false;
    }

    private void rollbackEngine(int grpcPort, JavaMockEngineCluster.FastRpcService service) {
        Server server = serversByPort.remove(grpcPort);
        if (server != null) {
            server.shutdownNow();
        }
        service.setStopped(true);
        service.drainAndShutdown();
        service.shutdown();
        services.remove(grpcPort, service);
    }

    private void rewriteDiscoveryFileLocked() throws IOException {
        if (discoveryFileStore != null) {
            discoveryFileStore.rewrite(services);
        }
    }
}
