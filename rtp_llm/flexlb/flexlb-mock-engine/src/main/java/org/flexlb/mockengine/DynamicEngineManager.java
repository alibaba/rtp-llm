package org.flexlb.mockengine;

import io.grpc.Server;
import io.netty.channel.EventLoopGroup;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
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
 *   <li>{@code removeEngine} PERMANENTLY detaches the engine, in two modes
 *       (user ruling 2026-09: a PLANNED scale-in under load must not lose or
 *       fail any request):
 *       <ul>
 *       <li><b>graceful</b> (default) — the production rolling scale-in
 *           order. Phase 1 strips the engine's discovery-file entry FIRST so
 *           the master stops routing NEW requests to it (the engine keeps
 *           serving everything already accepted, including stragglers that
 *           race past the strip window). Phase 2 waits a BOUNDED interval
 *           (drainTimeoutMs) for all in-flight work to finish (running
 *           tasks, both pending queues, cross-engine P→D ownership). Phase 3
 *           tears the gRPC server down and removes every bookkeeping entry.
 *           If the deadline expires with work still in flight, the removal
 *           falls back to the abrupt teardown and reports
 *           {@code drained=false} rather than blocking forever.</li>
 *       <li><b>abrupt</b> (legacy, for chaos-style fault cases) — stopped
 *           flag + cancel-sweep bookkeeping + {@code shutdownNow} cutting
 *           in-flight RPC streams immediately; master-side 3-consecutive-
 *           failure / TTL logic handles the fallout.</li>
 *       </ul>
 *       Both modes end with removal from the services map AND the discovery
 *       file — remove never leaves residue (unlike stop/start, which keeps
 *       the discovery entry for an expected same-port comeback).</li>
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

    /** Outcome of a successful remove — counters are sampled when the drain
     * STARTS (the "at removal time" report), plus the drain result. */
    record RemovedEngine(String engineName, int grpcPort, int runningAtRemoval,
                         int waitingAtRemoval, String mode, boolean drained, long drainMs) {
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
     * Ports mid-graceful-drain: excluded from EVERY discovery-file rewrite
     * (including rewrites triggered by concurrent add_engine) until the
     * teardown completes, so a draining engine can never be resurrected in
     * the master's routing view. Guarded by {@link #mutationLock}.
     */
    private final Set<Integer> pendingRemovalPorts = new HashSet<>();
    /**
     * Per-engine drain futures for idempotent concurrent removals (grpc port
     * → outcome). Every caller that hits an in-progress drain awaits the SAME
     * future, so concurrent removes of one engine all observe a consistent
     * post-teardown state. Guarded by {@link #mutationLock}.
     */
    private final Map<Integer, CompletableFuture<RemovedEngine>> drainingFutures = new HashMap<>();
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

    /** Permanently remove an engine (resolved by gRPC port or engine name).
     *
     * <p>See the class javadoc for the two modes. The graceful call BLOCKS
     * until the drain settles (bounded by drainTimeoutMs plus the teardown
     * margin) — the HTTP control plane's cached thread pool absorbs the wait.
     *
     * @param mode          "graceful" (default) or "abrupt"
     * @param drainTimeoutMs graceful drain bound; on expiry the removal
     *                      falls back to the abrupt teardown (drained=false)
     */
    RemovedEngine removeEngine(JavaMockEngineCluster.FastRpcService service, String mode, long drainTimeoutMs)
            throws EngineOperationException, IOException {
        int port = service.getGrpcPort();
        CompletableFuture<RemovedEngine> drain;
        // "At removal time" counters are sampled at the DECISION instant (lock
        // held, before any waiting): for graceful this is the drain START —
        // sampling later (at the teardown) would report zeros because the
        // in-flight set the drain waited out has already emptied.
        int runningAtRemoval;
        int waitingAtRemoval;
        synchronized (mutationLock) {
            CompletableFuture<RemovedEngine> inProgress = drainingFutures.get(port);
            if (inProgress != null) {
                // Idempotent re-entry: another caller is already draining this
                // engine — await the SAME outcome instead of racing it.
                drain = inProgress;
            } else {
                runningAtRemoval = service.getRunningCount();
                waitingAtRemoval = removalWaitingCount(service);
                if ("abrupt".equalsIgnoreCase(mode)) {
                    return detachLocked(service, "abrupt", false, 0L,
                            runningAtRemoval, waitingAtRemoval);
                }
                // Phase 1 (lock held): strip the discovery entry FIRST so the
                // master stops routing new requests; the engine keeps serving
                // everything it already accepted (production rolling scale-in
                // order — the in-flight set, not the admission gate, is what
                // the drain below waits on).
                pendingRemovalPorts.add(port);
                try {
                    rewriteDiscoveryFileLocked();
                } catch (IOException e) {
                    pendingRemovalPorts.remove(port);
                    throw new IOException("engine " + service.getEngineName()
                            + " removal aborted (engine NOT removed): discovery file rewrite failed: "
                            + e.getMessage(), e);
                }
                drain = new CompletableFuture<>();
                drainingFutures.put(port, drain);
                final int sampledRunning = runningAtRemoval;
                final int sampledWaiting = waitingAtRemoval;
                Thread drainer = new Thread(
                        () -> runGracefulDrain(service, drain, drainTimeoutMs,
                                sampledRunning, sampledWaiting),
                        "engine-drain-" + service.getEngineName());
                drainer.setDaemon(true);
                drainer.start();
            }
        }
        // Phase 2 (NO lock held): a drain must never block add/remove of OTHER
        // engines — await this engine's drainer from outside the lock.
        try {
            return drain.get(drainTimeoutMs + 30_000, TimeUnit.MILLISECONDS);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            throw new IOException("engine " + service.getEngineName()
                    + " graceful removal failed: " + cause, cause);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new EngineOperationException(503,
                    "graceful removal of engine " + service.getEngineName() + " interrupted");
        } catch (TimeoutException e) {
            throw new EngineOperationException(504,
                    "graceful removal of engine " + service.getEngineName() + " did not settle within "
                            + (drainTimeoutMs + 30_000) + "ms");
        }
    }

    /**
     * Drainer body (dedicated daemon thread): bounded wait for
     * {@code hasInflightWork()} to clear, then the teardown critical section,
     * then publish the outcome to every awaiting caller.  The "at removal"
     * counters are the drain-START samples taken by {@link #removeEngine}.
     */
    private void runGracefulDrain(JavaMockEngineCluster.FastRpcService service,
                                  CompletableFuture<RemovedEngine> outcome,
                                  long drainTimeoutMs,
                                  int runningAtRemoval,
                                  int waitingAtRemoval) {
        long start = System.nanoTime();
        long deadline = start + TimeUnit.MILLISECONDS.toNanos(drainTimeoutMs);
        boolean drained = false;
        while (System.nanoTime() < deadline) {
            if (!service.hasInflightWork()) {
                drained = true;
                break;
            }
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        long drainMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
        try {
            RemovedEngine result;
            synchronized (mutationLock) {
                result = detachLocked(service, "graceful", drained, drainMs,
                        runningAtRemoval, waitingAtRemoval);
            }
            outcome.complete(result);
        } catch (Throwable failure) {
            outcome.completeExceptionally(failure);
        } finally {
            synchronized (mutationLock) {
                drainingFutures.remove(service.getGrpcPort(), outcome);
            }
        }
    }

    /**
     * Teardown critical section (mutationLock held, single place shared by
     * both modes): cancel-sweep bookkeeping, cut the gRPC server, drop the
     * services-map entry, rewrite the discovery file, and report.
     *
     * <p>On a completed graceful drain every structure the sweep touches is
     * already empty, so the cut is bloodless; the abrupt path (and the
     * graceful timeout fallback) relies on the sweep to net the counters out.
     */
    private RemovedEngine detachLocked(JavaMockEngineCluster.FastRpcService service,
                                       String mode,
                                       boolean drained,
                                       long drainMs,
                                       int runningAtRemoval,
                                       int waitingAtRemoval) throws IOException {
        int port = service.getGrpcPort();
        pendingRemovalPorts.remove(port);
        drainingFutures.remove(port);
        // Same rejection semantics as /stop_engine first, then drain bookkeeping
        // so in-flight counters net out (no leak report for a removed engine)…
        service.setStopped(true);
        service.drainAndShutdown();
        // …then cut the RPC streams.
        Server server = serversByPort.remove(port);
        if (server != null) {
            if ("graceful".equals(mode) && drained) {
                // Bloodless teardown: every handler has finished, so give the
                // transport a bounded beat to flush the LAST completion frames
                // of the drained requests before any cut — a shutdownNow racing
                // the flush would turn a finished request into a transport
                // error on the client side (exactly what graceful mode exists
                // to prevent). Falls back to the hard cut if a straggler stream
                // (e.g. a master FetchResponse park) refuses to end.
                server.shutdown();
                try {
                    server.awaitTermination(2, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                if (!server.isTerminated()) {
                    server.shutdownNow();
                }
            } else {
                server.shutdownNow();
            }
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
        return new RemovedEngine(service.getEngineName(), port,
                runningAtRemoval, waitingAtRemoval, mode, drained, drainMs);
    }

    // ────────────────── Internals (mutationLock held) ──────────────────

    /** "At removal time" waiting-depth sample, by role (decode parks in its
     * pending queue; prefill parks in the master-composed waiting set). */
    private static int removalWaitingCount(JavaMockEngineCluster.FastRpcService service) {
        return "DECODE".equals(service.getRoleName())
                ? service.getDecodePendingQueueDepth()
                : service.getWaitingCount();
    }

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
            if (pendingRemovalPorts.isEmpty()) {
                discoveryFileStore.rewrite(services);
            } else {
                // A mid-drain engine is still hosted (it is draining, not gone)
                // but must stay invisible to the master's routing view until
                // the teardown completes — rewrite an EXCLUDED view so a
                // concurrent add_engine can never resurrect its entry.
                Map<Integer, JavaMockEngineCluster.FastRpcService> view = new HashMap<>(services);
                for (int drainingPort : pendingRemovalPorts) {
                    view.remove(drainingPort);
                }
                discoveryFileStore.rewrite(view);
            }
        }
    }
}
