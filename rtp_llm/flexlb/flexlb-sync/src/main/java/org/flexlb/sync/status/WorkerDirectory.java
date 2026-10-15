package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Process-local owner of discovered worker generations and routable endpoint
 * generations.
 *
 * <p>A status entry describes the generation currently being polled. Routing
 * authority always comes from the matching endpoint generation in
 * {@link EndpointRegistry}; callers must capture a pin before using it. Keeping
 * both views behind this directory makes disagreement visible instead of
 * hiding it behind a derived capacity value.</p>
 */
@Component
public final class WorkerDirectory {
    private static final Logger diagnosticLogger = LoggerFactory.getLogger("syncLogger");

    private final Map<RoleType, ConcurrentHashMap<String, WorkerStatus>>
            statusesByRole = new EnumMap<>(RoleType.class);
    private final EndpointRegistry endpointRegistry;

    public WorkerDirectory(EndpointRegistry endpointRegistry) {
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        for (RoleType role : RoleType.values()) {
            statusesByRole.put(role, new ConcurrentHashMap<>());
        }
    }

    /** Immutable point-in-time view of discovered generations for one role. */
    public Map<String, WorkerStatus> statusSnapshot(RoleType role) {
        if (role == null) {
            return Map.of();
        }
        return Map.copyOf(statusesByRole.get(role));
    }

    /** Identity check used by asynchronous callbacks under the status lock. */
    public boolean isCurrentStatus(
            RoleType role, String address, WorkerStatus expected) {
        return expected != null && role != null && address != null
                && statusesByRole.get(role).get(address) == expected;
    }

    /**
     * Atomically install one newly discovered generation. Discovery publishes
     * status identity only; its endpoint remains absent until the first valid
     * WorkerStatus response commits.
     */
    public WorkerStatus currentOrDiscover(
            RoleType role,
            String address,
            Supplier<WorkerStatus> discoveredFactory) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(address, "address");
        Objects.requireNonNull(discoveredFactory, "discoveredFactory");
        return statusesByRole.get(role).computeIfAbsent(address, ignored -> {
            WorkerStatus discovered = Objects.requireNonNull(
                    discoveredFactory.get(), "discovered status");
            if (discovered.getRole() != role
                    || !address.equals(discovered.getIpPort())) {
                throw new IllegalArgumentException(
                        "Discovered WorkerStatus identity does not match directory key");
            }
            return discovered;
        });
    }

    public int discoveredCount(RoleType role) {
        return role == null ? 0 : statusesByRole.get(role).size();
    }

    public int discoveredCount() {
        int total = 0;
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            total += statuses.size();
        }
        return total;
    }

    /** Capacity means currently published routing capacity, not discovery. */
    public int routingCapacity(RoleType role) {
        return role == null ? 0 : endpointRegistry.getEndpointCount(role);
    }

    /** Exact endpoint for a discovered generation, or {@code null}. */
    public WorkerEndpoint exactEndpoint(
            RoleType role, String address, WorkerStatus expectedStatus) {
        return endpointRegistry.get(role, address, expectedStatus);
    }

    /** Publish the first endpoint for one exact discovered generation. */
    public EndpointRegistry.EndpointPublication publishPreparedEndpoint(
            String address,
            WorkerStatus status,
            WorkerStatus.PreparedStatus prepared) {
        EndpointRegistry.EndpointPublication publication = endpointRegistry.publishPreparedEndpoint(
                address, status, prepared);
        if (publication.endpoint() != null) {
            status.trace().record(Stage.ENDPOINT_PUBLISHED);
        }
        return publication;
    }

    /**
     * Close and detach the exact endpoint gate before publishing RETIRING.
     * The caller must hold the matching {@link WorkerStatus#lock}.
     */
    public EndpointRegistry.DetachedGeneration beginRetirement(
            RoleType role, String address, WorkerStatus status,
            String reason, long timeoutUs) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(address, "address");
        status.requireGenerationLock();
        status.requireActiveGeneration();

        status.trace().recordUnavailable(reason, status.pollHealth().lastSuccessfulPollUs(),
                timeoutUs, status.pollHealth().consecutiveTransportFailures());

        WorkerEndpoint expected = endpointRegistry.get(
                role, address, status);
        EndpointRegistry.DetachedGeneration detached =
                endpointRegistry.detachAndBeginRetirement(
                        role, address, status);
        if ((detached == null) != (expected == null)
                || detached != null && !detached.ownsEndpoint(expected)) {
            throw new IllegalStateException(
                    "Exact endpoint detach invariant failed for "
                            + address + "#" + status.getGenerationId());
        }
        if (detached == null
                && !status.beginRetirementAfterEndpointGateClosed()) {
            throw new IllegalStateException(
                    "WorkerStatus generation changed while its lock was held: "
                            + address + "#" + status.getGenerationId());
        }
        status.trace().record(Stage.ENDPOINT_WITHDRAWN);
        // Format bounded evidence; the existing async appender handles file I/O.
        reportTrace(address, status, "PROGRESS", "UNKNOWN");
        return detached;
    }

    /**
     * Await endpoint cleanup, clear any detailed cache index, and remove the
     * exact RETIRING status identity. No replacement can publish in between.
     */
    public void completeRetirement(
            RoleType role,
            String address,
            WorkerStatus status,
            EndpointRegistry.DetachedGeneration detached,
            CacheAwareService cacheAwareService,
            Logger logger) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(address, "address");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        Objects.requireNonNull(logger, "logger");

        Throwable cleanupFailure = null;
        boolean finalized;
        status.trace().record(Stage.RETIREMENT_CLEANUP_BEGIN);
        try {
            if (detached != null) {
                status.trace().record(Stage.ENDPOINT_AWAIT_BEGIN);
                detached.retireAndAwait();
                status.trace().record(Stage.ENDPOINT_AWAIT_END);
            }
        } catch (Throwable retirementFailure) {
            cleanupFailure = retirementFailure;
            status.trace().record(Stage.ENDPOINT_AWAIT_FAILED, retirementFailure);
        } finally {
            finalized = finalizeRetirement(
                    role, address, status, cacheAwareService, logger);
        }
        String outcome = cleanupFailure == null && finalized ? "SUCCESS" : "ERROR";
        status.trace().record(Stage.RETIREMENT_CLEANUP_END);
        // Report once after cleanup, outside the worker lock.
        reportTrace(address, status, "RETIREMENT", outcome);
        if (cleanupFailure != null) {
            logger.error(
                    "Endpoint cleanup failed after retiring generation {} for {}",
                    status.getGenerationId(), address, cleanupFailure);
        }
    }

    private boolean finalizeRetirement(
            RoleType role,
            String address,
            WorkerStatus status,
            CacheAwareService cacheAwareService,
            Logger logger) {
        Map<String, WorkerStatus> statuses = statusesByRole.get(role);
        status.trace().record(Stage.FINALIZE_LOCK_ATTEMPT);
        status.lock.lock();
        try {
            status.trace().record(Stage.FINALIZE_LOCK_ACQUIRED);
            status.requireRetiringGeneration();
            if (statuses.get(address) != status) {
                status.trace().record(Stage.FINALIZE_FAILED, "GENERATION_CHANGED");
                logger.error(
                        "Status identity changed before retirement finalized for {}#{}",
                        address, status.getGenerationId());
                return false;
            }
            boolean cacheCleanupSucceeded = true;
            try {
                if (role.requiresCacheKeys()) {
                    status.trace().record(Stage.CACHE_CLEANUP_BEGIN);
                    cacheAwareService.removeEngineBlockCache(address, status.trace());
                    status.trace().record(Stage.CACHE_CLEANUP_END);
                }
            } catch (Throwable cacheCleanupFailure) {
                cacheCleanupSucceeded = false;
                status.trace().record(Stage.CACHE_CLEANUP_FAILED, cacheCleanupFailure);
                logger.error(
                        "Cache cleanup failed while retiring generation {} for {}",
                        status.getGenerationId(), address, cacheCleanupFailure);
            }
            if (!statuses.remove(address, status)) {
                status.trace().record(Stage.FINALIZE_FAILED, "STATUS_REMOVAL_FAILED");
                logger.error(
                        "Exact status removal failed while its generation lock was held for {}#{}",
                        address, status.getGenerationId());
                return false;
            }
            status.trace().record(Stage.STATUS_REMOVED);
            return cacheCleanupSucceeded;
        } catch (Throwable finalizationFailure) {
            status.trace().record(Stage.FINALIZE_FAILED, finalizationFailure);
            logger.error(
                    "Status retirement finalization failed for {}#{}",
                    address, status.getGenerationId(), finalizationFailure);
            return false;
        } finally {
            status.lock.unlock();
            status.trace().record(Stage.FINALIZE_LOCK_RELEASED);
        }
    }

    private void reportTrace(String address, WorkerStatus status, String operation, String result) {
        try {
            if (!diagnosticLogger.isWarnEnabled()) { return; }
            var snapshot = status.trace().snapshot();
            // syncLogger already uses a bounded, never-blocking Logback AsyncAppender.
            diagnosticLogger.warn("event=worker_trace endpoint={} generation={} operation={} result={} capturedMs={} {}",
                    address, status.getGenerationId(), operation, result, snapshot.capturedAtMs(), snapshot.compact());
        } catch (RuntimeException | LinkageError unavailable) {
            // Diagnostics cannot interrupt endpoint withdrawal or cleanup.
        }
    }

    /** Immutable, non-owning address snapshot for lazy selection. */
    public List<String> endpointAddressSnapshot(RoleType role) {
        return endpointRegistry.endpointAddressSnapshot(role);
    }

    /**
     * Immutable, non-owning Prefill generations for one routing traversal.
     * The winning address must still be captured and identity-checked.
     */
    public List<EndpointRegistry.PrefillRoutingEntry> prefillRoutingSnapshot(
            RoleType role) {
        return endpointRegistry.prefillRoutingSnapshot(role);
    }

    /** Capture one exact currently published endpoint generation by address. */
    public WorkerEndpoint.GenerationPin captureEndpoint(
            RoleType role, String ipPort) {
        return endpointRegistry.capture(role, ipPort);
    }

    /** Immutable Decode routing values for one group. */
    public List<DecodeEndpoint.DecodeRoutingView> decodeRoutingSnapshot(
            String group) {
        // The selector copies this lazy view into its primitive candidate
        // buffer in one pass. Avoid an intermediate immutable list when no
        // group filter is needed.
        List<DecodeEndpoint.DecodeRoutingView> snapshots =
                endpointRegistry.decodeRoutingSnapshot();
        if (group == null || snapshots.isEmpty()) {
            return snapshots;
        }
        ArrayList<DecodeEndpoint.DecodeRoutingView> matching =
                new ArrayList<>(snapshots.size());
        for (int index = 0; index < snapshots.size(); index++) {
            DecodeEndpoint.DecodeRoutingView snapshot = snapshots.get(index);
            if (group.equals(snapshot.topology().group())) {
                matching.add(snapshot);
            }
        }
        return List.copyOf(matching);
    }

    /** Pin the exact generation represented by a Decode routing snapshot. */
    public WorkerEndpoint.GenerationPin captureDecodeGeneration(
            DecodeEndpoint.DecodeRoutingView expected) {
        return endpointRegistry.captureDecodeGeneration(expected);
    }

    /** Capture exact endpoint generations for one routing decision. */
    public List<WorkerEndpoint.GenerationPin> captureEndpoints(
            RoleType role, String group) {
        List<WorkerEndpoint.GenerationPin> captured = endpointRegistry.capture(role);
        List<WorkerEndpoint.GenerationPin> matching =
                new ArrayList<>(captured.size());
        try {
            for (int index = 0; index < captured.size(); index++) {
                WorkerEndpoint.GenerationPin pin = captured.get(index);
                WorkerStatus.TopologySnapshot topology =
                        pin.endpoint().getStatus().topologySnapshot();
                if (group != null && !group.equals(topology.group())) {
                    pin.close();
                    captured.set(index, null);
                    continue;
                }
                matching.add(pin);
                captured.set(index, null);
            }
            return matching;
        } catch (Throwable failure) {
            closePins(captured);
            closePins(matching);
            throw failure;
        }
    }

    /** Non-owning monitoring projection; no live endpoint map escapes. */
    public List<String> endpointAddresses(RoleType role, String group) {
        List<WorkerEndpoint.GenerationPin> captured = captureEndpoints(role, group);
        try {
            List<String> addresses = new ArrayList<>(captured.size());
            for (WorkerEndpoint.GenerationPin pin : captured) {
                addresses.add(pin.endpoint().ipPort());
            }
            return addresses;
        } finally {
            closePins(captured);
        }
    }

    private static void closePins(List<WorkerEndpoint.GenerationPin> pins) {
        for (WorkerEndpoint.GenerationPin pin : pins) {
            if (pin != null) {
                pin.close();
            }
        }
    }
}
