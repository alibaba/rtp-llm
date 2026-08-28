package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.function.LongPredicate;

@Component
public class EndpointRegistry {

    private enum RegistryPhase {
        OPEN,
        CLOSING,
        CLOSED
    }

    /** Result of publishing a fully initialized endpoint generation. */
    public record EndpointPublication(
            WorkerEndpoint endpoint,
            Runnable statusProjection) {
    }

    /**
     * One exact endpoint generation moved out of the routing registry.
     *
     * <p>The capability is the sole completion token for the registry's
     * detached-retirement barrier. It initiates and awaits endpoint cleanup
     * outside the registry gate, then resolves that token exactly once.</p>
     */
    public static final class DetachedGeneration {
        private final EndpointRegistry registry;
        private final WorkerEndpoint endpoint;
        private final AtomicBoolean retirementClaimed = new AtomicBoolean();

        private DetachedGeneration(
                EndpointRegistry registry,
                WorkerEndpoint endpoint) {
            this.registry = registry;
            this.endpoint = endpoint;
        }

        /** Exact identity check without exposing the owned endpoint resource. */
        public boolean ownsEndpoint(WorkerEndpoint exactEndpoint) {
            return endpoint == exactEndpoint;
        }

        /** Drain this exact generation and resolve its registry barrier token. */
        public void retireAndAwait() {
            if (!retirementClaimed.compareAndSet(false, true)) {
                throw new IllegalStateException(
                        "Detached endpoint retirement was already claimed: "
                                + endpoint.getStatus().getIpPort() + "#"
                                + endpoint.getStatus().getGenerationId());
            }
            Throwable failure = null;
            try {
                try {
                    endpoint.close();
                } catch (Throwable closeFailure) {
                    failure = closeFailure;
                }
                try {
                    endpoint.awaitRetirement();
                } catch (Throwable awaitFailure) {
                    failure = appendFailure(failure, awaitFailure);
                }
            } finally {
                try {
                    registry.resolveDetachedGeneration(this);
                } catch (Throwable resolutionFailure) {
                    failure = appendFailure(failure, resolutionFailure);
                }
            }
            rethrowDetachedRetirementFailure(failure);
        }
    }

    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>>
            endpointsByRole = endpointMaps();
    /**
     * Immutable, generation-neutral Prefill address directory.
     *
     * <p>Structural Prefill writers prebuild the next directory while holding
     * {@link #lifecycleGate}, publish the exact map mutation, and only then
     * release the immutable address view with one volatile write. Readers must
     * still exact-capture an endpoint generation by address.</p>
     */
    private volatile List<String> prefillAddressDirectory = List.of();
    /**
     * Advisory Decode routing directory. Writers publish a complete immutable
     * replacement after changing the endpoint map. A racing reader may finish
     * an older traversal, but it cannot acquire stale ownership because the
     * selected address and generation are always revalidated by
     * {@link #captureDecodeGeneration(DecodeEndpoint.DecodeRoutingView)}.
     */
    private volatile List<Map.Entry<String, DecodeEndpoint>> decodeDirectory =
            List.of();
    private final ConfigService configService;
    private final EndpointEventProjector endpointEvents;
    private final BatchSchedulerReporter reporter;
    private final DeliveryStrategy deliveryStrategy;
    private final PlacementAvailability placementAvailability;
    private final Object lifecycleGate = new Object();
    private RegistryPhase registryPhase = RegistryPhase.OPEN;
    private int inflightPublications;
    private int inflightDetachedRetirements;
    private Throwable closeFailure;

    private static EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>>
            endpointMaps() {
        EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>> maps =
                new EnumMap<>(RoleType.class);
        for (RoleType role : List.of(
                RoleType.PREFILL,
                RoleType.DECODE,
                RoleType.PDFUSION,
                RoleType.VIT)) {
            maps.put(role, new ConcurrentHashMap<>());
        }
        return maps;
    }

    private ConcurrentHashMap<String, WorkerEndpoint> endpoints(RoleType role) {
        return endpointsByRole.get(role);
    }

    @Autowired
    public EndpointRegistry(ConfigService configService,
                            EndpointEventProjector endpointEvents,
                            BatchSchedulerReporter reporter,
                            DeliveryStrategy deliveryStrategy,
                            PlacementAvailability placementAvailability) {
        this.configService = java.util.Objects.requireNonNull(
                configService, "configService");
        this.endpointEvents = java.util.Objects.requireNonNull(
                endpointEvents, "endpointEvents");
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.deliveryStrategy = java.util.Objects.requireNonNull(
                deliveryStrategy, "deliveryStrategy");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        Map<String, WorkerEndpoint> endpoints = endpoints(roleType);
        return endpoints == null ? null : endpoints.get(ipPort);
    }

    /**
     * Capture one exact currently published endpoint generation.
     *
     * <p>The pin is acquired inside the address key's CHM remapping critical
     * section. Capture therefore linearizes either before exact detach (and
     * retirement waits for the pin) or after detach (and returns {@code null}).
     * The returned route capability is thread-confined.</p>
     */
    public WorkerEndpoint.GenerationPin capture(
            RoleType roleType,
            String ipPort) {
        ConcurrentHashMap<String, WorkerEndpoint> endpoints =
                endpoints(roleType);
        return endpoints == null ? null : capture(endpoints, ipPort);
    }

    /**
     * Capture the currently addressable generations without exposing a live
     * registry map. Addresses are snapshotted first; every exact pin is then
     * acquired through {@link #capture(RoleType, String)}.
     */
    public List<WorkerEndpoint.GenerationPin> capture(RoleType roleType) {
        List<String> addresses = endpointAddressSnapshot(roleType);
        List<WorkerEndpoint.GenerationPin> pins =
                new ArrayList<>(addresses.size());
        try {
            for (String address : addresses) {
                WorkerEndpoint.GenerationPin pin = capture(roleType, address);
                if (pin != null) {
                    pins.add(pin);
                }
            }
            return pins;
        } catch (RuntimeException | Error captureFailure) {
            for (WorkerEndpoint.GenerationPin pin : pins) {
                try {
                    pin.close();
                } catch (Throwable closeFailure) {
                    captureFailure.addSuppressed(closeFailure);
                }
            }
            throw captureFailure;
        }
    }

    /**
     * Return an immutable point-in-time address snapshot for one role.
     *
     * <p>No endpoint object or live registry map escapes.  A caller which needs
     * generation ownership must still pass an address back through
     * {@link #capture(RoleType, String)}; same-address replacement therefore
     * remains linearized by the address key's remapping critical section.</p>
     */
    public List<String> endpointAddressSnapshot(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillAddressDirectory;
        }
        Map<String, WorkerEndpoint> endpoints = endpoints(roleType);
        return endpoints == null
                ? List.of() : List.copyOf(endpoints.keySet());
    }

    /**
     * Return immutable routing values for the Decode generations observed
     * during this traversal.
     *
     * <p>The registry is intentionally not locked while an endpoint takes its
     * admission lock.  This avoids introducing a map-bin/admission-lock order;
     * publication races are resolved when the selected generation is pinned
     * by {@link #captureDecodeGeneration(DecodeEndpoint.DecodeRoutingView)}.</p>
     */
    public List<DecodeEndpoint.DecodeRoutingView> decodeRoutingSnapshot() {
        List<Map.Entry<String, DecodeEndpoint>> directory = decodeDirectory;
        return new AbstractList<>() {
            @Override
            public DecodeEndpoint.DecodeRoutingView get(int index) {
                Map.Entry<String, DecodeEndpoint> entry = directory.get(index);
                return entry.getValue().routingViewSnapshot(entry.getKey());
            }

            @Override
            public int size() {
                return directory.size();
            }
        };
    }

    /**
     * Capture the exact Decode generation represented by a routing snapshot.
     *
     * <p>The routing values are advisory and may change after selection. The
     * subsequent reservation/dispatch transaction revalidates capacity under
     * the endpoint admission lock. Requiring an identical admission version
     * Every rejected or exceptional capture is closed here; a successful
     * caller owns the returned generation pin.</p>
     */
    public WorkerEndpoint.GenerationPin captureDecodeGeneration(
            DecodeEndpoint.DecodeRoutingView expected) {
        WorkerEndpoint.GenerationPin pin =
                capture(RoleType.DECODE, expected.address());
        if (pin == null) {
            return null;
        }
        boolean transferred = false;
        try {
            if (pin.generationId() != expected.generationId()
                    || !(pin.endpoint() instanceof DecodeEndpoint)) {
                return null;
            }
            transferred = true;
            return pin;
        } finally {
            if (!transferred) {
                pin.close();
            }
        }
    }

    private static WorkerEndpoint.GenerationPin capture(
            ConcurrentHashMap<String, WorkerEndpoint> endpoints,
            String ipPort) {
        AtomicReference<WorkerEndpoint.GenerationPin> captured =
                new AtomicReference<>();
        endpoints.computeIfPresent(ipPort, (ignored, current) -> {
            captured.set(current.tryPinGeneration());
            return current;
        });
        return captured.get();
    }

    /** Return the endpoint only when it belongs to the expected status generation. */
    public WorkerEndpoint get(
            RoleType roleType,
            String ipPort,
            WorkerStatus expectedStatus) {
        WorkerEndpoint endpoint = get(roleType, ipPort);
        return endpoint != null && endpoint.getStatus() == expectedStatus
                ? endpoint : null;
    }

    /**
     * Reduce a private candidate from a new status delta, commit the validated
     * Engine observation, then make the candidate routable. A private candidate
     * cannot own any published RequestSlot identity, so its typed scheduler
     * reduction is necessarily empty.
     *
     * <p>A factory, endpoint-reducer, or status-commit failure closes the
     * candidate before routing publication. If final map publication fails
     * after commit, the caller withdraws the entire WorkerStatus generation;
     * it is never restored.</p>
     */
    public EndpointPublication publishPreparedEndpoint(
            String address,
            WorkerStatus status,
            WorkerStatus.PreparedStatus prepared) {
        beginCandidatePublication();
        try {
            requireGenerationLock(status);
            status.requireActiveGeneration();
            WorkerStatus.StatusObservation observation =
                    prepared.observation();
            if (observation.owner() != status) {
                throw new IllegalArgumentException(
                        "staged status belongs to another worker generation");
            }
            RoleType role = observation.role();
            if (role != status.getRole()) {
                throw new IllegalArgumentException(
                        "staged status role does not match its worker generation");
            }
            java.util.Objects.requireNonNull(address, "address");
            if (status.appliedStatusCursor().statusVersion() >= 0L) {
                throw new IllegalStateException(
                        "A committed WorkerStatus generation cannot publish a second endpoint: "
                                + address + "#" + status.getGenerationId());
            }
            WorkerEndpoint candidate = null;
            EndpointPublication publication;
            try {
                candidate = createEndpoint(
                        status, role, observation.engine());
                Runnable projection = candidate.initializeFromPreparedStatus(
                        status, observation);
                publication = new EndpointPublication(
                        candidate, projection);
                startCandidate(candidate);
                status.publishPreparedStatus(prepared);
            } catch (Throwable reductionOrCommitFailure) {
                closeCandidate(candidate, reductionOrCommitFailure);
                throw propagate(reductionOrCommitFailure, address);
            }
            return publishPrivateEndpoint(
                    role,
                    endpoints(role),
                    address,
                    candidate,
                    publication);
        } finally {
            endCandidatePublication();
        }
    }

    private void beginCandidatePublication() {
        synchronized (lifecycleGate) {
            if (registryPhase != RegistryPhase.OPEN) {
                throw new IllegalStateException(
                        "EndpointRegistry is closing");
            }
            inflightPublications++;
        }
    }

    private void endCandidatePublication() {
        synchronized (lifecycleGate) {
            if (inflightPublications <= 0) {
                throw new IllegalStateException(
                        "EndpointRegistry publication count underflow");
            }
            inflightPublications--;
            if (inflightPublications == 0) {
                lifecycleGate.notifyAll();
            }
        }
    }

    private EndpointPublication publishPrivateEndpoint(
            RoleType role,
            ConcurrentHashMap<String, WorkerEndpoint> endpoints,
            String ipPort,
            WorkerEndpoint candidate,
            EndpointPublication publication) {
        try {
            mutateEndpointMap(role, endpoints, ipPort, (ignored, current) -> {
                if (current != null
                        && current.getStatus() == candidate.getStatus()) {
                    throw new IllegalStateException(
                            "Endpoint generation is already published for "
                                    + ipPort);
                }
                if (current != null) {
                    throw new IllegalStateException(
                            "Existing endpoint generation must be withdrawn before publication for "
                                    + ipPort);
                }
                return candidate;
            });
            signalPublishedEndpoint(candidate);
            return publication;
        } catch (Throwable publicationFailure) {
            withdrawAndCloseCandidate(
                    role, endpoints, ipPort, candidate, publicationFailure);
            throw propagate(publicationFailure, ipPort);
        }
    }

    private void withdrawAndCloseCandidate(
            RoleType role,
            ConcurrentHashMap<String, WorkerEndpoint> endpoints,
            String ipPort,
            WorkerEndpoint candidate,
            Throwable primaryFailure) {
        try {
            mutateEndpointMap(
                    role,
                    endpoints,
                    ipPort,
                    (ignored, current) -> current == candidate
                            ? null : current);
        } catch (Throwable withdrawalFailure) {
            addSuppressedNoFail(primaryFailure, withdrawalFailure);
        }
        closeCandidate(candidate, primaryFailure);
    }

    /**
     * Execute one exact-address map mutation. Decode and Prefill mutations also
     * publish their immutable routing directories under the lifecycle gate.
     */
    private WorkerEndpoint mutateEndpointMap(
            RoleType role,
            ConcurrentHashMap<String, WorkerEndpoint> endpoints,
            String address,
            BiFunction<String, WorkerEndpoint, WorkerEndpoint> mutation) {
        boolean updatesPrefillDirectory = role == RoleType.PREFILL;
        boolean updatesDecodeDirectory = role == RoleType.DECODE;
        if (!updatesPrefillDirectory && !updatesDecodeDirectory) {
            return endpoints.compute(address, mutation);
        }
        synchronized (lifecycleGate) {
            WorkerEndpoint exactCurrent = endpoints.get(address);
            WorkerEndpoint next = mutation.apply(address, exactCurrent);
            if (next == exactCurrent) {
                return exactCurrent;
            }
            List<String> nextPrefillDirectory = updatesPrefillDirectory
                    ? registryPhase == RegistryPhase.OPEN
                            ? prefillDirectoryAfterMutationLocked(
                                    prefillAddressDirectory,
                                    address,
                                    exactCurrent,
                                    next)
                            : List.of()
                    : null;
            List<Map.Entry<String, DecodeEndpoint>> nextDecodeDirectory =
                    updatesDecodeDirectory
                            ? decodeDirectoryAfterMutationLocked(
                                    this.decodeDirectory,
                                    address,
                                    (DecodeEndpoint) next)
                            : null;
            WorkerEndpoint published = endpoints.compute(
                    address, (ignored, observed) -> {
                if (observed != exactCurrent) {
                    throw new IllegalStateException(
                            role + " endpoint mapping changed outside its directory transaction: "
                                    + address);
                }
                return next;
            });
            if (updatesPrefillDirectory) {
                prefillAddressDirectory = nextPrefillDirectory;
            } else {
                this.decodeDirectory = nextDecodeDirectory;
            }
            return published;
        }
    }

    /** Build the next immutable Prefill address view before its CHM write. */
    private List<String> prefillDirectoryAfterMutationLocked(
            List<String> previous,
            String address,
            WorkerEndpoint exactCurrent,
            WorkerEndpoint next) {
        boolean wasPresent = exactCurrent != null;
        boolean willBePresent = next != null;
        if (wasPresent == willBePresent) {
            return previous;
        }
        List<String> updated = new ArrayList<>(
                previous.size() + (willBePresent ? 1 : 0));
        updated.addAll(previous);
        if (willBePresent) {
            updated.add(address);
        } else {
            updated.remove(address);
        }
        return List.copyOf(updated);
    }

    /** Build the exact post-mutation directory before the CHM write. */
    private List<Map.Entry<String, DecodeEndpoint>>
            decodeDirectoryAfterMutationLocked(
            List<Map.Entry<String, DecodeEndpoint>> previous,
            String address,
            DecodeEndpoint next) {
        if (registryPhase != RegistryPhase.OPEN) {
            return List.of();
        }
        List<Map.Entry<String, DecodeEndpoint>> entries = new ArrayList<>(
                previous.size() + (next == null ? 0 : 1));
        for (Map.Entry<String, DecodeEndpoint> entry : previous) {
            if (!entry.getKey().equals(address)) {
                entries.add(entry);
            }
        }
        if (next != null) {
            entries.add(Map.entry(address, next));
        }
        return List.copyOf(entries);
    }

    private static void closeCandidate(
            WorkerEndpoint candidate,
            Throwable primaryFailure) {
        if (candidate == null) {
            return;
        }
        try {
            candidate.closeAsynchronously();
        } catch (Throwable closeFailure) {
            addSuppressedNoFail(primaryFailure, closeFailure);
        }
    }

    private static void startCandidate(WorkerEndpoint candidate) {
        if (candidate instanceof PrefillEndpoint prefill) {
            prefill.startGeneration();
        }
    }

    private static void addSuppressedNoFail(
            Throwable primary,
            Throwable leaf) {
        if (primary == null || leaf == null || primary == leaf) {
            return;
        }
        try {
            primary.addSuppressed(leaf);
        } catch (Throwable ignoredAggregationFailure) {
            // The exact candidate cleanup was still initiated.
        }
    }

    private static RuntimeException propagate(
            Throwable failure,
            String ipPort) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "Endpoint publication failed for " + ipPort, failure);
    }

    private static void requireGenerationLock(WorkerStatus status) {
        if (!status.lock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "Endpoint publication requires the WorkerStatus generation lock");
        }
    }

    /**
     * Close and remove one exact endpoint generation from routing without
     * waiting for its potentially blocking drain. The caller holds the ACTIVE
     * WorkerStatus generation lock. This method publishes RETIRING only after
     * the exact endpoint gate has closed and routing withdrawal is visible;
     * the caller closes the returned endpoint outside that lock.
     */
    public DetachedGeneration detachAndBeginRetirement(
            RoleType roleType,
            String ipPort,
            WorkerStatus expectedStatus) {
        if (expectedStatus == null) {
            return null;
        }
        expectedStatus.requireActiveGeneration();
        synchronized (lifecycleGate) {
            if (registryPhase != RegistryPhase.OPEN) {
                throw new IllegalStateException(
                        "EndpointRegistry is closing");
            }
            ConcurrentHashMap<String, WorkerEndpoint> endpoints =
                    endpoints(roleType);
            DetachedGeneration detached = endpoints == null
                    ? null
                    : detach(roleType, endpoints, ipPort, expectedStatus);
            if (detached != null) {
                inflightDetachedRetirements++;
                if (!expectedStatus
                        .beginRetirementAfterEndpointGateClosed()) {
                    throw new IllegalStateException(
                            "WorkerStatus generation changed while its lock was held: "
                                    + ipPort + "#"
                                    + expectedStatus.getGenerationId());
                }
            }
            return detached;
        }
    }

    private DetachedGeneration detach(
            RoleType role,
            ConcurrentHashMap<String, WorkerEndpoint> endpoints,
            String ipPort,
            WorkerStatus expectedStatus) {
        if (!Thread.holdsLock(lifecycleGate)) {
            throw new IllegalStateException(
                    "Endpoint detach requires the registry lifecycle gate");
        }
        AtomicReference<DetachedGeneration> detached =
                new AtomicReference<>();
        mutateEndpointMap(
                role,
                endpoints,
                ipPort,
                (ignored, current) -> {
                    if (current == null) {
                        return null;
                    }
                    if (current.getStatus() != expectedStatus) {
                        return current;
                    }
                    // Admission closes while the exact mapping is still
                    // current. Removal becomes visible only after this
                    // non-blocking transition succeeds.
                    current.beginRetirement();
                    detached.set(new DetachedGeneration(this, current));
                    return null;
                });
        return detached.get();
    }

    /** Resolve one exact detached-generation barrier token. */
    private void resolveDetachedGeneration(DetachedGeneration exact) {
        synchronized (lifecycleGate) {
            if (exact == null || exact.registry != this) {
                throw new IllegalArgumentException(
                        "Detached generation belongs to another registry");
            }
            if (inflightDetachedRetirements <= 0) {
                throw new IllegalStateException(
                        "Detached retirement count underflow");
            }
            inflightDetachedRetirements--;
            if (inflightDetachedRetirements == 0) {
                lifecycleGate.notifyAll();
            }
        }
    }

    private WorkerEndpoint createEndpoint(
            WorkerStatus status,
            RoleType role,
            WorkerStatus.EngineObservation engineStatus) {
        if (role == RoleType.FRONTEND) {
            throw new IllegalArgumentException("Unsupported role: " + role);
        }
        boolean prefill = role == RoleType.PREFILL
                || role == RoleType.PDFUSION;
        if (prefill && engineStatus.dpSize() > 1) {
            throw new UnsupportedOperationException(
                    role + " DP group endpoint not yet supported: ipPort="
                            + status.getIpPort() + ", dp_size="
                            + engineStatus.dpSize());
        }
        prepareEndpointMetrics(role, status);
        return switch (role) {
            case PREFILL, PDFUSION -> new PrefillEndpoint(
                    status,
                    configService.loadBalanceConfig(),
                    deliveryStrategy,
                    endpointEvents,
                    reporter,
                    placementAvailability);
            case DECODE -> new DecodeEndpoint(
                    status, endpointEvents, placementAvailability);
            case VIT -> new WorkerEndpoint(status);
            case FRONTEND -> throw new AssertionError("validated above");
        };
    }

    private void prepareEndpointMetrics(RoleType roleType, WorkerStatus status) {
        try {
            reporter.prepareEndpointMetrics(roleType.name(), status.getIp());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("Endpoint metric preparation failed: role={}, engine={}",
                    roleType, status.getIp(), telemetryFailure);
        }
    }

    private void signalPublishedEndpoint(WorkerEndpoint endpoint) {
        if (endpoint == null) {
            return;
        }
        WorkerStatus.TopologySnapshot topology =
                endpoint.getStatus().topologySnapshot();
        placementAvailability.capacityChanged(
                endpoint.getStatus().getRole(), topology.group());
    }

    public void close() {
        boolean interrupted = false;
        boolean closeOwner = false;
        synchronized (lifecycleGate) {
            if (registryPhase == RegistryPhase.OPEN) {
                registryPhase = RegistryPhase.CLOSING;
                prefillAddressDirectory = List.of();
                decodeDirectory = List.of();
                closeOwner = true;
            }
            while (registryPhase == RegistryPhase.CLOSING
                    && (!closeOwner || inflightPublications != 0)) {
                try {
                    lifecycleGate.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            if (!closeOwner) {
                Throwable completedFailure = closeFailure;
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                rethrowCloseFailure(completedFailure);
                return;
            }
        }

        Throwable failure;
        try {
            failure = closeEndpointGenerations();
        } catch (RuntimeException | Error unexpectedFailure) {
            failure = unexpectedFailure;
        }
        try {
            awaitDetachedRetirements();
        } catch (RuntimeException | Error detachedBarrierFailure) {
            failure = appendFailure(failure, detachedBarrierFailure);
        }
        synchronized (lifecycleGate) {
            closeFailure = failure;
            registryPhase = RegistryPhase.CLOSED;
            lifecycleGate.notifyAll();
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        rethrowCloseFailure(failure);
    }

    /** Wait for every generation detached before the close gate linearized. */
    private void awaitDetachedRetirements() {
        boolean interrupted = false;
        synchronized (lifecycleGate) {
            while (inflightDetachedRetirements != 0) {
                try {
                    lifecycleGate.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private Throwable closeEndpointGenerations() {
        IdentityHashMap<WorkerEndpoint, Boolean> seen =
                new IdentityHashMap<>();
        List<WorkerEndpoint> endpoints = new ArrayList<>();
        Throwable failure = null;
        for (Map<String, WorkerEndpoint> roleEndpoints
                : endpointsByRole.values()) {
            failure = collectEndpointGenerations(
                    roleEndpoints, seen, endpoints, failure);
        }
        // Phase 1: close admission for every exact generation before any
        // endpoint-local cleanup or callback can run.
        for (WorkerEndpoint endpoint : endpoints) {
            try {
                endpoint.beginRetirement();
            } catch (Throwable retirementFailure) {
                failure = appendFailure(failure, retirementFailure);
            }
        }
        // Phase 2: initiate every cleanup. A generation with an accepted
        // handoff may transfer its cleanup to a dedicated continuation.
        for (WorkerEndpoint endpoint : endpoints) {
            try {
                endpoint.close();
            } catch (Throwable closeEndpointFailure) {
                failure = appendFailure(failure, closeEndpointFailure);
            }
        }
        // Phase 3: the registry is not closed until every exact generation has
        // completed cleanup and all retirement callbacks have returned.
        for (WorkerEndpoint endpoint : endpoints) {
            try {
                endpoint.awaitRetirement();
            } catch (Throwable awaitFailure) {
                failure = appendFailure(failure, awaitFailure);
            }
        }
        return failure;
    }

    private static Throwable collectEndpointGenerations(
            Map<String, ? extends WorkerEndpoint> source,
            IdentityHashMap<WorkerEndpoint, Boolean> seen,
            List<WorkerEndpoint> endpoints,
            Throwable failure) {
        try {
            for (WorkerEndpoint endpoint : source.values()) {
                if (seen.put(endpoint, Boolean.TRUE) == null) {
                    endpoints.add(endpoint);
                }
            }
        } catch (RuntimeException | Error snapshotFailure) {
            return appendFailure(failure, snapshotFailure);
        }
        return failure;
    }

    private static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (next == null) {
            return first;
        }
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static void rethrowCloseFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "EndpointRegistry close failed", failure);
        }
    }

    private static void rethrowDetachedRetirementFailure(
            Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Detached endpoint retirement failed", failure);
        }
    }

    /** Immutable point-in-time view; publication remains owned by this registry. */
    @SuppressWarnings("unchecked")
    public Map<String, PrefillEndpoint> snapshotPrefillEndpoints() {
        return (Map<String, PrefillEndpoint>) (Map<?, ?>)
                Map.copyOf(endpoints(RoleType.PREFILL));
    }

    /** Immutable point-in-time view; publication remains owned by this registry. */
    @SuppressWarnings("unchecked")
    public Map<String, DecodeEndpoint> snapshotDecodeEndpoints() {
        return (Map<String, DecodeEndpoint>) (Map<?, ?>)
                Map.copyOf(endpoints(RoleType.DECODE));
    }

    public int getEndpointCount(RoleType roleType) {
        Map<String, WorkerEndpoint> endpoints = endpoints(roleType);
        return endpoints == null ? 0 : endpoints.size();
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    public void evictExpiredOrphans(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        endpoints(RoleType.PREFILL).forEach((endpoint, worker) -> {
            PrefillEndpoint ep = (PrefillEndpoint) worker;
            logEndpointEviction(RoleType.PREFILL, endpoint,
                    ep.evictExpiredInflight(
                            ttlMs, schedulerOwnsRequest), ttlMs);
        });
        endpoints(RoleType.DECODE).forEach((endpoint, worker) -> {
            DecodeEndpoint ep = (DecodeEndpoint) worker;
            logEndpointEviction(RoleType.DECODE, endpoint,
                    ep.evictExpiredRequests(
                            ttlMs, schedulerOwnsRequest), ttlMs);
        });
        endpoints(RoleType.PDFUSION).forEach((endpoint, worker) -> {
            PrefillEndpoint ep = (PrefillEndpoint) worker;
            logEndpointEviction(RoleType.PDFUSION, endpoint,
                    ep.evictExpiredBatches(ttlMs), ttlMs);
        });
    }

    private static void logEndpointEviction(RoleType role,
                                            String endpoint,
                                            int evicted,
                                            long ttlMs) {
        if (evicted > 0) {
            Logger.info("event=endpoint_inflight_ttl_eviction role={} endpoint={} "
                            + "evicted={} ttl_ms={}",
                    role, endpoint, evicted, ttlMs);
        }
    }

}
