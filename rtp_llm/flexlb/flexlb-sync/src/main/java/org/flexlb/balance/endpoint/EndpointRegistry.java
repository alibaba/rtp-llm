package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.LongPredicate;

@Component
public class EndpointRegistry {

    private static final int DECODE_DIRECTORY_SPIN_LIMIT = 32;

    private enum RegistryPhase {
        OPEN,
        CLOSING,
        CLOSED
    }

    /** Result of publishing a fully initialized endpoint generation. */
    public record EndpointPublication(
            WorkerEndpoint endpoint,
            EndpointStatusReduction statusReduction) {
    }

    /**
     * Immutable, non-owning Decode routing projection.
     *
     * <p>The projection deliberately contains no endpoint or registry-map
     * reference.  Its generation and admission version are observation tokens,
     * not admission authority; a route must still exact-capture and revalidate
     * the winning generation before it can hand off a request.</p>
     */
    public record DecodeRoutingSnapshot(
            String address,
            long generationId,
            WorkerStatus.TopologySnapshot topology,
            DecodeEndpoint.DecodeRoutingView routing) {
        public DecodeRoutingSnapshot {
            if (generationId <= 0L) {
                throw new IllegalArgumentException(
                        "Decode routing snapshot requires a positive generation");
            }
        }
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

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
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
     * Copy-on-write Decode directory used by the routing hot path.
     *
     * <p>The private entries intentionally retain exact endpoint identities;
     * none of them escape this registry. Every structural Decode-map writer
     * first publishes a distinct mutating state and, in the same
     * {@link #lifecycleGate} transaction, publishes a stable directory rebuilt
     * from the map. A reader validates the stable state identity after its
     * traversal, so it can never return a complete pre-mutation directory.</p>
     */
    private volatile DecodeDirectoryState decodeDirectoryState =
            new DecodeDirectoryState(0L, true, List.of());
    /** Preallocated close publication: close cannot strand an old routable view. */
    private final DecodeDirectoryState closedDecodeDirectoryState =
            new DecodeDirectoryState(Long.MAX_VALUE, true, List.of());
    private final ConfigService configService;
    private final EndpointEventSink endpointEventSink;
    private final BatchSchedulerReporter reporter;
    private final DeliveryStrategy deliveryStrategy;
    private final PlacementAvailability placementAvailability;
    private final Object lifecycleGate = new Object();
    private RegistryPhase registryPhase = RegistryPhase.OPEN;
    private int inflightPublications;
    private int inflightDetachedRetirements;
    private Throwable closeFailure;

    private record DecodeDirectoryState(
            long sequence,
            boolean stable,
            List<DecodeDirectoryEntry> entries) {

        private DecodeDirectoryState {
            entries = List.copyOf(entries);
        }
    }

    /** Private owner-bearing entry; only its neutral projection may escape. */
    private static final class DecodeDirectoryEntry {
        private final String address;
        private final DecodeEndpoint endpoint;
        private volatile DecodeSnapshotCache snapshotCache;

        private DecodeDirectoryEntry(
                String address,
                DecodeEndpoint endpoint) {
            this.address = address;
            this.endpoint = endpoint;
        }

        private DecodeRoutingSnapshot routingSnapshot() {
            WorkerStatus status = endpoint.getStatus();
            WorkerStatus.TopologySnapshot topology =
                    status.topologySnapshot();
            DecodeEndpoint.DecodeRoutingView routing =
                    endpoint.routingViewSnapshot();
            DecodeSnapshotCache cached = snapshotCache;
            if (cached != null
                    && cached.endpoint() == endpoint
                    && cached.routing() == routing
                    && (cached.topology() == topology
                            || cached.topology().equals(topology))) {
                return cached.snapshot();
            }
            DecodeRoutingSnapshot snapshot = new DecodeRoutingSnapshot(
                    address,
                    status.getGenerationId(),
                    topology,
                    routing);
            snapshotCache = new DecodeSnapshotCache(
                    endpoint, topology, routing, snapshot);
            return snapshot;
        }
    }

    /** Exact cache key for one neutral Decode routing projection. */
    private record DecodeSnapshotCache(
            DecodeEndpoint endpoint,
            WorkerStatus.TopologySnapshot topology,
            DecodeEndpoint.DecodeRoutingView routing,
            DecodeRoutingSnapshot snapshot) {
    }

    public EndpointRegistry(ConfigService configService,
                            EndpointEventSink endpointEventSink,
                            BatchSchedulerReporter reporter,
                            DeliveryStrategy deliveryStrategy) {
        this(configService, endpointEventSink, reporter,
                deliveryStrategy, new PlacementAvailability());
    }

    @Autowired
    public EndpointRegistry(ConfigService configService,
                            EndpointEventSink endpointEventSink,
                            BatchSchedulerReporter reporter,
                            DeliveryStrategy deliveryStrategy,
                            PlacementAvailability placementAvailability) {
        this.configService = java.util.Objects.requireNonNull(
                configService, "configService");
        this.endpointEventSink = java.util.Objects.requireNonNull(
                endpointEventSink, "endpointEventSink");
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.deliveryStrategy = java.util.Objects.requireNonNull(
                deliveryStrategy, "deliveryStrategy");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        if (roleType == RoleType.PREFILL) {
            return getPrefill(ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return getDecode(ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return getPdFusion(ipPort);
        }
        if (roleType == RoleType.VIT) {
            return getVit(ipPort);
        }
        return null;
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
        if (roleType == RoleType.PREFILL) {
            return capture(prefillEndpoints, ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return capture(decodeEndpoints, ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return capture(pdFusionEndpoints, ipPort);
        }
        if (roleType == RoleType.VIT) {
            return capture(vitEndpoints, ipPort);
        }
        return null;
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
        if (roleType == RoleType.DECODE) {
            return List.copyOf(decodeEndpoints.keySet());
        }
        if (roleType == RoleType.PDFUSION) {
            return List.copyOf(pdFusionEndpoints.keySet());
        }
        if (roleType == RoleType.VIT) {
            return List.copyOf(vitEndpoints.keySet());
        }
        return List.of();
    }

    /**
     * Return immutable routing values for the Decode generations observed
     * during this traversal.
     *
     * <p>The registry is intentionally not locked while an endpoint takes its
     * admission lock.  This avoids introducing a map-bin/admission-lock order;
     * publication races are resolved when the selected generation is pinned
     * by {@link #captureDecodeGeneration(DecodeRoutingSnapshot)}.</p>
     */
    public List<DecodeRoutingSnapshot> decodeRoutingSnapshot() {
        int spins = 0;
        while (true) {
            DecodeDirectoryState directory = decodeDirectoryState;
            if (!directory.stable()) {
                if (++spins < DECODE_DIRECTORY_SPIN_LIMIT) {
                    Thread.onSpinWait();
                } else {
                    // A Decode writer holds this gate until it publishes its
                    // stable outcome. Enter and leave only to await that
                    // publication; endpoint traversal remains lock-free.
                    synchronized (lifecycleGate) {
                        // Memory synchronization is the only operation needed.
                    }
                    spins = 0;
                }
                continue;
            }
            spins = 0;
            List<DecodeRoutingSnapshot> snapshots =
                    new ArrayList<>(directory.entries().size());
            for (DecodeDirectoryEntry entry : directory.entries()) {
                snapshots.add(entry.routingSnapshot());
            }
            if (decodeDirectoryState == directory) {
                // The local ArrayList is never mutated after publication. Use
                // an unmodifiable view so the hot path does not copy its
                // backing array a second time.
                return Collections.unmodifiableList(snapshots);
            }
        }
    }

    /**
     * Capture the exact Decode generation represented by a routing snapshot.
     *
     * <p>The routing values are advisory and may change after selection. The
     * subsequent reservation/dispatch transaction revalidates capacity under
     * the endpoint admission lock. Requiring an identical admission version
     * here would turn ordinary request traffic into a false placement miss.
     * Every rejected or exceptional capture is closed here; a successful
     * caller owns the returned generation pin.</p>
     */
    public WorkerEndpoint.GenerationPin captureDecodeGeneration(
            DecodeRoutingSnapshot expected) {
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

    private static <T extends WorkerEndpoint>
            WorkerEndpoint.GenerationPin capture(
            ConcurrentHashMap<String, T> endpoints,
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

    public PrefillEndpoint getPrefill(String ipPort) {
        return prefillEndpoints.get(ipPort);
    }

    public DecodeEndpoint getDecode(String ipPort) {
        return decodeEndpoints.get(ipPort);
    }

    private PrefillEndpoint getPdFusion(String ipPort) {
        return pdFusionEndpoints.get(ipPort);
    }

    private SimpleWorkerEndpoint getVit(String ipPort) {
        return vitEndpoints.get(ipPort);
    }

    /**
     * Register an endpoint whose status has already been initialized by a
     * test fixture or an embedded harness.
     *
     * <p>Production status synchronization uses the staged/committed methods
     * below so an uninitialized endpoint can never become routable.</p>
     */
    public WorkerEndpoint registerPreinitializedEndpoint(
            RoleType roleType,
            String ipPort,
            WorkerStatus status) {
        beginCandidatePublication();
        try {
            if (!status.isActiveGeneration()) {
                throw new IllegalStateException(
                        "Cannot publish endpoint for retiring WorkerStatus generation: "
                                + status.getGenerationId());
            }
            if (roleType == RoleType.PREFILL) {
                return registerPreinitializedPrefillEndpoint(
                        ipPort, status, roleType);
            }
            if (roleType == RoleType.DECODE) {
                return registerPreinitializedDecodeEndpoint(ipPort, status);
            }
            if (roleType == RoleType.PDFUSION) {
                return registerPreinitializedPdFusionEndpoint(
                        ipPort, status, roleType);
            }
            if (roleType == RoleType.VIT) {
                return registerPreinitializedVitEndpoint(ipPort, status);
            }
            throw new IllegalArgumentException(
                    "Unsupported role: " + roleType);
        } finally {
            endCandidatePublication();
        }
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
    public EndpointPublication initializeAndPublishNewStatusEndpoint(
            RoleType roleType,
            String ipPort,
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
            if (status.appliedStatusCursor().statusVersion() >= 0L) {
                throw new IllegalStateException(
                        "A committed WorkerStatus generation cannot publish a second endpoint: "
                                + ipPort + "#" + status.getGenerationId());
            }
            WorkerStatus.EngineObservation staged = observation.engine();
            if (roleType == RoleType.PREFILL) {
                return initializeAndPublishNewStatusEndpoint(
                        prefillEndpoints, ipPort, status,
                        candidateStatus -> createPrefillEndpoint(
                                candidateStatus, roleType, staged), prepared);
            }
            if (roleType == RoleType.DECODE) {
                return initializeAndPublishNewStatusEndpoint(
                        decodeEndpoints, ipPort, status,
                        candidateStatus -> createDecodeEndpoint(
                                candidateStatus, staged), prepared);
            }
            if (roleType == RoleType.PDFUSION) {
                return initializeAndPublishNewStatusEndpoint(
                        pdFusionEndpoints, ipPort, status,
                        candidateStatus -> createPrefillEndpoint(
                                candidateStatus, roleType, staged), prepared);
            }
            if (roleType == RoleType.VIT) {
                return initializeAndPublishNewStatusEndpoint(
                        vitEndpoints, ipPort, status,
                        candidateStatus -> createSimpleEndpoint(
                                candidateStatus, RoleType.VIT, staged),
                        prepared);
            }
            throw new IllegalArgumentException(
                    "Unsupported role: " + roleType);
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

    private PrefillEndpoint registerPreinitializedPrefillEndpoint(
            String ipPort,
            WorkerStatus status,
            RoleType roleType) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        WorkerStatus.EngineObservation engine =
                status.committedEngineObservation();
        return registerPreinitializedEndpoint(prefillEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(
                        candidateStatus, roleType, engine));
    }

    private DecodeEndpoint registerPreinitializedDecodeEndpoint(
            String ipPort,
            WorkerStatus status) {
        WorkerStatus.EngineObservation engine =
                status.committedEngineObservation();
        return registerPreinitializedEndpoint(decodeEndpoints, ipPort, status,
                candidateStatus -> createDecodeEndpoint(
                        candidateStatus, engine));
    }

    private PrefillEndpoint registerPreinitializedPdFusionEndpoint(
            String ipPort,
            WorkerStatus status,
            RoleType roleType) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        WorkerStatus.EngineObservation engine =
                status.committedEngineObservation();
        return registerPreinitializedEndpoint(pdFusionEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(
                        candidateStatus, roleType, engine));
    }

    private SimpleWorkerEndpoint registerPreinitializedVitEndpoint(
            String ipPort,
            WorkerStatus status) {
        WorkerStatus.EngineObservation engine =
                status.committedEngineObservation();
        return registerPreinitializedEndpoint(vitEndpoints, ipPort, status,
                candidateStatus -> createSimpleEndpoint(
                        candidateStatus, RoleType.VIT, engine));
    }

    private <T extends WorkerEndpoint> T registerPreinitializedEndpoint(
            ConcurrentHashMap<String, T> endpoints,
            String ipPort,
            WorkerStatus status,
            Function<WorkerStatus, T> factory) {
        AtomicReference<T> published = new AtomicReference<>();
        AtomicReference<T> displaced = new AtomicReference<>();
        T candidate = factory.apply(status);
        BiFunction<String, T, T> publicationAction;
        try {
            publicationAction = (ignored, current) -> {
                if (current != null && current.getStatus() == status) {
                    published.set(current);
                    return current;
                }
                if (current != null) {
                    // Close A's admission gate before B becomes visible at the
                    // same address. Final drain remains outside the map lock.
                    current.beginRetirement();
                    displaced.set(current);
                }
                published.set(candidate);
                return candidate;
            };
            startCandidate(candidate);
            mutateEndpointMap(endpoints, ipPort, publicationAction);
        } catch (Throwable publicationFailure) {
            withdrawAndCloseCandidate(
                    endpoints, ipPort, candidate, publicationFailure);
            throw propagate(publicationFailure, ipPort);
        }

        T result = published.get();
        if (result != candidate) {
            Throwable duplicateFailure = null;
            try {
                candidate.close();
            } catch (Throwable closeFailure) {
                duplicateFailure = closeFailure;
            }
            try {
                candidate.awaitRetirement();
            } catch (Throwable awaitFailure) {
                duplicateFailure = appendFailure(
                        duplicateFailure, awaitFailure);
            }
            if (duplicateFailure != null) {
                throw propagate(duplicateFailure, ipPort);
            }
        }
        T retired = displaced.get();
        if (retired != null) {
            Throwable retirementFailure = null;
            try {
                retired.close();
            } catch (Throwable closeFailure) {
                retirementFailure = closeFailure;
            }
            try {
                retired.awaitRetirement();
            } catch (Throwable awaitFailure) {
                retirementFailure = appendFailure(
                        retirementFailure, awaitFailure);
            }
            if (retirementFailure != null) {
                throw propagate(retirementFailure, ipPort);
            }
        }
        signalPublishedEndpoint(result);
        return result;
    }

    private <T extends WorkerEndpoint> EndpointPublication
            initializeAndPublishNewStatusEndpoint(
            ConcurrentHashMap<String, T> endpoints,
            String ipPort,
            WorkerStatus status,
            Function<WorkerStatus, T> factory,
            WorkerStatus.PreparedStatus prepared) {
        T candidate = null;
        EndpointStatusReduction reduction;
        EndpointPublication publication;
        BiFunction<String, T, T> publicationAction;
        try {
            candidate = factory.apply(status);
            reduction = candidate.initializeFromPreparedStatus(
                    status, prepared.observation());
            publication = new EndpointPublication(candidate, reduction);
            T exactCandidate = candidate;
            publicationAction = (ignored, current) -> {
                if (current != null && current.getStatus() == status) {
                    throw new IllegalStateException(
                            "Endpoint generation is already published for "
                                    + ipPort);
                }
                if (current != null) {
                    throw new IllegalStateException(
                            "Existing endpoint generation must be withdrawn before publication for "
                                    + ipPort);
                }
                return exactCandidate;
            };
            startCandidate(candidate);
            status.publishPreparedStatus(prepared);
        } catch (Throwable reductionOrCommitFailure) {
            closeCandidate(candidate, reductionOrCommitFailure);
            throw propagate(reductionOrCommitFailure, ipPort);
        }
        return publishPrivateEndpoint(
                endpoints, ipPort, candidate, publication,
                publicationAction);
    }

    private <T extends WorkerEndpoint> EndpointPublication publishPrivateEndpoint(
            ConcurrentHashMap<String, T> endpoints,
            String ipPort,
            T candidate,
            EndpointPublication publication,
            BiFunction<String, T, T> publicationAction) {
        try {
            mutateEndpointMap(endpoints, ipPort, publicationAction);
            signalPublishedEndpoint(candidate);
            return publication;
        } catch (Throwable publicationFailure) {
            withdrawAndCloseCandidate(
                    endpoints, ipPort, candidate, publicationFailure);
            throw propagate(publicationFailure, ipPort);
        }
    }

    private <T extends WorkerEndpoint> void withdrawAndCloseCandidate(
            ConcurrentHashMap<String, T> endpoints,
            String ipPort,
            T candidate,
            Throwable primaryFailure) {
        try {
            mutateEndpointMap(
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
     * Execute one exact-address map mutation. Decode mutations additionally
     * form a seqlock-style copy-on-write directory transaction.
     *
     * <p>For Decode, both the mutating marker and the complete stable outcome
     * are allocated before the CHM write. After a successful write the only
     * remaining operation is one volatile publication. If the CHM operation
     * throws, restoring the previous stable identity is likewise allocation
     * free, so readers can never be stranded on a mutating state.</p>
     */
    private <T extends WorkerEndpoint> T mutateEndpointMap(
            ConcurrentHashMap<String, T> endpoints,
            String address,
            BiFunction<String, T, T> mutation) {
        if (endpoints == (Object) prefillEndpoints) {
            return mutatePrefillEndpointMap(
                    endpoints, address, mutation);
        }
        if (endpoints != (Object) decodeEndpoints) {
            return endpoints.compute(address, mutation);
        }
        synchronized (lifecycleGate) {
            DecodeDirectoryState previous = decodeDirectoryState;
            if (!previous.stable()) {
                throw new IllegalStateException(
                        "Decode directory writer observed a mutating state");
            }

            T exactCurrent = endpoints.get(address);
            T next = mutation.apply(address, exactCurrent);
            if (next == exactCurrent) {
                return exactCurrent;
            }
            DecodeDirectoryState mutating = new DecodeDirectoryState(
                    previous.sequence() + 1L, false, List.of());
            DecodeDirectoryState stable = decodeDirectoryAfterMutationLocked(
                    previous,
                    address,
                    (DecodeEndpoint) exactCurrent,
                    (DecodeEndpoint) next);

            decodeDirectoryState = mutating;
            T published;
            try {
                published = endpoints.compute(address, (ignored, observed) -> {
                    if (observed != exactCurrent) {
                        throw new IllegalStateException(
                                "Decode endpoint mapping changed outside its directory transaction: "
                                        + address);
                    }
                    return next;
                });
            } catch (RuntimeException | Error mutationFailure) {
                decodeDirectoryState = previous;
                throw mutationFailure;
            }
            decodeDirectoryState = stable;
            return published;
        }
    }

    /**
     * Apply one Prefill map mutation and its immutable address publication as a
     * single writer transaction.
     *
     * <p>The next directory is fully allocated before the CHM write. A failed
     * mutation therefore leaves the previous directory published. Replacement
     * at an existing address reuses that exact directory instance because
     * generation ownership is established only by {@link #capture}.</p>
     */
    private <T extends WorkerEndpoint> T mutatePrefillEndpointMap(
            ConcurrentHashMap<String, T> endpoints,
            String address,
            BiFunction<String, T, T> mutation) {
        synchronized (lifecycleGate) {
            T exactCurrent = endpoints.get(address);
            T next = mutation.apply(address, exactCurrent);
            if (next == exactCurrent) {
                return exactCurrent;
            }

            List<String> nextDirectory = registryPhase == RegistryPhase.OPEN
                    ? prefillDirectoryAfterMutationLocked(
                            prefillAddressDirectory,
                            address,
                            exactCurrent,
                            next)
                    : List.of();
            T published = endpoints.compute(address, (ignored, observed) -> {
                if (observed != exactCurrent) {
                    throw new IllegalStateException(
                            "Prefill endpoint mapping changed outside its directory transaction: "
                                    + address);
                }
                return next;
            });
            prefillAddressDirectory = nextDirectory;
            return published;
        }
    }

    /** Build the exact, stable-order Prefill directory before its CHM write. */
    private List<String> prefillDirectoryAfterMutationLocked(
            List<String> previous,
            String address,
            WorkerEndpoint exactCurrent,
            WorkerEndpoint next) {
        if (!Thread.holdsLock(lifecycleGate)) {
            throw new IllegalStateException(
                    "Prefill directory construction requires the lifecycle gate");
        }
        boolean wasPresent = exactCurrent != null;
        boolean willBePresent = next != null;
        if (wasPresent == willBePresent) {
            return previous;
        }

        int addressIndex = previous.indexOf(address);
        if (!wasPresent) {
            if (addressIndex >= 0) {
                throw new IllegalStateException(
                        "Prefill directory contains an unpublished endpoint: "
                                + address);
            }
            List<String> updated = new ArrayList<>(previous.size() + 1);
            updated.addAll(previous);
            updated.add(address);
            return List.copyOf(updated);
        }
        if (addressIndex < 0) {
            throw new IllegalStateException(
                    "Prefill directory does not contain its published endpoint: "
                            + address);
        }
        List<String> updated = new ArrayList<>(previous);
        updated.remove(addressIndex);
        return List.copyOf(updated);
    }

    /** Build the exact post-mutation stable state before the CHM write. */
    private DecodeDirectoryState decodeDirectoryAfterMutationLocked(
            DecodeDirectoryState previous,
            String address,
            DecodeEndpoint exactCurrent,
            DecodeEndpoint next) {
        if (!Thread.holdsLock(lifecycleGate)) {
            throw new IllegalStateException(
                    "Decode directory construction requires the lifecycle gate");
        }
        if (registryPhase != RegistryPhase.OPEN) {
            return new DecodeDirectoryState(
                    previous.sequence() + 2L, true, List.of());
        }

        List<DecodeDirectoryEntry> entries = new ArrayList<>(
                previous.entries().size() + (exactCurrent == null ? 1 : 0));
        boolean found = false;
        for (DecodeDirectoryEntry entry : previous.entries()) {
            if (!entry.address.equals(address)) {
                entries.add(entry);
                continue;
            }
            if (found || entry.endpoint != exactCurrent) {
                throw new IllegalStateException(
                        "Decode directory does not match its endpoint map: "
                                + address);
            }
            found = true;
            if (next == exactCurrent) {
                entries.add(entry);
            } else if (next != null) {
                entries.add(new DecodeDirectoryEntry(address, next));
            }
        }
        if (!found) {
            if (exactCurrent != null) {
                throw new IllegalStateException(
                        "Decode directory is missing its endpoint mapping: "
                                + address);
            }
            if (next != null) {
                entries.add(new DecodeDirectoryEntry(address, next));
            }
        }
        return new DecodeDirectoryState(
                previous.sequence() + 2L, true, entries);
    }

    /** Publish an empty stable directory when registry close linearizes. */
    private void publishClosedDecodeDirectoryLocked() {
        if (!Thread.holdsLock(lifecycleGate)) {
            throw new IllegalStateException(
                    "Decode directory close requires the lifecycle gate");
        }
        decodeDirectoryState = closedDecodeDirectoryState;
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
            DetachedGeneration detached;
            if (roleType == RoleType.PREFILL) {
                detached = detach(
                        prefillEndpoints, ipPort, expectedStatus);
            } else if (roleType == RoleType.DECODE) {
                detached = detach(
                        decodeEndpoints, ipPort, expectedStatus);
            } else if (roleType == RoleType.PDFUSION) {
                detached = detach(
                        pdFusionEndpoints, ipPort, expectedStatus);
            } else if (roleType == RoleType.VIT) {
                detached = detach(
                        vitEndpoints, ipPort, expectedStatus);
            } else {
                detached = null;
            }
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

    private <T extends WorkerEndpoint> DetachedGeneration detach(
            ConcurrentHashMap<String, T> endpoints,
            String ipPort,
            WorkerStatus expectedStatus) {
        if (!Thread.holdsLock(lifecycleGate)) {
            throw new IllegalStateException(
                    "Endpoint detach requires the registry lifecycle gate");
        }
        AtomicReference<DetachedGeneration> detached =
                new AtomicReference<>();
        mutateEndpointMap(
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

    private PrefillEndpoint createPrefillEndpoint(
            WorkerStatus status,
            RoleType roleType,
            WorkerStatus.EngineObservation engineStatus) {
        if (engineStatus.dpSize() > 1) {
            throw new UnsupportedOperationException(
                    roleType + " DP group endpoint not yet supported: ipPort="
                            + status.getIpPort() + ", dp_size="
                            + engineStatus.dpSize());
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(roleType, status);
        return new PrefillEndpoint(
                status,
                config,
                deliveryStrategy,
                endpointEventSink,
                endpointEventSink,
                reporter,
                placementAvailability);
    }

    private DecodeEndpoint createDecodeEndpoint(
            WorkerStatus status,
            WorkerStatus.EngineObservation engineStatus) {
        prepareEndpointMetrics(RoleType.DECODE, status);
        return new DecodeEndpoint(
                status, endpointEventSink, placementAvailability);
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(
            WorkerStatus status,
            RoleType roleType,
            WorkerStatus.EngineObservation engineStatus) {
        prepareEndpointMetrics(roleType, status);
        return new SimpleWorkerEndpoint(status);
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
                publishClosedDecodeDirectoryLocked();
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
        EndpointGenerationSnapshot snapshot = snapshotEndpointGenerations();
        Throwable failure = snapshot.failure();
        // Phase 1: close admission for every exact generation before any
        // endpoint-local cleanup or callback can run.
        for (WorkerEndpoint endpoint : snapshot.endpoints()) {
            try {
                endpoint.beginRetirement();
            } catch (Throwable retirementFailure) {
                failure = appendFailure(failure, retirementFailure);
            }
        }
        // Phase 2: initiate every cleanup. A generation with an accepted
        // handoff may transfer its cleanup to a dedicated continuation.
        for (WorkerEndpoint endpoint : snapshot.endpoints()) {
            try {
                endpoint.close();
            } catch (Throwable closeEndpointFailure) {
                failure = appendFailure(failure, closeEndpointFailure);
            }
        }
        // Phase 3: the registry is not closed until every exact generation has
        // completed cleanup and all retirement callbacks have returned.
        for (WorkerEndpoint endpoint : snapshot.endpoints()) {
            try {
                endpoint.awaitRetirement();
            } catch (Throwable awaitFailure) {
                failure = appendFailure(failure, awaitFailure);
            }
        }
        return failure;
    }

    private EndpointGenerationSnapshot snapshotEndpointGenerations() {
        IdentityHashMap<WorkerEndpoint, Boolean> seen =
                new IdentityHashMap<>();
        List<WorkerEndpoint> endpoints = new ArrayList<>();
        Throwable failure = null;
        failure = collectEndpointGenerations(
                prefillEndpoints, seen, endpoints, failure);
        failure = collectEndpointGenerations(
                decodeEndpoints, seen, endpoints, failure);
        failure = collectEndpointGenerations(
                pdFusionEndpoints, seen, endpoints, failure);
        failure = collectEndpointGenerations(
                vitEndpoints, seen, endpoints, failure);
        return new EndpointGenerationSnapshot(
                endpoints, failure);
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

    private record EndpointGenerationSnapshot(
            List<WorkerEndpoint> endpoints,
            Throwable failure) {
    }

    /** Immutable point-in-time view; publication remains owned by this registry. */
    public Map<String, PrefillEndpoint> snapshotPrefillEndpoints() {
        return Map.copyOf(prefillEndpoints);
    }

    /** Immutable point-in-time view; publication remains owned by this registry. */
    public Map<String, DecodeEndpoint> snapshotDecodeEndpoints() {
        return Map.copyOf(decodeEndpoints);
    }

    public int getEndpointCount(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints.size();
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints.size();
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints.size();
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints.size();
        }
        return 0;
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    public void evictExpiredOrphans(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        prefillEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.PREFILL, endpoint,
                        ep.evictExpiredInflight(ttlMs, schedulerOwnsRequest), ttlMs));
        decodeEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.DECODE, endpoint,
                        ep.evictExpiredRequests(ttlMs, schedulerOwnsRequest), ttlMs));
        pdFusionEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.PDFUSION, endpoint,
                        ep.evictExpiredBatches(ttlMs), ttlMs));
    }

    /**
     * Log and report one endpoint-ledger TTL eviction pass: endpoint-side
     * evictions were previously log-only, invisible to the
     * inflight.ttl.expired.qps series family. On this architecture the
     * endpoint ledgers have a single stale-unobserved exit, so every evicted
     * entry reports the {@code ttl} reason bucket; only non-zero counts are
     * reported, keeping the series sparse.
     */
    private void logEndpointEviction(RoleType role,
                                     String endpoint,
                                     int evicted,
                                     long ttlMs) {
        if (evicted > 0) {
            reporter.reportEndpointInflightTtlExpired(
                    role.name(), endpoint, "ttl", evicted);
            Logger.info("event=endpoint_inflight_ttl_eviction role={} endpoint={} "
                            + "evicted={} ttl_ms={}",
                    role, endpoint, evicted, ttlMs);
        }
    }

}
