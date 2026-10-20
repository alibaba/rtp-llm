package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.LongPredicate;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** Owns discovered worker identities and their published routing generations. */
@Component
public class EndpointRegistry {

    /**
     * One non-owning Prefill generation exposed to routing.
     *
     * <p>The endpoint identity is advisory: a caller must capture the address
     * and verify that the pinned endpoint is still this exact instance before
     * transferring generation ownership to a request.</p>
     */
    public record PrefillRoutingEntry(
            String address,
            PrefillEndpoint endpoint) {
        public PrefillRoutingEntry {
            java.util.Objects.requireNonNull(address, "address");
            java.util.Objects.requireNonNull(endpoint, "endpoint");
        }
    }

    /** Owns one retiring status identity and its optional published endpoint. */
    public final class Retirement {
        private final RoleType role;
        private final String address;
        private final WorkerStatus status;
        private final WorkerEndpoint endpoint;
        private final AtomicBoolean claimed = new AtomicBoolean();

        private Retirement(RoleType role, String address, WorkerStatus status, WorkerEndpoint endpoint) {
            this.role = role;
            this.address = address;
            this.status = status;
            this.endpoint = endpoint;
        }

        public WorkerStatus status() {
            return status;
        }

        /** Drain outside the generation lock, then clear cache and remove this exact status. */
        public void complete(CacheAwareService cacheAwareService, org.slf4j.Logger logger) {
            Objects.requireNonNull(cacheAwareService, "cacheAwareService");
            Objects.requireNonNull(logger, "logger");
            if (!claimed.compareAndSet(false, true)) {
                throw new IllegalStateException("Worker retirement was already claimed: "
                        + address + "#" + status.getGenerationId());
            }
            Runnable finalizeExact = () -> finalizeExact(cacheAwareService, logger);
            if (endpoint == null) {
                finalizeExact.run();
                return;
            }
            // The continuation owns the fence from this point onward. A bounded
            // wait failure cannot discard a still-draining generation.
            endpoint.onRetired(finalizeExact);
            Throwable failure = Failures.run(null, () -> placementAvailability.changed(
                    role, status.topologySnapshot().group(), address));
            failure = Failures.run(failure, endpoint::close);
            failure = Failures.run(failure, () -> endpoint.awaitRetirement(retirementTimeoutMs));
            if (failure != null) {
                logger.error("Endpoint cleanup failed after retiring generation {} for {}",
                        status.getGenerationId(), address, failure);
            }
        }

        private void finalizeExact(CacheAwareService cacheAwareService, org.slf4j.Logger logger) {
            if (finalizeRetirement(role, address, status, cacheAwareService, logger)) {
                resolveDetachedGeneration();
            } else if (closeCompletion == null) {
                CompletableFuture.delayedExecutor(1L, TimeUnit.SECONDS)
                        .execute(() -> finalizeExact(cacheAwareService, logger));
            }
        }
    }

    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerStatus>> statusesByRole =
            roleMaps(RoleType.values());

    private final EnumMap<RoleType, ConcurrentHashMap<String, WorkerEndpoint>>
            endpointsByRole = roleMaps(
                    RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION, RoleType.VIT);
    /** Advisory Prefill generations, atomically replaced after each map write. */
    private volatile List<PrefillRoutingEntry> prefillDirectory = List.of();
    /** PDFusion uses the same Prefill planning path but a distinct role map. */
    private volatile List<PrefillRoutingEntry> pdFusionDirectory = List.of();
    /**
     * Advisory Decode routing directory. Writers publish a complete immutable
     * replacement after changing the endpoint map. A racing reader may finish
     * an older traversal, but it cannot acquire stale ownership because the
     * selected address and generation are always revalidated by
     * {@link #captureDecodeGeneration(DecodeResources.DecodeRoutingView)}.
     */
    private volatile List<Map.Entry<String, DecodeEndpoint>> decodeDirectory =
            List.of();
    private final ConfigService configService;
    private final RequestRepository requests;
    private final DeliveryMetricsReporter reporter;
    private final DeliveryDispatcher dispatcher;
    private final PlacementAvailability placementAvailability;
    private final Object lifecycleGate = new Object();
    /** Null while open; every closing caller observes the same completed cleanup result. */
    private volatile CompletableFuture<Throwable> closeCompletion;
    private int inflightPublications;
    private int inflightDetachedRetirements;
    private final long retirementTimeoutMs = Long.getLong("flexlb.endpoint.retirement.timeout.ms", 30_000L);

    private static <T> EnumMap<RoleType, ConcurrentHashMap<String, T>> roleMaps(RoleType... roles) {
        EnumMap<RoleType, ConcurrentHashMap<String, T>> maps =
                new EnumMap<>(RoleType.class);
        for (RoleType role : roles) {
            maps.put(role, new ConcurrentHashMap<>());
        }
        return maps;
    }

    @Autowired
    public EndpointRegistry(ConfigService configService,
                            RequestRepository requests,
                            DeliveryMetricsReporter reporter,
                            DeliveryDispatcher dispatcher,
                            PlacementAvailability placementAvailability) {
        checkArgument(retirementTimeoutMs > 0L, "endpoint retirement timeout must be positive");
        this.configService = java.util.Objects.requireNonNull(
                configService, "configService");
        this.requests = java.util.Objects.requireNonNull(
                requests, "requests");
        this.reporter = java.util.Objects.requireNonNull(reporter, "reporter");
        this.dispatcher = java.util.Objects.requireNonNull(
                dispatcher, "dispatcher");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(roleType);
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
                endpointsByRole.get(roleType);
        return endpoints == null ? null : capture(endpoints, ipPort);
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
        if (roleType != null && roleType.supportsPrefill()) {
            List<PrefillRoutingEntry> directory = prefillRoutingSnapshot(roleType);
            return new AbstractList<>() {
                @Override
                public String get(int index) {
                    return directory.get(index).address();
                }

                @Override
                public int size() {
                    return directory.size();
                }
            };
        }
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(roleType);
        return endpoints == null
                ? List.of() : List.copyOf(endpoints.keySet());
    }

    /**
     * Return the immutable Prefill routing directory observed at one instant.
     *
     * <p>This removes per-request registry lookups while deliberately avoiding
     * fleet-wide generation pins. A selected entry remains advisory until its
     * address is captured and the endpoint identity is revalidated.</p>
     */
    public List<PrefillRoutingEntry> prefillRoutingSnapshot(RoleType roleType) {
        return switch (roleType) {
            case PREFILL -> prefillDirectory;
            case PDFUSION -> pdFusionDirectory;
            default -> List.of();
        };
    }

    /**
     * Append immutable Decode routing values to caller-owned storage, which
     * the registry never retains. Values reflect generations seen during this traversal.
     *
     * <p>The registry is intentionally not locked while an endpoint takes its
     * admission lock.  This avoids introducing a map-bin/admission-lock order;
     * publication races are resolved when the selected generation is pinned
     * by {@link #captureDecodeGeneration(DecodeResources.DecodeRoutingView)}.</p>
     */
    public void decodeRoutingSnapshot(String group, List<DecodeResources.DecodeRoutingView> target) {
        for (Map.Entry<String, DecodeEndpoint> entry : decodeDirectory) {
            var view = Objects.requireNonNull(entry.getValue().routingView(entry.getKey()));
            if (group == null || group.equals(view.topology().group())) { target.add(view); }
        }
    }

    /**
     * Capture the exact Decode generation represented by a routing snapshot.
     *
     * <p>The routing values are advisory and may change after selection. The
     * subsequent reservation/dispatch transaction revalidates capacity under
     * the endpoint admission lock. This method authorizes only an identical
     * endpoint generation; every rejected or exceptional capture is closed
     * here, while a successful caller owns the returned generation pin.</p>
     */
    public WorkerEndpoint.GenerationPin captureDecodeGeneration(
            DecodeResources.DecodeRoutingView expected) {
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
     * cannot own any published RequestContext identity, so there are no request
     * transitions to apply before publication.
     *
     * <p>A factory, endpoint-reducer, or status-commit failure closes the
     * candidate before routing publication. If final map publication fails
     * after commit, the caller withdraws the entire WorkerStatus generation;
     * it is never restored.</p>
     */
    public WorkerEndpoint publishPreparedEndpoint(
            String address,
            WorkerStatus status,
            WorkerStatus.PreparedStatus prepared) {
        beginCandidatePublication();
        WorkerEndpoint candidate = null;
        boolean publicationAttempted = false;
        Throwable failure = null;
        try {
            status.requireGenerationLock();
            status.requireActiveGeneration();
            WorkerStatus.StatusObservation observation = prepared.observation();
            checkArgument(observation.owner() == status, "staged status belongs to another worker generation");
            RoleType role = observation.role();
            checkArgument(role == status.getRole(), "staged status role does not match its worker generation");
            java.util.Objects.requireNonNull(address, "address");
            if (status.appliedStatusCursor().statusVersion() >= 0L) {
                throw new IllegalStateException(
                        "A committed WorkerStatus generation cannot publish a second endpoint: "
                                + address + "#" + status.getGenerationId());
            }
            candidate = createEndpoint(status, role, observation.engine());
            candidate.initializeFromPreparedStatus(status, observation);
            status.publishPreparedStatus(prepared);
            WorkerEndpoint exact = candidate;
            publicationAttempted = true;
            mutateEndpointMap(role, endpointsByRole.get(role), address, (ignored, current) -> {
                checkState(current == null || current.getStatus() != status,
                        "Endpoint generation is already published for %s", address);
                checkState(current == null,
                        "Existing endpoint generation must be withdrawn before publication for %s", address);
                return exact;
            });
            WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
            placementAvailability.changed(role, topology.group(), address);
            return candidate;
        } catch (Throwable publicationFailure) {
            failure = publicationFailure;
            throw Failures.propagate(failure, "Endpoint publication failed for " + address);
        } finally {
            try {
                if (failure != null && candidate != null) {
                    WorkerEndpoint exact = candidate;
                    try {
                        if (publicationAttempted) {
                            RoleType role = status.getRole();
                            mutateEndpointMap(role, endpointsByRole.get(role), address,
                                    (ignored, current) -> current == exact ? null : current);
                        }
                    } catch (Throwable withdrawalFailure) {
                        Failures.append(failure, withdrawalFailure);
                    } finally {
                        Failures.run(failure, exact::closeAsynchronously);
                    }
                }
            } finally {
                endCandidatePublication();
            }
        }
    }

    private void beginCandidatePublication() {
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            inflightPublications++;
        }
    }

    private void endCandidatePublication() {
        synchronized (lifecycleGate) {
            checkState(inflightPublications > 0, "EndpointRegistry publication count underflow");
            inflightPublications--;
            if (inflightPublications == 0) {
                lifecycleGate.notifyAll();
            }
        }
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
        boolean updatesPrefillDirectory = role != null && role.supportsPrefill();
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
            List<PrefillRoutingEntry> nextPrefillDirectory = updatesPrefillDirectory
                    ? directoryAfterMutation(
                            role == RoleType.PREFILL ? prefillDirectory : pdFusionDirectory,
                            address, PrefillRoutingEntry::address,
                            next == null ? null : new PrefillRoutingEntry(address, (PrefillEndpoint) next))
                    : null;
            List<Map.Entry<String, DecodeEndpoint>> nextDecodeDirectory = updatesDecodeDirectory
                    ? directoryAfterMutation(decodeDirectory, address, Map.Entry::getKey,
                            next == null ? null : Map.entry(address, (DecodeEndpoint) next))
                    : null;
            WorkerEndpoint published = endpoints.compute(
                    address, (ignored, observed) -> {
                checkState(observed == exactCurrent,
                        "%s endpoint mapping changed outside its directory transaction: %s", role, address);
                return next;
            });
            if (updatesPrefillDirectory) {
                if (role == RoleType.PREFILL) {
                    prefillDirectory = nextPrefillDirectory;
                } else {
                    pdFusionDirectory = nextPrefillDirectory;
                }
            } else {
                this.decodeDirectory = nextDecodeDirectory;
            }
            return published;
        }
    }

    /** Keep the immutable directory in address order; request routing then needs no full sort. */
    private <T> List<T> directoryAfterMutation(List<T> previous, String address,
                                              Function<T, String> addressOf, T next) {
        if (closeCompletion != null) {
            return List.of();
        }
        List<T> updated = new ArrayList<>(previous.size() + (next == null ? 0 : 1));
        boolean inserted = next == null;
        for (T entry : previous) {
            String currentAddress = addressOf.apply(entry);
            if (!inserted && address.compareTo(currentAddress) <= 0) {
                updated.add(next);
                inserted = true;
            }
            if (!address.equals(currentAddress)) {
                updated.add(entry);
            }
        }
        if (!inserted) {
            updated.add(next);
        }
        return List.copyOf(updated);
    }

    /** Revalidate the exact generation and close its routing gate only after its heartbeat expires. */
    public Retirement beginRetirementIfStale(RoleType role, String address, WorkerStatus status, long staleAfterUs) {
        status.lock.lock();
        try {
            if (!isCurrentStatus(role, address, status) || !status.isActiveGeneration()
                    || TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - status.pollHealth().lastSuccessfulPollUs()
                    <= staleAfterUs) {
                return null;
            }
            return beginRetirement(role, address, status);
        } finally {
            status.lock.unlock();
        }
    }

    /**
     * Close and remove one exact endpoint generation from routing without
     * waiting for its potentially blocking drain. The caller holds the ACTIVE
     * WorkerStatus generation lock. This method publishes RETIRING only after
     * the exact endpoint gate has closed and routing withdrawal is visible;
     * the caller completes the returned retirement outside that lock. A status
     * without a published endpoint still returns a finalization capability.
     */
    public Retirement beginRetirement(
            RoleType roleType, String ipPort, WorkerStatus expectedStatus) {
        Objects.requireNonNull(expectedStatus, "status");
        Objects.requireNonNull(roleType, "role");
        Objects.requireNonNull(ipPort, "address");
        expectedStatus.requireActiveGeneration();
        Retirement retirement;
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            WorkerEndpoint expected = get(roleType, ipPort, expectedStatus);
            retirement = new Retirement(roleType, ipPort, expectedStatus, expected);
            if (expected != null) {
                mutateEndpointMap(roleType, endpointsByRole.get(roleType), ipPort, (ignored, current) -> {
                    checkState(current == expected, "Endpoint changed while its generation lock was held: %s", ipPort);
                    // Close admission while the exact mapping is still visible.
                    current.beginRetirement();
                    return null;
                });
            }
            if (!expectedStatus.beginRetirementAfterEndpointGateClosed()) {
                throw new IllegalStateException("WorkerStatus generation changed while its lock was held: "
                        + ipPort + "#" + expectedStatus.getGenerationId());
            }
            inflightDetachedRetirements++;
        }
        return retirement;
    }

    /** Resolve the barrier after the exact generation's drain and finalization. */
    private void resolveDetachedGeneration() {
        synchronized (lifecycleGate) {
            checkState(inflightDetachedRetirements > 0, "Detached retirement count underflow");
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
        checkArgument(role != RoleType.FRONTEND, "Unsupported role: %s", role);
        boolean prefill = role == RoleType.PREFILL
                || role == RoleType.PDFUSION;
        if (prefill && engineStatus.dpSize() > 1) {
            throw new UnsupportedOperationException(
                    role + " DP group endpoint not yet supported: ipPort="
                            + status.getIpPort() + ", dp_size="
                            + engineStatus.dpSize());
        }
        try {
            reporter.prepareEndpointMetrics(role.name(), status.getIp());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("Endpoint metric preparation failed: role={}, engine={}",
                    role, status.getIp(), telemetryFailure);
        }
        return switch (role) {
            case PREFILL, PDFUSION -> new PrefillEndpoint(
                    status,
                    configService.loadBalanceConfig(),
                    dispatcher,
                    reporter,
                    placementAvailability);
            case DECODE -> new DecodeEndpoint(
                    status, requests, placementAvailability);
            case VIT -> new WorkerEndpoint(status);
            case FRONTEND -> throw new AssertionError("validated above");
        };
    }

    public void close() {
        boolean closeOwner;
        CompletableFuture<Throwable> completion;
        synchronized (lifecycleGate) {
            closeOwner = closeCompletion == null;
            if (closeOwner) {
                closeCompletion = new CompletableFuture<>();
                prefillDirectory = List.of();
                pdFusionDirectory = List.of();
                decodeDirectory = List.of();
            }
            completion = closeCompletion;
        }
        if (!closeOwner) {
            // Join outside the gate; concurrent closers retain their interruption status.
            Failures.rethrow(completion.join(), "EndpointRegistry close failed");
            return;
        }

        boolean interrupted = false;
        synchronized (lifecycleGate) {
            while (inflightPublications != 0) {
                try {
                    lifecycleGate.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
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
            failure = Failures.append(failure, detachedBarrierFailure);
        } finally {
            completion.complete(failure);
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        Failures.rethrow(failure, "EndpointRegistry close failed");
    }

    /** Wait for every generation detached before the close gate linearized. */
    private void awaitDetachedRetirements() {
        boolean interrupted = false;
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(retirementTimeoutMs);
        try {
            synchronized (lifecycleGate) {
                while (inflightDetachedRetirements != 0) {
                    long remaining = deadlineNanos - System.nanoTime();
                    checkState(remaining > 0L, "Detached worker retirement timed out after %sms: pending=%s",
                            retirementTimeoutMs, inflightDetachedRetirements);
                    try { TimeUnit.NANOSECONDS.timedWait(lifecycleGate, remaining); }
                    catch (InterruptedException interruption) { interrupted = true; }
                }
            }
        } finally {
            if (interrupted) { Thread.currentThread().interrupt(); }
        }
    }

    private Throwable closeEndpointGenerations() {
        Set<WorkerEndpoint> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<WorkerEndpoint> endpoints = new ArrayList<>();
        Throwable failure = null;
        for (Map<String, WorkerEndpoint> roleEndpoints
                : endpointsByRole.values()) {
            try {
                for (WorkerEndpoint endpoint : roleEndpoints.values()) {
                    if (seen.add(endpoint)) {
                        endpoints.add(endpoint);
                    }
                }
            } catch (RuntimeException | Error snapshotFailure) {
                failure = Failures.append(failure, snapshotFailure);
            }
        }
        // Phase 1: close admission for every exact generation before any
        // endpoint-local cleanup or callback can run.
        for (WorkerEndpoint endpoint : endpoints) {
            failure = Failures.run(failure, () -> {
                WorkerStatus status = endpoint.getStatus();
                status.lock.lock();
                try {
                    endpoint.beginRetirement();
                    status.beginRetirementAfterEndpointGateClosed();
                } finally { status.lock.unlock(); }
            });
        }
        // Discovered generations may have an in-flight cache poll before their
        // first endpoint is published. Fence these callbacks under the same lock.
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            for (WorkerStatus status : statuses.values()) {
                failure = Failures.run(failure, () -> {
                    status.lock.lock();
                    try { status.beginRetirementAfterEndpointGateClosed(); }
                    finally { status.lock.unlock(); }
                });
            }
        }
        // Phase 2: initiate every cleanup. A generation with an accepted
        // handoff may transfer its cleanup to a dedicated continuation.
        for (WorkerEndpoint endpoint : endpoints) {
            failure = Failures.run(failure, () -> {
                endpoint.close();
            });
        }
        // Phase 3 shares one endpoint-drain budget; detached finalizers have a
        // separate bounded barrier. A timeout reports failure without settling ownership.
        long deadlineNanos = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(retirementTimeoutMs);
        for (WorkerEndpoint endpoint : endpoints) {
            failure = Failures.run(failure, () -> endpoint.awaitRetirement(Math.max(1L,
                    TimeUnit.NANOSECONDS.toMillis(deadlineNanos - System.nanoTime()))));
        }
        return failure;
    }

    /** Immutable diagnostic view; callers know the role of each resource owner. */
    public Map<String, WorkerEndpoint> endpointSnapshot(RoleType role) {
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(role);
        return endpoints == null ? Map.of() : Map.copyOf(endpoints);
    }

    public record Counts(int discovered, int published) { }

    /** Cheap role counts; no endpoint or request snapshots are constructed. */
    public Counts counts(RoleType role) {
        Objects.requireNonNull(role, "role");
        Map<String, WorkerEndpoint> endpoints = endpointsByRole.get(role);
        return new Counts(statusesByRole.get(role).size(), endpoints == null ? 0 : endpoints.size());
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     * @param retainForSchedulerCleanup current directory lookup, called outside endpoint locks;
     *                                  candidates are revalidated under the ledger lock before eviction
     */
    public void evictExpiredOrphans(long ttlMs,
                                    LongPredicate retainForSchedulerCleanup) {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION)) {
            endpointsByRole.get(role).forEach((address, worker) -> {
                int evicted = role == RoleType.DECODE
                        ? ((DecodeEndpoint) worker).evictExpiredRequests(ttlMs, retainForSchedulerCleanup)
                        : ((PrefillEndpoint) worker).evictExpiredInflight(ttlMs, retainForSchedulerCleanup);
                if (evicted > 0) {
                    reporter.reportEndpointInflightTtlExpired(role.name(), address, "ttl", evicted);
                    Logger.info("event=endpoint_inflight_ttl_eviction role={} endpoint={} evicted={} ttl_ms={}",
                            role, address, evicted, ttlMs);
                }
            });
        }
    }

    /** Immutable point-in-time view of discovered generations for one role. */
    public Map<String, WorkerStatus> statusSnapshot(RoleType role) {
        if (role == null) {
            return Map.of();
        }
        return Map.copyOf(statusesByRole.get(role));
    }

    /** Accept updates only for an exact current generation while this registry is open. */
    public boolean isCurrentStatus(
            RoleType role, String address, WorkerStatus expected) {
        return closeCompletion == null && expected != null && role != null && address != null
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
        checkState(closeCompletion == null, "EndpointRegistry is closing");
        ConcurrentHashMap<String, WorkerStatus> statuses = statusesByRole.get(role);
        WorkerStatus current = statuses.get(address);
        if (current != null) { return current; }
        // Construct private discovery state outside the gate: factories must not
        // introduce a lifecycle-gate -> generation-lock edge.
        WorkerStatus discovered = Objects.requireNonNull(discoveredFactory.get(), "discovered status");
        checkArgument(discovered.getRole() == role && address.equals(discovered.getIpPort()),
                "Discovered WorkerStatus identity does not match directory key");
        synchronized (lifecycleGate) {
            checkState(closeCompletion == null, "EndpointRegistry is closing");
            WorkerStatus raced = statuses.putIfAbsent(address, discovered);
            return raced == null ? discovered : raced;
        }
    }

    public int discoveredCount() {
        int total = 0;
        for (Map<String, WorkerStatus> statuses : statusesByRole.values()) {
            total += statuses.size();
        }
        return total;
    }

    private boolean finalizeRetirement(
            RoleType role,
            String address,
            WorkerStatus status,
            CacheAwareService cacheAwareService,
            org.slf4j.Logger logger) {
        Map<String, WorkerStatus> statuses = statusesByRole.get(role);
        status.lock.lock();
        try {
            status.requireRetiringGeneration();
            if (statuses.get(address) != status) {
                logger.error(
                        "Status identity changed before retirement finalized for {}#{}",
                        address, status.getGenerationId());
                return true;
            }
            try {
                if (role.requiresCacheKeys()) {
                    cacheAwareService.removeEngineBlockCache(address);
                }
            } catch (Throwable cacheCleanupFailure) {
                logger.error(
                        "Cache cleanup failed while retiring generation {} for {}",
                        status.getGenerationId(), address, cacheCleanupFailure);
                return false;
            }
            if (!statuses.remove(address, status)) {
                logger.error(
                        "Exact status removal failed while its generation lock was held for {}#{}",
                        address, status.getGenerationId());
            }
            return true;
        } catch (Throwable finalizationFailure) {
            logger.error(
                    "Status retirement finalization failed for {}#{}",
                    address, status.getGenerationId(), finalizationFailure);
            return false;
        } finally {
            status.lock.unlock();
        }
    }

}
