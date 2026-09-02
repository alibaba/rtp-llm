package org.flexlb.dispatcher;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Holds the dispatcher's view of the FE pool and keeps it in sync with
 * {@link ServiceDiscovery}. {@link FePool} reads through {@link #source()} on every
 * {@code next()}.
 *
 * <p>Two refresh paths feed the same {@link AtomicReference}, mirroring sync's
 * {@code EngineAddressNameResolver}:
 * <ul>
 *   <li><b>Listener</b> — {@code serviceDiscovery.listen()} registers a callback so a
 *       push from the discovery client lands instantly. Real-time when push is available.</li>
 *   <li><b>Poll</b> — {@code @Scheduled} re-reads via {@code getHosts()} every 30 s on the
 *       shared Spring {@code task-scheduler} pool. Guarantees freshness even when push is
 *       unavailable, dropped, or registered against an implementation that silently no-ops.</li>
 * </ul>
 * Both paths route through {@link #applyUrls} so they share the change-detection / WARN
 * logic, and both are racey against each other in the same way master accepts: the loser of
 * a concurrent {@code getAndUpdate} is overwritten on the next event, and the URL list itself
 * is an idempotent snapshot so transient ordering does not corrupt downstream state.
 *
 * <p>Constructed only when the dispatcher is enabled — Spring's {@code @ConditionalOnProperty}
 * gates this {@code @Component} on {@code dispatch.fe-pool-service-id} being a non-blank value,
 * so disabled deployments do not pay for an idle scheduler or a parked listener.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatcherFePoolRefresher {

    private static final AtomicInteger DISCOVERY_THREAD_ID = new AtomicInteger();

    /**
     * Bound on a synchronous discovery lookup. Both the inline boot seed and the periodic poll run
     * on shared pools (Spring bean construction, then the shared {@code task-scheduler} that also
     * carries ZK-election heartbeats), so a hung discovery client must degrade to a timeout rather
     * than park a shared thread — the same reason {@code WorkerAddressService} bounds its calls.
     */
    private static final long DISCOVERY_TIMEOUT_MS = 3_000;

    /** Hard cap on discovery workers that may remain wedged after ignoring interruption. */
    private static final int MAX_RETIRED_DISCOVERY_EXECUTORS = 2;

    /**
     * Fallback for the empty-discovery grace window when the configured value is absent or
     * non-positive. Matches {@code FlexlbConfig#discoveryFailureGraceMs}'s own default so an
     * unconfigured deployment behaves identically on both sides.
     */
    private static final long DEFAULT_EMPTY_DISCOVERY_GRACE_NANOS = TimeUnit.MINUTES.toNanos(5);

    /**
     * How long a run of empty discovery results may keep displacing the last non-empty pool.
     * Within the window an empty snapshot is treated as suspect (a discovery hiccup, a client
     * that swallows failures into an empty list) and the known FEs are kept; past it the empty
     * result is accepted as the truth — the fleet scaled to zero or was torn down — and
     * {@link FePool#next()} fails fast instead of timing out against removed hosts forever.
     * Sourced from the same {@code FlexlbConfig#discoveryFailureGraceMs} the sync side reads
     * (env override {@code DISCOVERY_FAILURE_GRACE_MS}), so one knob governs both discovery
     * consumers; {@link #DEFAULT_EMPTY_DISCOVERY_GRACE_NANOS} when unset or non-positive.
     */
    private final long emptyDiscoveryGraceNanos;

    private final ServiceDiscovery serviceDiscovery;
    private final String serviceId;
    private final LongSupplier nanoClock;
    private final long discoveryTimeoutMs;
    private final AtomicReference<List<String>> fePoolUrls = new AtomicReference<>(List.of());
    private final RateLimitedWarn emptyDiscoveryWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    /**
     * Monotonic instant of the last snapshot that proved the fleet non-empty. The grace clock
     * for accepting an empty snapshot counts from here, so intermittent non-empty results keep
     * resetting it and only a sustained run of emptiness drains the pool.
     */
    private volatile long lastNonEmptyNanos;

    /**
     * Dedicated single-thread executor for the bounded discovery lookup. A wedged discovery client
     * must not park a {@code ForkJoinPool.commonPool()} thread (shared JVM-wide) nor the master's
     * sync pool — its own daemon thread absorbs the stall and is cancelled on timeout.
     */
    private final AtomicReference<ExecutorService> discoveryExecutor =
            new AtomicReference<>(newDiscoveryExecutor());
    private final Object executorLock = new Object();
    private final List<ExecutorService> retiredExecutors = new ArrayList<>();

    @Autowired
    public DispatcherFePoolRefresher(ServiceDiscovery serviceDiscovery, DispatchConfig cfg,
                                     ConfigService configService) {
        this(serviceDiscovery, cfg,
                configService.loadBalanceConfig().getDiscoveryFailureGraceMs(), System::nanoTime,
                DISCOVERY_TIMEOUT_MS);
    }

    /**
     * Visible for tests: {@code emptyDiscoveryGraceMs} lets a small grace be configured directly
     * and {@code nanoClock} lets grace-window expiry be simulated without sleeping.
     */
    DispatcherFePoolRefresher(ServiceDiscovery serviceDiscovery, DispatchConfig cfg,
                              long emptyDiscoveryGraceMs, LongSupplier nanoClock) {
        this(serviceDiscovery, cfg, emptyDiscoveryGraceMs, nanoClock, DISCOVERY_TIMEOUT_MS);
    }

    /** Test seam for exercising timeout recovery without waiting the production three seconds. */
    DispatcherFePoolRefresher(ServiceDiscovery serviceDiscovery, DispatchConfig cfg,
                              long emptyDiscoveryGraceMs, LongSupplier nanoClock,
                              long discoveryTimeoutMs) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceId = cfg.getFePoolServiceId();
        this.emptyDiscoveryGraceNanos = resolveGraceNanos(emptyDiscoveryGraceMs);
        this.nanoClock = nanoClock;
        this.discoveryTimeoutMs = discoveryTimeoutMs;
        this.lastNonEmptyNanos = nanoClock.getAsLong();
        // Boot seed first — guarantees source() returns the freshest available view by the
        // time downstream beans (FePool, FeHealthChecker) read it during their own init.
        // A discovery hiccup here degrades to an empty snapshot that the listener and poll
        // paths repair; it must not take the whole application down with it.
        try {
            applyUrls(toUrls(boundedGetHosts()), "boot");
        } catch (Exception e) {
            Logger.warn("dispatcher FE pool boot seed failed (listener/poll will retry): serviceId={}, err={}",
                    serviceId, DispatcherResponses.briefReason(e));
        }
        // The discovery client drives this callback from its own push thread; an exception
        // escaping here (or out of listen() itself) could kill that thread and silence future
        // pushes. Swallow and lean on the 30s poll to repair.
        try {
            serviceDiscovery.listen(serviceId, hosts -> {
                try {
                    applyUrls(toUrls(hosts), "listener");
                } catch (Exception e) {
                    Logger.warn("dispatcher FE pool listener callback failed (poll will repair): "
                            + "serviceId={}, err={}", serviceId, DispatcherResponses.briefReason(e));
                }
            });
        } catch (Exception e) {
            Logger.warn("dispatcher FE pool listen() registration failed (poll will repair): "
                    + "serviceId={}, err={}", serviceId, DispatcherResponses.briefReason(e));
        }
    }

    /**
     * A non-positive configured grace is a misconfiguration, not a request to drain instantly —
     * that would turn every discovery hiccup into a full FE-pool drop. Fall back to the 5-minute
     * default the sync side also documents.
     */
    static long resolveGraceNanos(long graceMs) {
        return graceMs > 0 ? TimeUnit.MILLISECONDS.toNanos(graceMs) : DEFAULT_EMPTY_DISCOVERY_GRACE_NANOS;
    }

    /**
     * URL supplier passed into {@link FePool}. Returns a snapshot on every call; the
     * reference is updated atomically by {@link #applyUrls}.
     */
    public Supplier<List<String>> source() {
        return fePoolUrls::get;
    }

    /** Visible for tests. Number of URLs in the current snapshot. */
    public int currentSize() {
        return fePoolUrls.get().size();
    }

    /** Visible for tests. The resolved grace window in nanos, as this instance will apply it. */
    long graceNanos() {
        return emptyDiscoveryGraceNanos;
    }

    /**
     * Periodic poll. Re-reads the FE pool from {@link ServiceDiscovery} on the shared
     * Spring {@code task-scheduler} pool, matching the cadence sync's
     * {@code EngineAddressNameResolver} uses. Acts as a tolerance layer for any deployment
     * where the listener path is silent (e.g. {@code NoOpServiceDiscovery}'s one-shot
     * semantics, or a {@code listen()} implementation that dedupes late subscribers).
     */
    @Scheduled(fixedDelay = 30_000, initialDelay = 5_000)
    public void refresh() {
        try {
            applyUrls(toUrls(boundedGetHosts()), "poll");
        } catch (Exception e) {
            Logger.warn("dispatcher FE pool refresh failed: serviceId={}, err={}",
                    serviceId, DispatcherResponses.briefReason(e));
        }
    }

    /**
     * Replace the snapshot and WARN if (and only if) the URL set actually changed. Both
     * the listener callback and the {@code @Scheduled} poll route through here so the
     * "changed?" check stays in one place. Steady state — same list arriving twice — emits
     * nothing, so the log does not get flooded by no-op refreshes.
     *
     * <p>An empty snapshot displaces a non-empty pool only after it has persisted for
     * {@link #emptyDiscoveryGraceNanos}. Within the window it is treated as suspect — a
     * discovery hiccup, or a client that swallows a failed lookup into an empty list — and the
     * known FEs are kept ({@link FeHealthChecker} probes them directly, so hosts that are
     * genuinely gone leave rotation via the probe). Past the window the empty result is the
     * truth: the fleet scaled to zero, and holding the old list forever would keep
     * {@link FePool#next()}'s all-dead fallback shoveling traffic at removed hosts. Accepting
     * it empties the pool so {@code next()} fails fast until discovery reports hosts again.
     * (Outright lookup failures never reach here as an empty list — {@code boundedGetHosts}
     * throws and the callers keep the pool untouched.)
     */
    private void applyUrls(List<String> urls, String source) {
        if (!urls.isEmpty()) {
            lastNonEmptyNanos = nanoClock.getAsLong();
            List<String> prev = fePoolUrls.getAndSet(urls);
            if (!prev.equals(urls)) {
                Logger.warn("dispatcher FE pool updated ({}): serviceId={}, hosts={} (was {})",
                        source, serviceId, urls.size(), prev.size());
            }
            return;
        }
        List<String> prev = fePoolUrls.get();
        if (prev.isEmpty()) {
            return;
        }
        long emptyForNanos = nanoClock.getAsLong() - lastNonEmptyNanos;
        if (emptyForNanos <= emptyDiscoveryGraceNanos) {
            emptyDiscoveryWarn.warn("dispatcher FE pool: ignoring empty discovery result ({}) within "
                    + "grace, keeping {} known FE(s): serviceId={}", source, prev.size(), serviceId);
            return;
        }
        // CAS so a concurrent non-empty update wins over this (by definition stale) drain.
        if (fePoolUrls.compareAndSet(prev, List.of())) {
            Logger.warn("dispatcher FE pool: empty discovery persisted beyond grace ({}s), accepting "
                            + "it — dropping {} FE(s) and failing fast until discovery reports hosts again: "
                            + "serviceId={}, source={}",
                    TimeUnit.NANOSECONDS.toSeconds(emptyDiscoveryGraceNanos), prev.size(), serviceId, source);
        }
    }

    /**
     * Resolve the FE pool with a bound. Both callers run on shared pools, so a wedged discovery
     * client must not park a thread indefinitely — cap the wait and surface a timeout, which the
     * caller degrades to a warn (the retained snapshot is kept by {@link #applyUrls}).
     */
    private List<WorkerHost> boundedGetHosts() throws Exception {
        ExecutorService executor = acquireDiscoveryExecutor();
        Future<List<WorkerHost>> future = executor.submit(() -> serviceDiscovery.getHosts(serviceId));
        try {
            return future.get(discoveryTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            retireTimedOutExecutor(executor);
            throw e;
        }
    }

    private ExecutorService acquireDiscoveryExecutor() {
        synchronized (executorLock) {
            reapRetiredExecutors();
            ExecutorService current = discoveryExecutor.get();
            if (current != null) {
                return current;
            }
            if (retiredExecutors.size() >= MAX_RETIRED_DISCOVERY_EXECUTORS) {
                throw new RejectedExecutionException(
                        "discovery lookup circuit open: too many uninterruptible calls");
            }
            ExecutorService replacement = newDiscoveryExecutor();
            discoveryExecutor.set(replacement);
            return replacement;
        }
    }

    /**
     * Interrupting a native/socket-backed lookup is advisory. Retire the executor after a timeout
     * so the next poll can run instead of queueing forever behind the wedged call.
     */
    private void retireTimedOutExecutor(ExecutorService timedOut) {
        synchronized (executorLock) {
            if (!discoveryExecutor.compareAndSet(timedOut, null)) {
                return;
            }
            timedOut.shutdownNow();
            retiredExecutors.add(timedOut);
            reapRetiredExecutors();
            if (retiredExecutors.size() < MAX_RETIRED_DISCOVERY_EXECUTORS) {
                discoveryExecutor.set(newDiscoveryExecutor());
            }
        }
    }

    private void reapRetiredExecutors() {
        retiredExecutors.removeIf(ExecutorService::isTerminated);
    }

    private static ExecutorService newDiscoveryExecutor() {
        return Executors.newSingleThreadExecutor(r -> {
            Thread thread = new Thread(r,
                    "dispatcher-fe-discovery-" + DISCOVERY_THREAD_ID.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        });
    }

    @PreDestroy
    void shutdown() {
        synchronized (executorLock) {
            ExecutorService current = discoveryExecutor.getAndSet(null);
            if (current != null) {
                current.shutdownNow();
            }
            retiredExecutors.forEach(ExecutorService::shutdownNow);
            retiredExecutors.clear();
        }
    }

    private static List<String> toUrls(List<WorkerHost> hosts) {
        // Unmodifiable: FePool.livePool() may hand this very list back to callers by reference (the
        // all-alive fast path, no copy), and the reference is shared across threads via the
        // AtomicReference. Making it immutable turns "published, never mutated in place" from a
        // convention into a type guarantee, so a future FePool consumer cannot corrupt the snapshot.
        return hosts.stream().map(h -> "http://" + h.getIpPort()).collect(Collectors.toUnmodifiableList());
    }
}
