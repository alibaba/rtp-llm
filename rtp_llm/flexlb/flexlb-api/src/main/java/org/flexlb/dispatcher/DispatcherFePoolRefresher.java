package org.flexlb.dispatcher;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.util.Logger;
import org.flexlb.util.RateLimitedWarn;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
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

    /**
     * Bound on a synchronous discovery lookup. Both the inline boot seed and the periodic poll run
     * on shared pools (Spring bean construction, then the shared {@code task-scheduler} that also
     * carries ZK-election heartbeats), so a hung discovery client must degrade to a timeout rather
     * than park a shared thread — the same reason {@code WorkerAddressService} bounds its calls.
     */
    private static final long DISCOVERY_TIMEOUT_MS = 3_000;

    private final ServiceDiscovery serviceDiscovery;
    private final String serviceId;
    private final AtomicReference<List<String>> fePoolUrls = new AtomicReference<>(List.of());
    private final RateLimitedWarn emptyDiscoveryWarn = new RateLimitedWarn(1, TimeUnit.SECONDS);

    /**
     * Dedicated single-thread executor for the bounded discovery lookup. A wedged discovery client
     * must not park a {@code ForkJoinPool.commonPool()} thread (shared JVM-wide) nor the master's
     * sync pool — its own daemon thread absorbs the stall and is cancelled on timeout.
     */
    private final ExecutorService discoveryExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "dispatcher-fe-discovery");
        t.setDaemon(true);
        return t;
    });

    public DispatcherFePoolRefresher(ServiceDiscovery serviceDiscovery, DispatchConfig cfg) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceId = cfg.getFePoolServiceId();
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
     * <p>An empty snapshot never displaces a non-empty one. A discovery client that swallows a
     * failed lookup reports it as an empty list, indistinguishable from a fleet that scaled to
     * zero, and accepting it would drop every FE at once and fail 100% of batch traffic. Liveness
     * is not discovery's job here — {@link FeHealthChecker} probes the known FEs directly, so a
     * fleet that genuinely went away is taken out of rotation by the probe rather than by an
     * empty snapshot.
     */
    private void applyUrls(List<String> urls, String source) {
        List<String> prev = fePoolUrls.getAndUpdate(current ->
                urls.isEmpty() && !current.isEmpty() ? current : urls);
        if (urls.isEmpty() && !prev.isEmpty()) {
            emptyDiscoveryWarn.warn("dispatcher FE pool: ignoring empty discovery result ({}), "
                    + "keeping {} known FE(s): serviceId={}", source, prev.size(), serviceId);
            return;
        }
        if (!prev.equals(urls)) {
            Logger.warn("dispatcher FE pool updated ({}): serviceId={}, hosts={} (was {})",
                    source, serviceId, urls.size(), prev.size());
        }
    }

    /**
     * Resolve the FE pool with a bound. Both callers run on shared pools, so a wedged discovery
     * client must not park a thread indefinitely — cap the wait and surface a timeout, which the
     * caller degrades to a warn (the retained snapshot is kept by {@link #applyUrls}).
     */
    private List<WorkerHost> boundedGetHosts() throws Exception {
        Future<List<WorkerHost>> future = discoveryExecutor.submit(() -> serviceDiscovery.getHosts(serviceId));
        try {
            return future.get(DISCOVERY_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw e;
        }
    }

    @PreDestroy
    void shutdown() {
        discoveryExecutor.shutdownNow();
    }

    private static List<String> toUrls(List<WorkerHost> hosts) {
        return hosts.stream().map(h -> "http://" + h.getIpPort()).collect(Collectors.toList());
    }
}
