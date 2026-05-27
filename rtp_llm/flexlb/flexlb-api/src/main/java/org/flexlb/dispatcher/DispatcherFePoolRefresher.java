package org.flexlb.dispatcher;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;
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
 * a concurrent {@code getAndSet} is overwritten on the next event, and the URL list itself
 * is an idempotent snapshot so transient ordering does not corrupt downstream state.
 *
 * <p>Constructed only when the dispatcher is enabled — Spring's {@code @ConditionalOnProperty}
 * gates this {@code @Component} on {@code dispatch.enabled=true}, so disabled deployments do
 * not pay for an idle scheduler or a parked listener.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "enabled", havingValue = "true")
public class DispatcherFePoolRefresher {

    private final ServiceDiscovery serviceDiscovery;
    private final String serviceId;
    private final AtomicReference<List<String>> fePoolUrls = new AtomicReference<>(List.of());

    public DispatcherFePoolRefresher(ServiceDiscovery serviceDiscovery, DispatchConfig cfg) {
        this.serviceDiscovery = serviceDiscovery;
        this.serviceId = cfg.getFePoolServiceId();
        // Boot seed first — guarantees source() returns the freshest available view by the
        // time downstream beans (FePool, FeHealthChecker) read it during their own init.
        applyUrls(toUrls(serviceDiscovery.getHosts(serviceId)), "boot");
        serviceDiscovery.listen(serviceId, hosts -> applyUrls(toUrls(hosts), "listener"));
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
            applyUrls(toUrls(serviceDiscovery.getHosts(serviceId)), "poll");
        } catch (Exception e) {
            Logger.warn("dispatcher FE pool refresh failed: serviceId={}, err={}: {}",
                    serviceId, e.getClass().getSimpleName(), e.getMessage());
        }
    }

    /**
     * Replace the snapshot and WARN if (and only if) the URL set actually changed. Both
     * the listener callback and the {@code @Scheduled} poll route through here so the
     * "changed?" check stays in one place. Steady state — same list arriving twice — emits
     * nothing, so the log does not get flooded by no-op refreshes.
     */
    private void applyUrls(List<String> urls, String source) {
        List<String> prev = fePoolUrls.getAndSet(urls);
        if (!prev.equals(urls)) {
            Logger.warn("dispatcher FE pool updated ({}): serviceId={}, hosts={} (was {})",
                    source, serviceId, urls.size(), prev.size());
        }
    }

    private static List<String> toUrls(List<WorkerHost> hosts) {
        return hosts.stream().map(h -> "http://" + h.getIpPort()).collect(Collectors.toList());
    }
}
