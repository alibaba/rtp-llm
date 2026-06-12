package org.flexlb.service.optimizer;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Optimizer address resolver backed by a shared {@link ServiceDiscovery} bean.
 *
 * <p>Behavior aligned with {@code EngineAddressNameResolver}/{@code WorkerAddressService}:
 * a null or empty host list clears the resolved addresses. Caller decides whether to
 * invoke {@link #start()} synchronously or on a background thread.</p>
 *
 * <p>If {@code listen} fails, {@code started} is rolled back so the next {@link #start()} can retry.
 * {@link #shutdown()} only sets a flag (the underlying {@link ServiceDiscovery} is a shared bean
 * with no {@code unlisten} API), preventing late listener callbacks from mutating state.</p>
 */
@Slf4j
public class ServiceDiscoveryAddressResolver implements OptimizerAddressResolver {

    private final ServiceDiscovery serviceDiscovery;
    private final String domain;

    private final AtomicBoolean started = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private volatile List<String> resolvedAddresses = Collections.emptyList();

    public ServiceDiscoveryAddressResolver(ServiceDiscovery serviceDiscovery, String domain) {
        this.serviceDiscovery = serviceDiscovery;
        this.domain = domain;
    }

    public void start() {
        if (shutdown.get()) {
            log.info("ServiceDiscoveryAddressResolver already shutdown, skip start, domain={}", domain);
            return;
        }
        if (!started.compareAndSet(false, true)) {
            log.info("ServiceDiscoveryAddressResolver already started, skip duplicate start, domain={}", domain);
            return;
        }
        // Initial pull so getAddresses() is non-empty before listener fires
        try {
            updateFromHosts(serviceDiscovery.getHosts(domain));
        } catch (Throwable t) {
            log.warn("ServiceDiscovery.getHosts failed on start, domain={}, msg={}", domain, t.getMessage());
        }
        // Register push listener; roll back started on failure to allow retry
        boolean listenOk = false;
        try {
            serviceDiscovery.listen(domain, this::updateFromHosts);
            listenOk = true;
        } catch (Throwable t) {
            log.warn("ServiceDiscovery.listen failed, domain={}, msg={}", domain, t.getMessage());
        }
        if (!listenOk) {
            // Roll back so next start() can re-attempt initial pull + listen
            started.set(false);
            log.warn("ServiceDiscoveryAddressResolver listen registration failed, started rollback, domain={}", domain);
            return;
        }
        log.info("ServiceDiscoveryAddressResolver started: domain={}, initialCount={}",
                domain, resolvedAddresses.size());
    }

    private void updateFromHosts(List<WorkerHost> hosts) {
        // Ignore callbacks after shutdown to prevent stale listener mutations
        if (shutdown.get()) {
            return;
        }
        if (hosts == null || hosts.isEmpty()) {
            this.resolvedAddresses = Collections.emptyList();
            log.info("ServiceDiscoveryAddressResolver cleared, domain={}", domain);
            return;
        }
        List<String> addresses = new ArrayList<>(hosts.size());
        for (WorkerHost host : hosts) {
            addresses.add(host.getIp() + ":" + host.getPort());
        }
        this.resolvedAddresses = addresses;
        log.info("ServiceDiscoveryAddressResolver updated, domain={}, count={}", domain, addresses.size());
    }

    @Override
    public List<String> getAddresses() {
        return resolvedAddresses;
    }

    @Override
    public void shutdown() {
        // ServiceDiscovery is a shared Spring bean; only set a flag here
        // to block subsequent listener callbacks from mutating state.
        shutdown.set(true);
    }
}
