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
 * <p>Empty/null host list clears resolved addresses. {@link #start()} is idempotent
 * and rolls back on listen failure so callers can retry. {@link #shutdown()} only
 * sets a flag (ServiceDiscovery is shared, has no unlisten API) to block late callbacks.</p>
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

    /** Idempotent + retryable. See {@link OptimizerAddressResolver#start()}. */
    @Override
    public boolean start() {
        if (shutdown.get()) {
            log.info("ServiceDiscoveryAddressResolver already shutdown, skip start, domain={}", domain);
            return false;
        }
        if (!started.compareAndSet(false, true)) {
            return true;
        }
        // Initial pull so getAddresses() is non-empty before listener fires
        try {
            updateFromHosts(serviceDiscovery.getHosts(domain));
        } catch (Throwable t) {
            log.warn("ServiceDiscovery.getHosts failed on start, domain={}, msg={}", domain, t.getMessage());
        }
        try {
            serviceDiscovery.listen(domain, this::updateFromHosts);
        } catch (Throwable t) {
            // Roll back so the next start() can re-attempt
            started.set(false);
            log.warn("ServiceDiscovery.listen failed, domain={}, msg={}", domain, t.getMessage());
            return false;
        }
        log.info("ServiceDiscoveryAddressResolver started: domain={}, initialCount={}",
                domain, resolvedAddresses.size());
        return true;
    }

    private void updateFromHosts(List<WorkerHost> hosts) {
        // Drop callbacks after shutdown to avoid stale mutations
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
        this.resolvedAddresses = Collections.unmodifiableList(addresses);
        log.info("ServiceDiscoveryAddressResolver updated, domain={}, count={}", domain, addresses.size());
    }

    @Override
    public List<String> getAddresses() {
        return resolvedAddresses;
    }

    @Override
    public void shutdown() {
        // ServiceDiscovery is a shared bean; only flag here to block late callbacks.
        shutdown.set(true);
    }
}
