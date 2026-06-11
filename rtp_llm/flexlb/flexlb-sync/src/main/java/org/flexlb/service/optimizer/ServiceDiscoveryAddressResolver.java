package org.flexlb.service.optimizer;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Optimizer address resolver backed by a shared {@link ServiceDiscovery} bean.
 *
 * <p>Dynamic providers are pulled on a resolver-owned background thread. This avoids
 * installing a private listener in a shared provider and keeps {@link #getAddresses()}
 * non-blocking. A successful empty result clears the snapshot; a failed pull keeps the
 * last known snapshot. {@link #shutdown()} never closes the shared discovery bean.</p>
 */
@Slf4j
public class ServiceDiscoveryAddressResolver implements OptimizerAddressResolver {

    private final ServiceDiscovery serviceDiscovery;
    private final String address;
    private final Endpoint endpoint;
    private final ScheduledExecutorService refreshScheduler =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread thread = new Thread(r, "optimizer-discovery-refresh");
                thread.setDaemon(true);
                return thread;
            });

    private final AtomicBoolean started = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private volatile List<String> resolvedAddresses = Collections.emptyList();

    public ServiceDiscoveryAddressResolver(ServiceDiscovery serviceDiscovery, Endpoint endpoint) {
        this.serviceDiscovery = serviceDiscovery;
        this.endpoint = endpoint;
        this.address = endpoint.getAddress();
    }

    /** Idempotent + retryable. See {@link OptimizerAddressResolver#start()}. */
    @Override
    public boolean start() {
        if (shutdown.get()) {
            log.info("ServiceDiscoveryAddressResolver already shutdown, skip start, address={}", address);
            return false;
        }
        if (!started.compareAndSet(false, true)) {
            return true;
        }
        try {
            serviceDiscovery.validate(endpoint);
        } catch (Throwable t) {
            started.set(false);
            log.warn("ServiceDiscovery.validate failed, address={}, msg={}", address, t.getMessage());
            return false;
        }
        refreshSafely();
        if (shutdown.get()) {
            started.set(false);
            return false;
        }
        if (isDynamicDiscovery()) {
            long pollIntervalMs = endpoint.getDiscovery().getPollIntervalMs();
            if (pollIntervalMs <= 0) {
                started.set(false);
                log.warn("Service discovery poll interval must be greater than zero, address={}", address);
                return false;
            }
            try {
                refreshScheduler.scheduleWithFixedDelay(
                        this::refreshSafely,
                        pollIntervalMs,
                        pollIntervalMs,
                        TimeUnit.MILLISECONDS);
            } catch (RejectedExecutionException e) {
                started.set(false);
                log.warn("Service discovery refresh scheduling failed, address={}, msg={}",
                        address, e.getMessage());
                return false;
            }
        }
        log.info("ServiceDiscoveryAddressResolver started: address={}, initialCount={}",
                address, resolvedAddresses.size());
        return true;
    }

    private void refreshSafely() {
        if (shutdown.get()) {
            return;
        }
        try {
            updateFromHosts(serviceDiscovery.getHosts(endpoint));
        } catch (Throwable t) {
            log.warn("ServiceDiscovery.getHosts failed, address={}, msg={}", address, t.getMessage());
        }
    }

    private void updateFromHosts(List<WorkerHost> hosts) {
        // Drop callbacks after shutdown to avoid stale mutations
        if (shutdown.get()) {
            return;
        }
        if (hosts == null || hosts.isEmpty()) {
            if (!resolvedAddresses.isEmpty()) {
                this.resolvedAddresses = Collections.emptyList();
                log.info("ServiceDiscoveryAddressResolver cleared, address={}", address);
            }
            return;
        }
        List<String> addresses = new ArrayList<>(hosts.size());
        for (WorkerHost host : hosts) {
            String ip = host == null ? null : host.getIp();
            int port = host == null ? 0 : host.getPort();
            if (ip == null
                    || ip.isBlank()
                    || ip.indexOf(':') >= 0
                    || port <= 0
                    || port > 65535) {
                throw new IllegalArgumentException("Invalid discovered optimizer host: " + host);
            }
            addresses.add(ip + ":" + port);
        }
        List<String> snapshot = Collections.unmodifiableList(addresses);
        if (!snapshot.equals(resolvedAddresses)) {
            this.resolvedAddresses = snapshot;
            log.info("ServiceDiscoveryAddressResolver updated, address={}, count={}", address, addresses.size());
        }
    }

    @Override
    public List<String> getAddresses() {
        return resolvedAddresses;
    }

    @Override
    public void shutdown() {
        shutdown.set(true);
        refreshScheduler.shutdownNow();
    }

    private boolean isDynamicDiscovery() {
        ServiceDiscoveryType type = endpoint.getDiscovery().getType();
        return type == ServiceDiscoveryType.DASHSCOPE || type == ServiceDiscoveryType.VIPSERVER;
    }

}
