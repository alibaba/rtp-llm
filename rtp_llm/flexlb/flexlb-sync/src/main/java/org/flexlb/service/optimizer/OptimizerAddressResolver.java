package org.flexlb.service.optimizer;

import io.micrometer.core.instrument.util.NamedThreadFactory;
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
 * Resolves optimizer addresses from a shared {@link ServiceDiscovery} bean.
 *
 * <p>Discovery supplies candidate IPs; the configured port is applied to every
 * discovered host. Dynamic providers are pulled on a resolver-owned background thread.
 * This avoids installing a private listener in a shared provider and keeps
 * {@link #getAddresses()} non-blocking. A successful empty result clears the snapshot;
 * a failed pull keeps the last known snapshot. Invalid hosts are skipped; a response
 * with no valid hosts keeps the last known snapshot. {@link #shutdown()} never closes
 * the shared discovery bean.</p>
 */
@Slf4j
public class OptimizerAddressResolver {

    private final ServiceDiscovery serviceDiscovery;
    private final String address;
    private final Endpoint endpoint;
    private final int port;
    private final long pollIntervalMs;
    private final ScheduledExecutorService refreshScheduler =
            Executors.newSingleThreadScheduledExecutor(
                    new NamedThreadFactory("optimizer-discovery-refresh"));

    private final AtomicBoolean started = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private volatile List<String> resolvedAddresses = Collections.emptyList();

    public OptimizerAddressResolver(ServiceDiscovery serviceDiscovery, Endpoint endpoint, int port) {
        this(serviceDiscovery, endpoint, port, 1000L);
    }

    public OptimizerAddressResolver(
            ServiceDiscovery serviceDiscovery,
            Endpoint endpoint,
            int port,
            long pollIntervalMs) {
        this.serviceDiscovery = serviceDiscovery;
        this.endpoint = endpoint;
        this.address = endpoint.getAddress();
        this.port = port;
        this.pollIntervalMs = pollIntervalMs;
    }

    public void start() {
        if (shutdown.get()) {
            log.info("OptimizerAddressResolver already shutdown, skip start, address={}", address);
            return;
        }
        if (!started.compareAndSet(false, true)) {
            return;
        }
        if (isDynamicDiscovery()) {
            try {
                refreshScheduler.scheduleWithFixedDelay(
                        this::refreshSafely,
                        0,
                        pollIntervalMs,
                        TimeUnit.MILLISECONDS);
            } catch (RejectedExecutionException e) {
                log.warn("Service discovery refresh scheduling failed, address={}, msg={}",
                        address, e.getMessage());
                return;
            }
        } else {
            refreshSafely();
        }
        log.info("OptimizerAddressResolver started: address={}, initialCount={}",
                address, resolvedAddresses.size());
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
        if (shutdown.get()) {
            return;
        }
        if (hosts == null || hosts.isEmpty()) {
            if (!resolvedAddresses.isEmpty()) {
                this.resolvedAddresses = Collections.emptyList();
                log.info("OptimizerAddressResolver cleared, address={}", address);
            }
            return;
        }
        List<String> addresses = new ArrayList<>(hosts.size());
        for (WorkerHost host : hosts) {
            String ip = host == null ? null : host.getIp();
            if (ip == null
                    || ip.isBlank()
                    || ip.indexOf(':') >= 0
                    || port <= 0
                    || port > 65535) {
                log.warn("Ignoring invalid discovered optimizer host: {}", host);
                continue;
            }
            addresses.add(ip + ":" + port);
        }
        if (addresses.isEmpty()) {
            log.warn("OptimizerAddressResolver found no valid hosts, keeping previous snapshot: {}",
                    address);
            return;
        }
        List<String> snapshot = Collections.unmodifiableList(addresses);
        if (!snapshot.equals(resolvedAddresses)) {
            this.resolvedAddresses = snapshot;
            log.info("OptimizerAddressResolver updated, address={}, count={}", address, addresses.size());
        }
    }

    public List<String> getAddresses() {
        return resolvedAddresses;
    }

    public void shutdown() {
        shutdown.set(true);
        refreshScheduler.shutdownNow();
    }

    private boolean isDynamicDiscovery() {
        ServiceDiscoveryType type = endpoint.getDiscovery().getType();
        return type == ServiceDiscoveryType.DASHSCOPE || type == ServiceDiscoveryType.VIPSERVER;
    }

}
