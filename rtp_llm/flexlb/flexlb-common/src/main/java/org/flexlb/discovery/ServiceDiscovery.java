package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;

import java.util.List;

/**
 * ServiceDiscovery - Service discovery interface
 *
 * @author saichen.sm
 */
public interface ServiceDiscovery {

    /**
     * Synchronously get host list by service address.
     *
     * <p>An empty list means an empty fleet and nothing else. A failed lookup must throw —
     * callers act on "empty" as authoritative (embedding liveness marks a whole fleet dead on
     * it), so an implementation that swallows a failure into an empty list mass-kills healthy
     * workers.
     *
     * @param address Service address
     * @return Host list
     */
    List<WorkerHost> getHosts(String address);

    /**
     * Listen for host changes at service address.
     *
     * <p>Registration, not a lookup: implementations must not propagate a host-resolution
     * failure out of this call, so callers need no defensive wrapper around it.
     *
     * @param address  Service address
     * @param listener Host change listener
     */
    void listen(String address, ServiceHostListener listener);

    /**
     * Stop all listeners
     */
    void shutdown();
}
