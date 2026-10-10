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
     * Synchronously get host list by service address
     *
     * @param address Service address
     * @return Host list
     */
    List<WorkerHost> getHosts(String address);

    /** Control-plane targets, including registered instances not yet traffic-ready. */
    default List<WorkerHost> getAllHosts(String address) { return getHosts(address); }

    /**
     * Listen for host changes at service address
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
