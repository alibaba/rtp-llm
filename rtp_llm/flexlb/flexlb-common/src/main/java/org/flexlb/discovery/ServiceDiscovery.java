package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;

import java.util.List;

/**
 * ServiceDiscovery - Service discovery interface
 *
 * @author saichen.sm
 */
public interface ServiceDiscovery {

    /**
     * Validate an endpoint and initialize provider-specific resources.
     *
     * @param endpoint Endpoint and its discovery configuration
     */
    void validate(Endpoint endpoint);

    /**
     * Synchronously get the host list for an endpoint.
     *
     * @param endpoint Endpoint and its discovery configuration
     * @return Host list
     */
    List<WorkerHost> getHosts(Endpoint endpoint);

    /**
     * Listen for host changes for an endpoint.
     *
     * @param endpoint Endpoint and its discovery configuration
     * @param listener Host change listener
     */
    void listen(Endpoint endpoint, ServiceHostListener listener);

    /**
     * Stop all listeners
     */
    void shutdown();
}
