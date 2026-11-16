package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;

import java.util.List;

/**
 * Provider for one service discovery type.
 */
public interface ServiceDiscoveryProvider {

    ServiceDiscoveryType getType();

    void validate(Endpoint endpoint);

    List<WorkerHost> getHosts(Endpoint endpoint);

    void listen(Endpoint endpoint, ServiceHostListener listener);

    void shutdown();
}
