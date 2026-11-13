package org.flexlb.mock.master;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.discovery.LocalServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.discovery.ServiceHostListener;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Registers the Mock Runner's discovery file as an Engine endpoint provider.
 */
@Component
public final class LocalMasterDiscovery implements ServiceDiscoveryProvider {
    private final LocalServiceDiscovery fileDiscovery;

    public LocalMasterDiscovery(ConfigService configService) {
        this.fileDiscovery = new LocalServiceDiscovery(
                configService.modelServiceConfig().getDiscoveryFile());
    }

    @Override
    public ServiceDiscoveryType getType() {
        return ServiceDiscoveryType.FILE;
    }

    @Override
    public void validate(Endpoint endpoint) {
        getHosts(endpoint);
    }

    @Override
    public List<WorkerHost> getHosts(Endpoint endpoint) {
        return fileDiscovery.getHosts(endpoint.getAddress());
    }

    @Override
    public void listen(Endpoint endpoint, ServiceHostListener listener) {
        fileDiscovery.listen(endpoint.getAddress(), listener);
    }

    @Override
    public void shutdown() {}
}
