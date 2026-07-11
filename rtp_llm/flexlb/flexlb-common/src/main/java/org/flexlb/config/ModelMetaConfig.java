package org.flexlb.config;

import org.flexlb.dao.route.ServiceRoute;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class ModelMetaConfig {

    /**
     * Model metadata configuration
     */
    private final ConcurrentHashMap<String/*serviceId*/, ServiceRoute> modelServiceRoute = new ConcurrentHashMap<>();

    public void putServiceRoute(String serviceId, ServiceRoute serviceRoute) {
        modelServiceRoute.put(serviceId, serviceRoute);
    }

    public ServiceRoute getServiceRoute(String serviceId) {
        return modelServiceRoute.get(serviceId);
    }

    public Collection<ServiceRoute> getServiceRoutes() {
        return List.copyOf(modelServiceRoute.values());
    }
}
