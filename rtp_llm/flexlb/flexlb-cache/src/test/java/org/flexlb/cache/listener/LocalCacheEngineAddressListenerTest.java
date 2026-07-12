package org.flexlb.cache.listener;

import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class LocalCacheEngineAddressListenerTest {

    @Test
    void subscribesAndRemovesStaleCachesInLocalMode() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);

        LocalCacheEngineAddressListener listener = new LocalCacheEngineAddressListener(
                addressResolver, kvCacheManager, modelMetaConfig(false));
        verify(addressResolver).subscribe(listener);

        List<String> hosts = List.of("10.0.0.1:8080");
        listener.onAddressUpdate(hosts);
        listener.onAddressUpdate(null);

        verify(kvCacheManager).removeStaleEngineCaches(hosts);
        verify(kvCacheManager, never()).removeStaleEngineCaches(null);
    }

    @Test
    void doesNotSubscribeInKvcmMode() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        KvCacheManager kvCacheManager = mock(KvCacheManager.class);

        new LocalCacheEngineAddressListener(
                addressResolver, kvCacheManager, modelMetaConfig(true));

        verify(addressResolver, never()).subscribe(any());
    }

    private ModelMetaConfig modelMetaConfig(boolean kvcmEnabled) {
        ServiceRoute serviceRoute = new ServiceRoute();
        serviceRoute.setServiceId("test-service");
        if (kvcmEnabled) {
            KvcmConfig kvcmConfig = new KvcmConfig();
            kvcmConfig.setEnabled(true);
            serviceRoute.setKvcm(kvcmConfig);
        }

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(serviceRoute.getServiceId(), serviceRoute);
        return modelMetaConfig;
    }
}
