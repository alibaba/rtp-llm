package org.flexlb.engine.grpc.nameresolver;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineAddressNameResolverTest {

    private static final String MODEL_CONFIG = """
            {
              "service_id": "aigc.text-generation.generation.test-service",
              "role_endpoints": [{
                "group": "test-group",
                "prefill_endpoint": {
                  "address": "test.prefill",
                  "protocol": "http",
                  "path": "/"
                }
              }]
            }
            """;

    @Test
    void spring_constructs_resolver_with_explicit_model_config_dependency() {
        ServiceDiscovery discovery = mock(ServiceDiscovery.class);
        when(discovery.getHosts("test.prefill"))
                .thenReturn(List.of(new WorkerHost("10.0.0.1", 8080)));

        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext()) {
            context.registerBean(ModelMetaConfig.class, () -> new ModelMetaConfig(MODEL_CONFIG));
            context.registerBean(ServiceDiscovery.class, () -> discovery);
            context.registerBean(EngineAddressNameResolver.class);
            context.refresh();

            assertNotNull(context.getBean(EngineAddressNameResolver.class));
        }
    }

    @Test
    void unchanged_periodic_membership_does_not_notify_listener() {
        ServiceDiscovery discovery = mock(ServiceDiscovery.class);
        when(discovery.getHosts("test.prefill"))
                .thenReturn(List.of(new WorkerHost("10.0.0.1", 8080)));
        EngineAddressNameResolver resolver = new EngineAddressNameResolver(discovery, new ModelMetaConfig(MODEL_CONFIG));
        CustomNameResolver.Listener listener = mock(CustomNameResolver.Listener.class);
        resolver.start(listener);
        clearInvocations(listener);

        resolver.periodicHostUpdate();

        verify(listener, never()).onAddressUpdate(anyList());
    }
}
