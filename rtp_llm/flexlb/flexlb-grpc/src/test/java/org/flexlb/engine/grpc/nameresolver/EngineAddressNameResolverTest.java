package org.flexlb.engine.grpc.nameresolver;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.MapPropertySource;

import java.util.List;
import java.util.Map;

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
              "service_id": "test-service",
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
            context.getEnvironment().getPropertySources().addFirst(
                    new MapPropertySource("test", Map.of("MODEL_SERVICE_CONFIG", MODEL_CONFIG)));
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
        EngineAddressNameResolver resolver = new EngineAddressNameResolver(discovery, MODEL_CONFIG);
        CustomNameResolver.Listener listener = mock(CustomNameResolver.Listener.class);
        resolver.start(listener);
        clearInvocations(listener);

        resolver.periodicHostUpdate();

        verify(listener, never()).onAddressUpdate(anyList());
    }
}
