package org.flexlb.service.optimizer;

import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@ExtendWith(SystemStubsExtension.class)
class OnlineOptimizerHookerTest {

    @Mock
    private GeneralHttpNettyService httpService;

    @Mock
    private ServiceDiscovery serviceDiscovery;

    @SystemStub
    private EnvironmentVariables env = new EnvironmentVariables();

    @Test
    void should_be_disabled_when_no_env_vars_configured() {
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);

        assertFalse(hooker.isEnabled());
        assertNull(hooker.getClient());
        hooker.afterStartUp();
        assertNull(hooker.getClient());
    }

    @Test
    void should_be_disabled_when_only_address_configured() {
        env.set("OPTIMIZER_DIRECT_ADDRESS", "10.0.0.1:8082");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);

        assertFalse(hooker.isEnabled());
        assertNull(hooker.getClient());
    }

    @Test
    void should_be_enabled_when_address_and_group_configured() {
        env.set("OPTIMIZER_DIRECT_ADDRESS", "10.0.0.1:8082");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("ONLINE_OPTIMIZER_INSTANCE_ID", "test-instance");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);

        assertTrue(hooker.isEnabled());
        // client is created lazily in afterStartUp
        assertNull(hooker.getClient());

        hooker.afterStartUp();
        assertNotNull(hooker.getClient());
    }

    @Test
    void should_shutdown_safely_when_disabled() {
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);

        hooker.beforeShutdown();
    }

    @Test
    void should_shutdown_client_when_enabled() {
        env.set("OPTIMIZER_DIRECT_ADDRESS", "10.0.0.1:8082");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("ONLINE_OPTIMIZER_INSTANCE_ID", "test-instance");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);
        assertTrue(hooker.isEnabled());

        hooker.afterStartUp();
        assertNotNull(hooker.getClient());

        hooker.beforeShutdown();
    }

    @Test
    void should_resolve_instanceId_from_model_service_config() {
        env.set("OPTIMIZER_DIRECT_ADDRESS", "10.0.0.1:8082");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("MODEL_SERVICE_CONFIG", "{\"service_id\":\"function.qwen-72b\",\"role_endpoints\":[]}");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);
        assertTrue(hooker.isEnabled());

        hooker.afterStartUp();
        assertNotNull(hooker.getClient());
    }

    @Test
    void should_use_explicit_instanceId_when_configured() {
        env.set("OPTIMIZER_DIRECT_ADDRESS", "10.0.0.1:8082");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("ONLINE_OPTIMIZER_INSTANCE_ID", "my-custom-id");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);
        assertTrue(hooker.isEnabled());

        hooker.afterStartUp();
        assertNotNull(hooker.getClient());
    }

    @Test
    void should_be_enabled_when_vipserver_domain_configured() {
        env.set("OPTIMIZER_VIPSERVER_DOMAIN", "optimizer.test.domain.com");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("ONLINE_OPTIMIZER_INSTANCE_ID", "test-instance");
        when(serviceDiscovery.getHosts(anyString())).thenReturn(Collections.emptyList());

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);

        assertTrue(hooker.isEnabled());
        // No service-discovery interaction at construction time
        verify(serviceDiscovery, never()).getHosts(any());
        verify(serviceDiscovery, never()).listen(any(), any());

        hooker.afterStartUp();
        assertNotNull(hooker.getClient());
        // ServiceDiscovery is touched only after afterStartUp, via async retry
        verify(serviceDiscovery, timeout(3000)).getHosts("optimizer.test.domain.com");
        verify(serviceDiscovery, timeout(3000)).listen(any(), any());
    }

    @Test
    void should_be_disabled_when_vipserver_domain_set_but_no_service_discovery() {
        env.set("OPTIMIZER_VIPSERVER_DOMAIN", "optimizer.test.domain.com");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");
        env.set("ONLINE_OPTIMIZER_INSTANCE_ID", "test-instance");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, null);

        assertFalse(hooker.isEnabled());
        assertNull(hooker.getClient());
    }

    @Test
    void should_not_touch_service_discovery_when_disabled_due_to_missing_instanceGroup() throws Exception {
        // Only domain set; instanceGroup missing -> disabled, must not touch ServiceDiscovery
        env.set("OPTIMIZER_VIPSERVER_DOMAIN", "optimizer.test.domain.com");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);
        assertFalse(hooker.isEnabled());

        hooker.afterStartUp();
        Thread.sleep(50);
        verify(serviceDiscovery, never()).getHosts(any());
        verify(serviceDiscovery, never()).listen(any(), any());
        assertNull(hooker.getClient());
    }

    @Test
    void should_not_touch_service_discovery_when_disabled_due_to_missing_instanceId() throws Exception {
        // domain + instanceGroup set, instanceId missing -> disabled, must not touch ServiceDiscovery
        env.set("OPTIMIZER_VIPSERVER_DOMAIN", "optimizer.test.domain.com");
        env.set("ONLINE_OPTIMIZER_INSTANCE_GROUP", "test-group");

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(httpService, serviceDiscovery);
        assertFalse(hooker.isEnabled());

        hooker.afterStartUp();
        Thread.sleep(50);
        verify(serviceDiscovery, never()).getHosts(any());
        verify(serviceDiscovery, never()).listen(any(), any());
        assertNull(hooker.getClient());
    }
}
