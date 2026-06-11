package org.flexlb.service.optimizer;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.optimizer.CommonResponseHeader;
import org.flexlb.dao.optimizer.OptimizerErrorCode;
import org.flexlb.dao.optimizer.OptimizerGetInstanceResponse;
import org.flexlb.dao.optimizer.OptimizerRegisterRequest;
import org.flexlb.dao.optimizer.OptimizerRegisterResponse;
import org.flexlb.dao.route.DiscoveryConfig;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.OnlineOptimizerConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class OnlineOptimizerHookerTest {

    @Mock
    private GeneralHttpNettyService httpService;

    @Mock
    private ServiceDiscovery serviceDiscovery;

    @Test
    void disablesClientWhenOnlineOptimizerConfigurationIsAbsent() {
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(null));

        assertFalse(hooker.isEnabled());
        hooker.afterStartUp();
        assertNull(hooker.getClient());
        verify(serviceDiscovery, never()).getHosts(any());
    }

    @Test
    void disablesClientWhenOnlineOptimizerConfigurationIsDisabled() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        config.setEnabled(false);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        assertFalse(hooker.isEnabled());
        hooker.afterStartUp();
        assertNull(hooker.getClient());
        verify(serviceDiscovery, never()).getHosts(any());
    }

    @Test
    void usesStaticEndpointFromModelServiceConfiguration() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        assertTrue(hooker.isEnabled());
        hooker.afterStartUp();
        assertNotNull(hooker.getClient());

        ArgumentCaptor<Endpoint> endpointCaptor = ArgumentCaptor.forClass(Endpoint.class);
        verify(serviceDiscovery, timeout(3000)).validate(endpointCaptor.capture());
        Endpoint endpoint = endpointCaptor.getValue();
        assertEquals("optimizer-service", endpoint.getAddress());
        assertEquals("http", endpoint.getProtocol());
        assertEquals(ServiceDiscoveryType.STATIC_ENV, endpoint.getDiscovery().getType());
        assertEquals(List.of("10.0.0.1:8082"), endpoint.getDiscovery().getHosts());
        hooker.beforeShutdown();
    }

    @Test
    void usesDashScopeEndpointFromModelServiceConfiguration() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.DASHSCOPE);
        config.getDiscovery().setBaseUrl("http://127.0.0.1:18880");
        config.getDiscovery().setPollIntervalMs(1234);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        hooker.afterStartUp();

        ArgumentCaptor<Endpoint> endpointCaptor = ArgumentCaptor.forClass(Endpoint.class);
        verify(serviceDiscovery, timeout(3000)).validate(endpointCaptor.capture());
        Endpoint endpoint = endpointCaptor.getValue();
        assertEquals(ServiceDiscoveryType.DASHSCOPE, endpoint.getDiscovery().getType());
        assertEquals("http://127.0.0.1:18880", endpoint.getDiscovery().getBaseUrl());
        assertEquals(1234L, endpoint.getDiscovery().getPollIntervalMs());
        verify(serviceDiscovery, never()).listen(any(), any());
        hooker.beforeShutdown();
    }

    @Test
    void usesVipServerEndpointFromModelServiceConfiguration() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.VIPSERVER);
        config.getDiscovery().setPollIntervalMs(1234);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        hooker.afterStartUp();

        ArgumentCaptor<Endpoint> endpointCaptor = ArgumentCaptor.forClass(Endpoint.class);
        verify(serviceDiscovery, timeout(3000)).validate(endpointCaptor.capture());
        Endpoint endpoint = endpointCaptor.getValue();
        assertEquals(ServiceDiscoveryType.VIPSERVER, endpoint.getDiscovery().getType());
        assertEquals("optimizer-service", endpoint.getAddress());
        assertEquals("http", endpoint.getProtocol());
        assertEquals(1234L, endpoint.getDiscovery().getPollIntervalMs());
        verify(serviceDiscovery, never()).listen(any(), any());
        hooker.beforeShutdown();
    }

    @Test
    void sendsConfiguredIdentityAndLatestRegistrationPayload() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        config.setPath("/custom/optimizer");
        when(serviceDiscovery.getHosts(any(Endpoint.class)))
                .thenReturn(List.of(WorkerHost.of("10.0.0.1", 8082)));

        OptimizerGetInstanceResponse getResponse = new OptimizerGetInstanceResponse();
        getResponse.setHeader(header(OptimizerErrorCode.INSTANCE_NOT_EXIST));
        when(httpService.request(any(), any(URI.class), eq("/custom/optimizer/getInstance"),
                eq(OptimizerGetInstanceResponse.class))).thenReturn(Mono.just(getResponse));

        OptimizerRegisterResponse registerResponse = new OptimizerRegisterResponse();
        registerResponse.setHeader(header(OptimizerErrorCode.OK));
        when(httpService.request(any(), any(URI.class), eq("/custom/optimizer/registerInstance"),
                eq(OptimizerRegisterResponse.class))).thenReturn(Mono.just(registerResponse));

        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));
        hooker.afterStartUp();

        ArgumentCaptor<OptimizerRegisterRequest> requestCaptor =
                ArgumentCaptor.forClass(OptimizerRegisterRequest.class);
        verify(httpService, timeout(3000)).request(
                requestCaptor.capture(), any(URI.class),
                eq("/custom/optimizer/registerInstance"), eq(OptimizerRegisterResponse.class));
        OptimizerRegisterRequest request = requestCaptor.getValue();
        assertEquals("test-group", request.getInstanceGroup());
        assertEquals("test-instance", request.getInstanceId());
        assertEquals(64, request.getBlockSize());
        assertEquals(4, request.getLinearStep());
        assertEquals(4_294_967_296L, request.getLocationSpecInfos().getFirst().getSize());
        assertEquals(List.of("linear"), request.getLocationSpecGroups().get(1).getSpecNames());
        assertEquals("full-group", request.getOptimizerStateInfo().getFullLocationSpecGroupName());
        assertEquals("linear-group", request.getOptimizerStateInfo().getLinearLocationSpecGroupName());
        hooker.beforeShutdown();
    }

    @Test
    void startsOnlyOnceWhenStartupHookRunsMoreThanOnce() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        hooker.afterStartUp();
        hooker.afterStartUp();

        verify(serviceDiscovery, timeout(3000).times(1)).validate(any(Endpoint.class));
        hooker.beforeShutdown();
    }

    @Test
    void startup_hook_should_not_wait_for_service_discovery() throws Exception {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        CountDownLatch discoveryStarted = new CountDownLatch(1);
        CountDownLatch releaseDiscovery = new CountDownLatch(1);
        when(serviceDiscovery.getHosts(any(Endpoint.class))).thenAnswer(invocation -> {
            discoveryStarted.countDown();
            releaseDiscovery.await(3, TimeUnit.SECONDS);
            return List.of();
        });
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        assertTimeoutPreemptively(Duration.ofMillis(250), hooker::afterStartUp);
        assertTrue(discoveryStarted.await(3, TimeUnit.SECONDS));

        releaseDiscovery.countDown();
        hooker.beforeShutdown();
    }

    @Test
    void shutsDownSafelyWhenDisabled() {
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(null));

        hooker.beforeShutdown();

        assertNull(hooker.getClient());
    }

    @Test
    void shutdown_before_startup_prevents_late_client_creation() {
        OnlineOptimizerConfig config = validConfig(ServiceDiscoveryType.STATIC_ENV);
        OnlineOptimizerHooker hooker = new OnlineOptimizerHooker(
                httpService, serviceDiscovery, modelMetaConfig(config));

        hooker.beforeShutdown();
        hooker.afterStartUp();

        assertNull(hooker.getClient());
        verify(serviceDiscovery, never()).validate(any(Endpoint.class));
        verify(serviceDiscovery, never()).getHosts(any(Endpoint.class));
    }

    private static ModelMetaConfig modelMetaConfig(OnlineOptimizerConfig optimizerConfig) {
        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setOnlineOptimizer(optimizerConfig);
        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        return modelMetaConfig;
    }

    private static OnlineOptimizerConfig validConfig(ServiceDiscoveryType discoveryType) {
        DiscoveryConfig discovery = new DiscoveryConfig();
        discovery.setType(discoveryType);
        if (discoveryType == ServiceDiscoveryType.STATIC_ENV) {
            discovery.setHosts(List.of("10.0.0.1:8082"));
        }

        OnlineOptimizerConfig.LocationSpecInfo full = new OnlineOptimizerConfig.LocationSpecInfo();
        full.setName("full");
        full.setSize(4_294_967_296L);
        OnlineOptimizerConfig.LocationSpecInfo linear = new OnlineOptimizerConfig.LocationSpecInfo();
        linear.setName("linear");
        linear.setSize(65_536L);

        OnlineOptimizerConfig.LocationSpecGroup fullGroup = new OnlineOptimizerConfig.LocationSpecGroup();
        fullGroup.setName("full-group");
        fullGroup.setSpecNames(List.of("full"));
        OnlineOptimizerConfig.LocationSpecGroup linearGroup = new OnlineOptimizerConfig.LocationSpecGroup();
        linearGroup.setName("linear-group");
        linearGroup.setSpecNames(List.of("linear"));

        OnlineOptimizerConfig.OptimizerStateInfo stateInfo = new OnlineOptimizerConfig.OptimizerStateInfo();
        stateInfo.setFullLocationSpecGroupName("full-group");
        stateInfo.setLinearLocationSpecGroupName("linear-group");

        OnlineOptimizerConfig config = new OnlineOptimizerConfig();
        config.setEnabled(true);
        config.setAddress("optimizer-service");
        config.setDiscovery(discovery);
        config.setInstanceGroup("test-group");
        config.setInstanceId("test-instance");
        config.setBlockSize(64);
        config.setLinearStep(4);
        config.setLocationSpecInfos(List.of(full, linear));
        config.setLocationSpecGroups(List.of(fullGroup, linearGroup));
        config.setOptimizerStateInfo(stateInfo);
        return config;
    }

    private static CommonResponseHeader header(OptimizerErrorCode code) {
        CommonResponseHeader.Status status = new CommonResponseHeader.Status();
        status.setCode(code);
        CommonResponseHeader header = new CommonResponseHeader();
        header.setStatus(status);
        return header;
    }
}
