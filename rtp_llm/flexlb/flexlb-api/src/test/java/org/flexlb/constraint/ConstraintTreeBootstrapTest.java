package org.flexlb.constraint;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.httpserver.ConstraintTreeBootstrapServer;
import org.flexlb.util.IdUtils;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.server.RouterFunctions;

import java.net.InetSocketAddress;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ConstraintTreeBootstrapTest {
    @Test
    void implicitServiceRequiresExactlyOneConfiguredServiceAndPreservesRoleValidation() {
        var registry = new ConstraintTreeBootstrapRegistry();
        var models = mock(ModelMetaConfig.class);
        var leader = mock(LBStatusConsistencyService.class);
        when(leader.isMaster()).thenReturn(true);
        String service = IdUtils.getServiceIdByModelName("a");
        var route = mock(ServiceRoute.class);
        when(route.getRoleEndpoints(RoleType.PDFUSION)).thenReturn(List.of(new Endpoint()));
        when(models.getServiceRoute(service)).thenReturn(route);
        var handler = RouterFunctions.toWebHandler(
                new ConstraintTreeBootstrapServer(registry, models, leader).constraintTreeBootstrapRoutes());
        String body = "{\"role\":\"PDFUSION\",\"http_port\":23495}";
        for (var services : List.of(Set.<String>of(), Set.of(service, IdUtils.getServiceIdByModelName("b")),
                Set.of(service))) {
            when(models.getServiceIds()).thenReturn(services);
            var exchange = MockServerWebExchange.from(MockServerHttpRequest.post(ConstraintTreeBootstrapServer.PATH)
                    .remoteAddress(new InetSocketAddress("127.0.0.2", 54321))
                    .header("Content-Type", "application/json").body(body));
            handler.handle(exchange).block();
            assertEquals(services.isEmpty() ? 503 : services.size() > 1 ? 400 : 200,
                    exchange.getResponse().getRawStatusCode());
            if (services.size() != 1) {
                assertTrue(registry.pendingModels().isEmpty());
            }
        }
        assertEquals(Map.of("127.0.0.2:23495", URI.create("http://127.0.0.2:23495")), registry.merge("a", Map.of()));
        when(route.getRoleEndpoints(RoleType.PDFUSION)).thenReturn(List.of());
        var exchange = MockServerWebExchange.from(MockServerHttpRequest.post(ConstraintTreeBootstrapServer.PATH)
                .remoteAddress(new InetSocketAddress("127.0.0.3", 54321))
                .header("Content-Type", "application/json").body(body));
        handler.handle(exchange).block();
        assertEquals(400, exchange.getResponse().getRawStatusCode());
        assertFalse(registry.merge("a", Map.of()).containsKey("127.0.0.3:23495"));
    }

    @Test
    void springWiresBootstrapWithoutOptionalIgraphPoller() {
        try (var context = new org.springframework.context.annotation.AnnotationConfigApplicationContext()) {
            context.registerBean(org.flexlb.service.address.WorkerAddressService.class,
                    () -> mock(org.flexlb.service.address.WorkerAddressService.class));
            context.registerBean(org.flexlb.transport.GeneralHttpNettyService.class,
                    () -> mock(org.flexlb.transport.GeneralHttpNettyService.class));
            context.registerBean(LBStatusConsistencyService.class, () -> mock(LBStatusConsistencyService.class));
            context.register(ModelMetaConfig.class, ConstraintTreeBootstrapRegistry.class,
                    WhaleConstraintTreePublisher.class, ConstraintTreeBuildService.class,
                    ConstraintTreeBootstrapService.class, ConstraintTreeBootstrapServer.class);
            context.refresh();
            assertNotNull(context.getBean("constraintTreeBootstrapRoutes"));
            assertTrue(context.getBean(ConstraintTreeBootstrapRegistry.class).pendingModels().isEmpty());
        }
    }

    @Test
    void leaseRenewalExpiryDedupAndDiscoveryHandoffAreModelScoped() {
        var clock = new AtomicLong();
        var registry = new ConstraintTreeBootstrapRegistry(clock::get, Duration.ofSeconds(2));
        var uri = URI.create("http://127.0.0.1:23495");
        assertFalse(registry.register("a", uri));
        clock.set(Duration.ofSeconds(1).toNanos());
        assertFalse(registry.register("a", uri));
        clock.set(Duration.ofMillis(2500).toNanos());
        assertEquals(Map.of(uri.getAuthority(), uri), registry.merge("a", Map.of()));
        assertTrue(registry.merge("b", Map.of()).isEmpty());
        assertEquals(1, registry.merge("a", Map.of(uri.getAuthority(), uri)).size());
        assertTrue(registry.register("a", uri));
        assertTrue(registry.pendingModels().isEmpty());
        registry.merge("a", Map.of());
        assertFalse(registry.register("a", uri));
        clock.addAndGet(Duration.ofSeconds(2).toNanos());
        assertTrue(registry.pendingModels().isEmpty());
    }

    @Test
    @SuppressWarnings("unchecked")
    void coldStartTriggersExistingReaderOnceAndFollowerDoesNothing() {
        var registry = new ConstraintTreeBootstrapRegistry();
        registry.register("a", URI.create("http://127.0.0.1:23495"));
        var builds = mock(ConstraintTreeBuildService.class);
        var provider = (ObjectProvider<IgraphConstraintTreePoller>) mock(ObjectProvider.class);
        var poller = mock(IgraphConstraintTreePoller.class);
        when(provider.getIfAvailable()).thenReturn(poller);
        when(poller.getModel()).thenReturn("a");
        when(poller.trigger()).thenReturn(true);
        var leader = mock(LBStatusConsistencyService.class);
        var bootstrap = new ConstraintTreeBootstrapService(registry, builds, provider, leader);
        try {
            bootstrap.tick();
            verifyNoInteractions(poller);
            when(leader.isMaster()).thenReturn(true);
            bootstrap.tick();
            bootstrap.tick();
            verify(poller, times(1)).trigger();
            verify(builds, never()).reconcileCurrent();
        } finally { bootstrap.close(); }
    }

    @Test
    void httpRegistrationUsesPeerAddressAndValidatesConfiguredService() {
        var registry = new ConstraintTreeBootstrapRegistry();
        var models = mock(ModelMetaConfig.class);
        var route = mock(ServiceRoute.class);
        when(models.getServiceIds()).thenReturn(Set.of(IdUtils.getServiceIdByModelName("a"),
                IdUtils.getServiceIdByModelName("b")));
        when(route.getRoleEndpoints(RoleType.PDFUSION)).thenReturn(List.of(new Endpoint()));
        when(models.getServiceRoute(IdUtils.getServiceIdByModelName("a"))).thenReturn(route);
        var leader = mock(LBStatusConsistencyService.class);
        when(leader.isMaster()).thenReturn(true);
        var routes = new ConstraintTreeBootstrapServer(registry, models, leader).constraintTreeBootstrapRoutes();
        var handler = RouterFunctions.toWebHandler(routes);
        String body = "{\"service_id\":\"" + IdUtils.getServiceIdByModelName("a")
                + "\",\"role\":\"PDFUSION\",\"http_port\":23495,\"ip\":\"evil.example\"}";
        var exchange = MockServerWebExchange.from(MockServerHttpRequest.post(ConstraintTreeBootstrapServer.PATH)
                .remoteAddress(new InetSocketAddress("127.0.0.2", 54321))
                .header("Content-Type", "application/json").header("X-Forwarded-For", "127.0.0.3").body(body));
        handler.handle(exchange).block();
        assertEquals(200, exchange.getResponse().getRawStatusCode());
        assertEquals(Map.of("127.0.0.2:23495", URI.create("http://127.0.0.2:23495")), registry.merge("a", Map.of()));
        var client = WebTestClient.bindToRouterFunction(routes).build();
        for (String invalid : List.of("{}", body.replace("23495", "0"), body.replace("PDFUSION", "PREFILL"),
                body.replace(IdUtils.getServiceIdByModelName("a"), "unknown"))) {
            client.post().uri(ConstraintTreeBootstrapServer.PATH).header("Content-Type", "application/json")
                    .bodyValue(invalid).exchange().expectStatus().isBadRequest();
        }
        when(leader.isMaster()).thenReturn(false);
        when(leader.getMasterHostIpPort()).thenReturn("127.0.0.1:8080");
        client.post().uri(ConstraintTreeBootstrapServer.PATH).bodyValue(body).exchange()
                .expectStatus().isTemporaryRedirect().expectHeader()
                .valueEquals("Location", "http://127.0.0.1:8080" + ConstraintTreeBootstrapServer.PATH);
    }
}
