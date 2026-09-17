package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class HttpLoadBalanceServerTest {
    private GeneralHttpNettyService http;
    private RouteService route;
    private LBStatusConsistencyService consistency;
    private ActiveRequestCounter counter;
    private WebTestClient client;
    private Response response;

    @BeforeEach
    void setUp() {
        http = mock(GeneralHttpNettyService.class);
        route = mock(RouteService.class);
        consistency = mock(LBStatusConsistencyService.class);
        counter = new ActiveRequestCounter();
        var server = new HttpLoadBalanceServer(http, route, consistency, mock(EngineHealthReporter.class),
                mock(QueueManager.class), counter, mock(ConfigService.class));
        client = WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
        response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of());
    }

    private void request() {
        client.post().uri("/rtp_llm/vit/route").contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("request_id", 123, "media_keys", List.of("image")))
                .exchange().expectStatus().isOk().expectBody().jsonPath("$.success").isEqualTo(true);
        assertEquals(0, counter.getCount());
    }

    @Test
    void endpointSetsVitOnlyBeforeRouting() {
        when(route.route(any())).thenReturn(Mono.just(response));
        request();
        var ctx = ArgumentCaptor.forClass(BalanceContext.class);
        verify(route).route(ctx.capture());
        assertTrue(ctx.getValue().getRequest().isVitRouteOnly());
        assertEquals(List.of("image"), ctx.getValue().getRequest().getMediaKeys());
    }

    @Test
    void slaveForwardPreservesVitOnlyFlagOnWire() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(consistency.getMasterHostIpPort()).thenReturn("127.0.0.1:8000");
        when(http.request(any(), any(), eq("/rtp_llm/schedule"), eq(Response.class)))
                .thenReturn(Mono.just(response));
        request();
        var forwarded = ArgumentCaptor.forClass(Request.class);
        verify(http).request(forwarded.capture(), any(), eq("/rtp_llm/schedule"), eq(Response.class));
        assertTrue(forwarded.getValue().isVitRouteOnly());
        assertTrue(JsonUtils.toStringOrEmpty(forwarded.getValue()).contains("\"vit_route_only\":true"));
        verifyNoInteractions(route);
    }
}
