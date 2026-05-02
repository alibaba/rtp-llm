package org.flexlb.httpserver;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * V1-α: verify {@code POST /rtp_llm/cancel} endpoint forwards request_id to
 * {@link RouteService#cancelByRequestId(long)} and returns proper status codes.
 */
class CancelEndpointTest {

    private RouteService routeService;
    private WebTestClient client;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(GeneralHttpNettyService.class),
                routeService,
                mock(LBStatusConsistencyService.class),
                mock(EngineHealthReporter.class),
                mock(QueueManager.class),
                mock(ActiveRequestCounter.class));
        RouterFunction<ServerResponse> router = server.loadBalancePrefill();
        client = WebTestClient.bindToRouterFunction(router).build();
    }

    @Test
    void cancel_with_valid_request_id_returns_200_and_invokes_route_service() {
        Request body = new Request();
        body.setRequestId(42L);

        client.post().uri("/rtp_llm/cancel")
                .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody(Response.class)
                .value(resp -> {
                    org.junit.jupiter.api.Assertions.assertTrue(resp.isSuccess());
                    org.junit.jupiter.api.Assertions.assertEquals(200, resp.getCode());
                });

        verify(routeService, times(1)).cancelByRequestId(eq(42L));
    }

    @Test
    void cancel_with_missing_request_id_returns_400_and_does_not_invoke_route_service() {
        Request body = new Request();
        // requestId defaults to 0, treated as invalid per current protocol

        client.post().uri("/rtp_llm/cancel")
                .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isBadRequest();

        verify(routeService, never()).cancelByRequestId(org.mockito.ArgumentMatchers.anyLong());
    }

    @Test
    void cancel_unknown_request_id_still_returns_200() {
        // RouteService.cancelByRequestId is idempotent (DpBatchScheduler treats unknown
        // ids as a no-op). The HTTP layer must not treat "not found" as an error.
        Request body = new Request();
        body.setRequestId(99999L);

        client.post().uri("/rtp_llm/cancel")
                .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk();

        verify(routeService).cancelByRequestId(99999L);
    }
}
