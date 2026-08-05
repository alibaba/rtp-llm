package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.service.RouteService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.util.concurrent.ConcurrentHashMap;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Field-level assertions for the /rtp_llm/inflight_status diagnostic endpoint:
 * legacy fields (scheduler_inflight, inflight_batches, inflight_requests) must stay
 * unchanged, while both inflight layers and decode KV reservations are exposed
 * as independent fields per endpoint.
 */
class HttpLoadBalanceServerInflightStatusTest {

    private RouteService routeService;
    private EndpointRegistry endpointRegistry;
    private WebTestClient client;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        endpointRegistry = mock(EndpointRegistry.class);

        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class),
                mock(ConfigService.class),
                routeService,
                endpointRegistry,
                null,
                mock(ServerScheduleLatencyRecorder.class));
        client = WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
    }

    @Test
    void inflightStatus_exposesBothLayersAndKvReservations() {
        when(routeService.globalInflightSize()).thenReturn(4);
        when(routeService.globalInflightTotalSize()).thenReturn(6);

        PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        when(prefill.prefillInflightCount()).thenReturn(2);
        when(prefill.prefillEngineTaskCount()).thenReturn(3);
        when(prefill.prefillEngineWaitingCount()).thenReturn(1);
        when(prefill.prefillEngineRunningCount()).thenReturn(2);
        ConcurrentHashMap<String, PrefillEndpoint> prefillMap = new ConcurrentHashMap<>();
        prefillMap.put("10.0.0.1:8080", prefill);
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(prefillMap);

        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        when(decode.decodeInflightCount()).thenReturn(2);
        when(decode.decodeEngineTaskCount()).thenReturn(3);
        when(decode.decodeTotalLoad()).thenReturn(5);
        when(decode.decodeInflightHardKvReserved()).thenReturn(700L);
        when(decode.decodeInflightExpectedKvReserved()).thenReturn(900L);
        ConcurrentHashMap<String, DecodeEndpoint> decodeMap = new ConcurrentHashMap<>();
        decodeMap.put("10.0.0.2:8080", decode);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(decodeMap);

        client.get().uri("/rtp_llm/inflight_status")
                .accept(org.springframework.http.MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                // Scheduler-level fields (legacy, unchanged)
                .jsonPath("$.scheduler_inflight").isEqualTo(4)
                .jsonPath("$.scheduler_inflight_total").isEqualTo(6)
                // Prefill: legacy sum + per-layer breakdown
                .jsonPath("$.prefill_endpoints[0].ip_port").isEqualTo("10.0.0.1:8080")
                .jsonPath("$.prefill_endpoints[0].inflight_batches").isEqualTo(5)
                .jsonPath("$.prefill_endpoints[0].inflight_entries").isEqualTo(2)
                .jsonPath("$.prefill_endpoints[0].engine_tasks").isEqualTo(3)
                .jsonPath("$.prefill_endpoints[0].engine_waiting").isEqualTo(1)
                .jsonPath("$.prefill_endpoints[0].engine_running").isEqualTo(2)
                // Decode: legacy layer-1 + layer-2 + KV reservations
                .jsonPath("$.decode_endpoints[0].ip_port").isEqualTo("10.0.0.2:8080")
                .jsonPath("$.decode_endpoints[0].inflight_requests").isEqualTo(2)
                .jsonPath("$.decode_endpoints[0].engine_tasks").isEqualTo(3)
                .jsonPath("$.decode_endpoints[0].total_load").isEqualTo(5)
                .jsonPath("$.decode_endpoints[0].kv_reserved_hard").isEqualTo(700)
                .jsonPath("$.decode_endpoints[0].kv_reserved_expected").isEqualTo(900);
    }

    @Test
    void inflightStatus_emptyRegistry_keepsLegacyTopLevelFields() {
        when(routeService.globalInflightSize()).thenReturn(0);
        when(routeService.globalInflightTotalSize()).thenReturn(0);
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());

        client.get().uri("/rtp_llm/inflight_status")
                .accept(org.springframework.http.MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.scheduler_inflight").isEqualTo(0)
                .jsonPath("$.scheduler_inflight_total").isEqualTo(0)
                .jsonPath("$.prefill_endpoints").isEmpty()
                .jsonPath("$.decode_endpoints").isEmpty();
    }
}
