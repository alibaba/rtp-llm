package org.flexlb;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.cases.QueueStressTest;
import org.flexlb.cases.RequestCancelTest;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.RouteService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.reactive.server.WebTestClient;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.util.Map;

/**
 * Integration test with reusable Spring context
 */
@Slf4j
@ActiveProfiles("test")
@AutoConfigureMockMvc
@AutoConfigureWebTestClient
@ExtendWith({SystemStubsExtension.class})
@SpringBootTest(args = {"--server.port=7001", "--management.server.port=8803"},
        webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
public class ReuseSpringContextIntegrationTest {

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @SpyBean
    private ConfigService configService;
    @SpyBean
    private RouteService routeService;
    @Autowired
    private QueueManager queueManager;
    @Autowired
    private WebTestClient webTestClient;

    //========================= Integration Test Class ======================//

    @BeforeAll
    public static void setUp() {
        environmentVariables.set("MODEL_SERVICE_CONFIG",
                """
                        {
                            "service_id": "aigc.text-generation.generation.engine_service",\s
                            "role_endpoints": [
                                {
                                    "group": "nm     125_L20X_2TP",\s
                                    "prefill_endpoint": {
                                        "address": "com.prefill.hosts.address",\s
                                        "protocol": "http",\s
                                        "path": "/"
                                    },\s
                                    "decode_endpoint": {
                                        "address": "com.decode.hosts.address",\s
                                        "protocol": "http",\s
                                        "path": "/"
                                    }
                                }
                            ]
                        }
                """
        );
        environmentVariables.set("HIPPO_ROLE", "TEST_HIPPO_ROLE");
        environmentVariables.set("OTEL_EXPORTER_OTLP_ENDPOINT", "http://search-uniagent-trace-na61.vip.tbsite.net:4317");
    }

    private WebTestClient createWebClient() {
        return webTestClient.mutate()
                .baseUrl("http://localhost:7001")
                .build();
    }

    @Test
    @DisplayName("Request cancellation test")
    public void requestCancelTest() {
        RequestCancelTest.init(environmentVariables, configService, routeService).run();
    }

    @Test
    @DisplayName("Queue full rejection test")
    public void queueFullRejectionTest() {
        QueueStressTest.init(createWebClient(), environmentVariables, configService)
                .resetQueue(queueManager, 10)
                .testQueueFullRejection();
    }

    @Test
    @DisplayName("Concurrent enqueue thread safety test")
    public void concurrentEnqueueTest() {
        QueueStressTest.init(createWebClient(), environmentVariables, configService)
                .resetQueue(queueManager, 500)
                .testConcurrentEnqueue();
    }

    @Test
    @DisplayName("select_workers: requestId=0 returns INVALID_REQUEST")
    public void selectWorkersInvalidRequestTest() {
        String body = "{\"role\":\"PDFUSION\",\"count\":4,\"request_id\":0,\"request_time_ms\":0}";
        createWebClient().post().uri("/rtp_llm/select_workers")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.code").isEqualTo(8406);
    }

    @Test
    @DisplayName("select_workers: empty pool returns NO_PDFUSION_WORKER")
    public void selectWorkersNoWorkerTest() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        String body = "{\"role\":\"PDFUSION\",\"count\":4,\"request_id\":1,\"request_time_ms\":1}";
        createWebClient().post().uri("/rtp_llm/select_workers")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(false)
                .jsonPath("$.code").isEqualTo(8404);
    }

    @Test
    @DisplayName("select_workers: returns 4 of 8 healthy workers")
    public void selectWorkersSuccessTest() {
        Map<String, WorkerStatus> pool =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap();
        pool.clear();
        try {
            for (int i = 1; i <= 8; i++) {
                WorkerStatus w = new WorkerStatus();
                w.setIp("10.0.0." + i);
                w.setPort(28100);
                w.setAlive(true);
                w.setAvailableConcurrency((long) i);
                pool.put("10.0.0." + i + ":28100", w);
            }
            String body = "{\"role\":\"PDFUSION\",\"count\":4,\"request_id\":2,\"request_time_ms\":2}";
            createWebClient().post().uri("/rtp_llm/select_workers")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .exchange()
                    .expectStatus().isOk()
                    .expectBody()
                    .jsonPath("$.success").isEqualTo(true)
                    .jsonPath("$.code").isEqualTo(200)
                    .jsonPath("$.server_status.length()").isEqualTo(4)
                    .jsonPath("$.total_workers").isEqualTo(8)
                    .jsonPath("$.ttl_ms").isEqualTo(1000)
                    .jsonPath("$.version").exists();
        } finally {
            pool.clear();
        }
    }
}
