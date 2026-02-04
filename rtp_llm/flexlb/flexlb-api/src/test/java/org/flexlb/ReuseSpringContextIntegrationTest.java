package org.flexlb;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.cases.QueueStressTest;
import org.flexlb.cases.RequestCancelTest;
import org.flexlb.config.ConfigService;
import org.flexlb.service.RouteService;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.reactive.server.WebTestClient;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

/**
 * 复用Spring上下文的集成测试
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

    //========================= 集成测试类 ======================//

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
    @DisplayName("请求取消测试")
    public void requestCancelTest() {
        RequestCancelTest.init(environmentVariables, configService, routeService).run();
    }

    @Test
    @DisplayName("队列满载拒绝测试")
    public void queueFullRejectionTest() {
        QueueStressTest.init(createWebClient(), environmentVariables, configService)
                .resetQueue(queueManager, 10)
                .testQueueFullRejection();
    }

    @Test
    @DisplayName("并发入队线程安全测试")
    public void concurrentEnqueueTest() {
        QueueStressTest.init(createWebClient(), environmentVariables, configService)
                .resetQueue(queueManager, 500)
                .testConcurrentEnqueue();
    }
}
