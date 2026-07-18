package org.flexlb;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.test.context.ActiveProfiles;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

/**
 * Integration test with reusable Spring context.
 * Schedule/cancel tests moved to gRPC layer (FlexlbServiceImpl).
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
}
