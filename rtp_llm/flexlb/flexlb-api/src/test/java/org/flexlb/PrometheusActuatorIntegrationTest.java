package org.flexlb;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.monitor.prometheus.PrometheusFlexMonitor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.actuate.metrics.AutoConfigureMetrics;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@ActiveProfiles("test")
@AutoConfigureMetrics
@ExtendWith(SystemStubsExtension.class)
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT,
        properties = {
                "server.port=0",
                "management.server.port=7002"
        })
class PrometheusActuatorIntegrationTest {

    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    @Autowired
    private FlexMonitor monitor;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @BeforeAll
    static void setRequiredEnvironmentVariables() {
        environmentVariables.set("MODEL_SERVICE_CONFIG", """
                {"service_id":"aigc.text-generation.generation.prometheus_integration_test","role_endpoints":[{"group":"default",
                "pd_fusion_endpoint":{"address":"local-engine","protocol":"http","path":"/",
                "discovery":{"type":"static-env","hosts":["127.0.0.1:8080"]}}}]}
                """);
        environmentVariables.set("HIPPO_ROLE", "flexlb-test");
        environmentVariables.set("HIPPO_APP", "flexlb-test");
        environmentVariables.set("FLEXLB_CONFIG", "{\"loadBalanceStrategy\":\"SHORTEST_TTFT\"}");
        environmentVariables.set("FLEXLB_SYNC_CONSISTENCY_CONFIG", "{\"needConsistency\":false}");
        environmentVariables.set("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317");
        environmentVariables.set("OTEL_TRACE_SKIP_PATTERN", "/health|/hook/.*");
        environmentVariables.set("FLEXLB_MONITOR_PROVIDER", "prometheus");
    }

    @Test
    void managementServerExposesFlexLbMetricAtPrometheusEndpoint() throws Exception {
        assertThat(monitor).isInstanceOf(PrometheusFlexMonitor.class);

        String metricName = "actuator_integration_gauge_" + UUID.randomUUID().toString().replace("-", "");
        monitor.register(metricName, FlexMetricType.GAUGE, FlexMetricTags.of());
        monitor.report(metricName, 1.0);

        HttpRequest request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:7002/prometheus"))
                .timeout(Duration.ofSeconds(5))
                .GET()
                .build();
        HttpResponse<String> response = HTTP_CLIENT.send(request, HttpResponse.BodyHandlers.ofString());

        assertThat(response.statusCode()).isEqualTo(200);
        assertThat(response.body()).contains("flexlb_" + metricName + " 1.0");
    }
}
