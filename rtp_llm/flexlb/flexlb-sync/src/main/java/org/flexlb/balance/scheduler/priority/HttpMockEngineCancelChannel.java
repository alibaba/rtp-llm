package org.flexlb.balance.scheduler.priority;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * TEST-ONLY {@link EngineCancelChannel} that forwards cancel intents to the
 * Java mock engine cluster's HTTP control plane ({@code POST /cancel_request},
 * see MockControlServer in flexlb-mock-engine). This is the cross-process
 * wiring for accepted-eviction (8429) online evaluation: the master process
 * cannot reach the in-process MockEngineCancelChannel, so the cancel intent
 * travels over the mock control plane instead.
 *
 * <p><b>Production-safety wiring (four lines of defense):</b>
 * <ol>
 *   <li>The bean only exists when the property
 *       {@code flexlb.test.mock-cancel-control-url} (env
 *       {@code FLEXLB_TEST_MOCK_CANCEL_CONTROL_URL}) is set — absent property
 *       means Spring wires exactly today's {@link UnsupportedEngineCancelChannel}
 *       and the {@code @Primary} below never participates in resolution,</li>
 *   <li>the property name carries the {@code test.mock} prefix,</li>
 *   <li>{@link #logTestOnlyWarning()} shouts at startup,</li>
 *   <li>even if misconfigured, planning still requires the
 *       {@code AUTO_TPM_DECODE_ACCEPTED_EVICT_ENABLED} gate (EvictionPlanner
 *       double gate), which stays false in production.</li>
 * </ol>
 *
 * <p>Contract mirror of MockEngineCancelChannel: a cancel is an intent
 * injection only — release confirmation remains the next WorkerStatus report
 * (iron rule 4). Never throws synchronously; transport failures surface as a
 * failed future, local branches as a completed {@code CancelOutcome}.
 */
@Slf4j
@Component
@Primary
@ConditionalOnProperty("flexlb.test.mock-cancel-control-url")
public class HttpMockEngineCancelChannel implements EngineCancelChannel {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Duration REQUEST_TIMEOUT = Duration.ofMillis(500);

    private final String controlUrl;
    private final HttpClient httpClient;

    public HttpMockEngineCancelChannel(
            @Value("${flexlb.test.mock-cancel-control-url}") String controlUrl) {
        // Normalize so both ".../cancel_request"-less base URLs with and
        // without a trailing slash are accepted.
        this.controlUrl = controlUrl.endsWith("/")
                ? controlUrl.substring(0, controlUrl.length() - 1) : controlUrl;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(REQUEST_TIMEOUT)
                .build();
    }

    @PostConstruct
    void logTestOnlyWarning() {
        log.warn("=================================================================");
        log.warn("TEST-ONLY HttpMockEngineCancelChannel is ACTIVE (control URL: {})", controlUrl);
        log.warn("Cancel intents will be sent to the MOCK engine control plane.");
        log.warn("This bean must NEVER be enabled in production deployments.");
        log.warn("=================================================================");
    }

    /**
     * The mock control plane is a single cluster-wide address (110 topology),
     * so a configured URL supports every endpoint: the control server resolves
     * the target engine by gRPC port.
     */
    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return !controlUrl.isEmpty();
    }

    @Override
    public CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                                   long requestId,
                                                   CancelReason reason) {
        try {
            // TEST-ONLY routing: the mock control plane resolves the target
            // engine by the Decode endpoint's gRPC port (110 topology).
            DecodeEndpoint endpoint = target.decodeEndpoint();
            if (endpoint == null) {
                return CompletableFuture.completedFuture(CancelOutcome.unsupported());
            }
            String body = MAPPER.createObjectNode()
                    .put("port", endpoint.getGrpcPort())
                    .put("request_id", requestId)
                    .put("reason", reason == null ? null : reason.name())
                    .toString();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(controlUrl + "/cancel_request"))
                    .timeout(REQUEST_TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                    .thenApply(response -> mapResponse(response, requestId));
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Maps the control-plane HTTP status onto the simplified intent contract:
     * any 200 is an intent registration — accepted() regardless of the JSON
     * body (found / already_finished carry no decision-relevant information
     * anymore). A 404 (unknown engine/port) maps to the unsupported branch —
     * the planning gate should have kept us away from that endpoint.
     */
    private CancelOutcome mapResponse(HttpResponse<String> response, long requestId) {
        if (response.statusCode() == 404) {
            return CancelOutcome.unsupported();
        }
        if (response.statusCode() != 200) {
            throw new IllegalStateException("mock cancel control plane returned HTTP "
                    + response.statusCode() + " for request " + requestId
                    + ": " + response.body());
        }
        return CancelOutcome.accepted();
    }
}
