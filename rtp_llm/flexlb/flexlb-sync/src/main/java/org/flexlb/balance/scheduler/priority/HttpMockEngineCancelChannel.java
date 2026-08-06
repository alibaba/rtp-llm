package org.flexlb.balance.scheduler.priority;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.enums.TaskPhase;
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
 * failed future, protocol branches as a completed {@code CancelOutcome}.
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
    public CompletableFuture<CancelOutcome> cancel(DecodeEndpoint endpoint,
                                                   long requestId,
                                                   CancelReason reason) {
        try {
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
     * Maps the control-plane JSON ({@code {found, phase, already_finished}})
     * onto the {@link CancelOutcome} three-branch contract: found →
     * accepted(phase), already_finished → finishedBeforeCancel, otherwise
     * notFound. A 404 (unknown engine/port) maps to the unsupported branch —
     * the planning gate should have kept us away from that endpoint.
     */
    private CancelOutcome mapResponse(HttpResponse<String> response, long requestId) {
        if (response.statusCode() == 404) {
            return CancelOutcome.unsupportedEndpoint();
        }
        if (response.statusCode() != 200) {
            throw new IllegalStateException("mock cancel control plane returned HTTP "
                    + response.statusCode() + " for request " + requestId
                    + ": " + response.body());
        }
        try {
            JsonNode json = MAPPER.readTree(response.body());
            if (json.path("found").asBoolean(false)) {
                return CancelOutcome.accepted(parsePhase(json.path("phase").asText(null)));
            }
            if (json.path("already_finished").asBoolean(false)) {
                return CancelOutcome.finishedBeforeCancel();
            }
            return CancelOutcome.notFound();
        } catch (Exception e) {
            throw new IllegalStateException("failed to parse mock cancel response for request "
                    + requestId + ": " + response.body(), e);
        }
    }

    /** Engine proto phase names (TASK_PHASE_*) → scheduler {@link TaskPhase}. */
    private static TaskPhase parsePhase(String phase) {
        if (phase == null || phase.isEmpty()) {
            return null;
        }
        return switch (phase) {
            case "TASK_PHASE_PENDING" -> TaskPhase.PENDING;
            case "TASK_PHASE_RECEIVED" -> TaskPhase.RECEIVED;
            case "TASK_PHASE_KV_ALLOCATED" -> TaskPhase.KV_ALLOCATED;
            default -> TaskPhase.RUNNING;
        };
    }
}
