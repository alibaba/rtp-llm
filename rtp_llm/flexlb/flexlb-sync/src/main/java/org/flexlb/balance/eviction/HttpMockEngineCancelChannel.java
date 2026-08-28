package org.flexlb.balance.eviction;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;

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
 * <p><b>Production-safety wiring:</b>
 * <ol>
 *   <li>{@link EngineCancelChannelConfiguration} selects this implementation
 *       only when the property
 *       {@code flexlb.test.mock-cancel-control-url} (env
 *       {@code FLEXLB_TEST_MOCK_CANCEL_CONTROL_URL}) is non-blank; otherwise it
 *       selects the production {@link GrpcEngineCancelChannel},</li>
 *   <li>the property name carries the {@code test.mock} prefix,</li>
 *   <li>{@link #logTestOnlyWarning()} shouts at startup,</li>
 *   <li>even if misconfigured, planning still requires the
 *       {@code preemption.allowedVictimStages=DECODE_ENGINE_OWNED} gate (EvictionPlanner
 *       business gate).</li>
 * </ol>
 *
 * <p>Contract mirror of MockEngineCancelChannel: a cancel is an intent
 * injection only — settlement requires original-Prefill WorkerStatus carrying
 * typed {@code CANCELED+8429}. Never throws synchronously; transport failures surface as a
 * failed future, local branches as a completed {@code CancelOutcome}.
 */
@Slf4j
public class HttpMockEngineCancelChannel implements EngineCancelChannel {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Duration REQUEST_TIMEOUT = Duration.ofMillis(500);

    private final String controlUrl;
    private final HttpClient httpClient;

    public HttpMockEngineCancelChannel(String controlUrl) {
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
        log.warn("TEST-ONLY HttpMockEngineCancelChannel is ACTIVE at {}; never enable it in production", controlUrl);
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
                                                   long timeoutMs) {
        try {
            // TEST-ONLY routing: the mock control plane resolves the target
            // engine by the original Prefill endpoint's gRPC port.
            if (target == null || !target.isRoutable()) {
                return CompletableFuture.completedFuture(CancelOutcome.unsupported());
            }
            String body = MAPPER.createObjectNode()
                    .put("port", target.prefillGrpcPort())
                    .put("request_id", requestId)
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
     * Maps the control-plane response onto the engine Cancel contract. A 404
     * means the target engine itself is unsupported; a 200 body carries either
     * ACCEPTED or NOT_FOUND for the specifically addressed Prefill.
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
        try {
            String status = MAPPER.readTree(response.body()).path("status").asText();
            return switch (status) {
                case "ACCEPTED" -> CancelOutcome.accepted();
                case "NOT_FOUND" -> CancelOutcome.notFound();
                default -> throw new IllegalStateException(
                        "mock cancel control plane returned unknown status '" + status
                                + "' for request " + requestId);
            };
        } catch (Exception e) {
            if (e instanceof IllegalStateException illegalStateException) {
                throw illegalStateException;
            }
            throw new IllegalStateException(
                    "mock cancel control plane returned invalid JSON for request " + requestId, e);
        }
    }
}
