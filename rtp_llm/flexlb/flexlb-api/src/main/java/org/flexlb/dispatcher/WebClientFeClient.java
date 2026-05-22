package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;

public class WebClientFeClient implements FeClient {

    /**
     * Hard cap on a single FE response body. Protects same-JVM heap (Master's worker-status map
     * shares the heap) from a runaway / misbehaving FE returning a multi-MB response_batch.
     * 16 MiB is generous for K=5 chunks of normal completions; tune via DispatchConfig if needed.
     */
    public static final int DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final WebClient webClient;
    private final int timeoutMs;

    public WebClientFeClient(WebClient.Builder builder, int timeoutMs, int maxResponseBytes) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(maxResponseBytes))
                .build();
        this.webClient = builder.exchangeStrategies(strategies).build();
        this.timeoutMs = timeoutMs;
    }

    /** Convenience for tests/older call-sites; uses {@link #DEFAULT_MAX_RESPONSE_BYTES}. */
    public WebClientFeClient(WebClient.Builder builder, int timeoutMs) {
        this(builder, timeoutMs, DEFAULT_MAX_RESPONSE_BYTES);
    }

    @Override
    public Mono<JsonNode> postBatch(String feBaseUrl, ObjectNode body) {
        return webClient.post()
                .uri(feBaseUrl + "/batch_infer")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(Duration.ofMillis(timeoutMs));
    }
}
