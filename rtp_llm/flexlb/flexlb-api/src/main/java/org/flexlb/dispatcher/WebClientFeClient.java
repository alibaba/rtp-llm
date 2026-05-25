package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;

public class WebClientFeClient implements FeClient {

    private final WebClient webClient;
    private final int timeoutMs;

    public WebClientFeClient(WebClient.Builder builder, int timeoutMs, int maxResponseBytes) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(maxResponseBytes))
                .build();
        this.webClient = builder.exchangeStrategies(strategies).build();
        this.timeoutMs = timeoutMs;
    }

    @Override
    public Mono<JsonNode> post(String feBaseUrl, String fePath, ObjectNode body) {
        return webClient.post()
                .uri(feBaseUrl + fePath)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(JsonNode.class)
                .timeout(Duration.ofMillis(timeoutMs));
    }
}
