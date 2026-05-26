package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

public class WebClientFeClient implements FeClient {

    /**
     * Max in-memory size for a single FE sub-batch response. Per-chunk, not aggregate — the
     * dispatcher's final response to the client is N chunks × this value (bounded by heap, not
     * a config). 16MB covers extreme "long generation × large batch" workloads with margin while
     * staying well below typical 8-16GB heap allocations even under peak concurrency.
     */
    private static final int MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final WebClient webClient;

    public WebClientFeClient(WebClient.Builder builder) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(MAX_RESPONSE_BYTES))
                .build();
        this.webClient = builder.exchangeStrategies(strategies).build();
    }

    @Override
    public Mono<JsonNode> post(String feBaseUrl, String fePath, ObjectNode body) {
        return webClient.post()
                .uri(feBaseUrl + fePath)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(JsonNode.class);
    }
}
