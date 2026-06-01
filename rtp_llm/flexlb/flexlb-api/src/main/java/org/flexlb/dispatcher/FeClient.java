package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.time.Duration;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FeClient {

    /**
     * Max in-memory size for a single FE sub-batch response. Per-chunk, not aggregate — the
     * dispatcher's final response to the client is N chunks × this value (bounded by heap, not
     * a config). 16MB covers extreme "long generation × large batch" workloads with margin while
     * staying well below typical 8-16GB heap allocations even under peak concurrency.
     */
    private static final int MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    /**
     * TCP three-way-handshake timeout for dispatcher → FE batch connections. Aligned with the
     * codebase's sync-side HTTP defaults; same deployment class. Hardcoded because connect
     * timeout is almost never operator-tuned.
     */
    private static final int FE_CONNECT_TIMEOUT_MS = 1000;

    private final WebClient webClient;

    public FeClient(WebClient.Builder builder,
                    @Qualifier("dispatcherFeConnectionProvider") ConnectionProvider provider,
                    DispatchConfig cfg) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(MAX_RESPONSE_BYTES))
                .build();
        HttpClient httpClient = HttpClient.create(provider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, FE_CONNECT_TIMEOUT_MS)
                .responseTimeout(Duration.ofMillis(cfg.getBatchTimeoutMs()));
        this.webClient = builder.clone()
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .exchangeStrategies(strategies)
                .build();
    }

    /**
     * Caller serializes the chunk body with {@code JSON.toJSONBytes} and gets the FE response
     * as raw bytes to parse with {@code JSON.parseObject(byte[])} — no intermediate {@code String}
     * allocation on either edge.
     */
    public Mono<byte[]> postBytes(String feBaseUrl, String fePath, byte[] body) {
        return webClient.post()
                .uri(feBaseUrl + fePath)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(byte[].class);
    }
}
