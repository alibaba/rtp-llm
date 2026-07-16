package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.net.URI;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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

    private final WebClient webClient;
    private final Duration overallTimeout;

    /**
     * The live {@code (feBaseUrl, fePath)} pair set is FE pool size × registered spec paths, so
     * resolved URIs are memoized: {@code WebClient.uri(String)} would otherwise run a full template
     * expansion (regex parse → {@code UriComponents} graph → re-parse into {@code URI}) on every
     * chunk of every request. FE churn over a long-lived instance would grow this unbounded, so it
     * is capped: past {@link #URI_CACHE_MAX} distinct keys the cache is dropped and repopulated
     * from the current working set (a handful of re-parses), keeping it genuinely bounded.
     */
    private static final int URI_CACHE_MAX = 2048;
    private final Map<String, URI> uriCache = new ConcurrentHashMap<>();

    public FeClient(WebClient.Builder builder,
                    @Qualifier("dispatcherFeConnectionProvider") ConnectionProvider provider,
                    DispatchConfig cfg) {
        ExchangeStrategies strategies = ExchangeStrategies.builder()
                .codecs(c -> c.defaultCodecs().maxInMemorySize(MAX_RESPONSE_BYTES))
                .build();
        HttpClient httpClient = HttpClient.create(provider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, DispatcherConfiguration.FE_CONNECT_TIMEOUT_MS)
                .responseTimeout(Duration.ofMillis(cfg.getBatchTimeoutMs()));
        this.webClient = builder.clone()
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .exchangeStrategies(strategies)
                .build();
        this.overallTimeout = Duration.ofMillis(cfg.getBatchTimeoutMs() + cfg.getBodyReadMarginMs());
    }

    /**
     * Caller serializes the chunk body with {@code JSON.toJSONBytes} and gets the FE response
     * as raw bytes to parse with {@code JSON.parseObject(byte[])} — no intermediate {@code String}
     * allocation on either edge. The whole call (headers + body) is capped at
     * {@code batchTimeoutMs + bodyReadMarginMs}; a timeout surfaces as a transport failure
     * for that chunk only.
     */
    public Mono<byte[]> postBytes(String feBaseUrl, String fePath, byte[] body,
                                  HttpHeaders inboundHeaders, String rawQuery) {
        return webClient.post()
                .uri(resolveUri(feBaseUrl, fePath, rawQuery))
                // End-to-end headers first (Authorization, tenant, tracing — the caller's request
                // must not lose them just because it took the split path), then the content type of
                // the chunk body we re-serialized, which overrides any inbound value.
                .headers(h -> DispatcherHeaders.copyEndToEnd(inboundHeaders, h, DispatcherHeaders.FANOUT_SKIP))
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(byte[].class)
                .timeout(overallTimeout);
    }

    /**
     * Query-less calls (the overwhelmingly common case) hit the memo. A request that carries a
     * query string is built fresh: query strings vary per request, so caching them would thrash the
     * memo and defeat the bound.
     */
    private URI resolveUri(String feBaseUrl, String fePath, String rawQuery) {
        if (rawQuery != null && !rawQuery.isEmpty()) {
            return URI.create(feBaseUrl + fePath + "?" + rawQuery);
        }
        if (uriCache.size() >= URI_CACHE_MAX) {
            uriCache.clear();
        }
        return uriCache.computeIfAbsent(feBaseUrl + fePath, URI::create);
    }
}
