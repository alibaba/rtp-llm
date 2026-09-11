package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.BodyExtractors;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;
import reactor.netty.http.client.HttpClient;
import reactor.netty.resources.ConnectionProvider;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FeClient {

    /**
     * Hard in-memory ceiling for one FE sub-batch response. The parent fanout additionally passes
     * every child a reservation from the configurable shared aggregate budget, so the retained
     * total is bounded by both {@code N * MAX_RESPONSE_BYTES} and that request-level limit.
     */
    static final int MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final WebClient webClient;
    private final Duration overallTimeout;
    private final String trustedRoutingToken;

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

    @Autowired
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
        this.trustedRoutingToken = cfg.isPreAssignBe() ? cfg.getTrustedRoutingToken() : "";
    }

    /** Test seam for exercising streaming body accounting without a real socket. */
    FeClient(WebClient webClient, Duration overallTimeout) {
        this(webClient, overallTimeout, "");
    }

    /** Test seam for the dispatcher-owned routing credential. */
    FeClient(WebClient webClient, Duration overallTimeout, String trustedRoutingToken) {
        this.webClient = webClient;
        this.overallTimeout = overallTimeout;
        this.trustedRoutingToken = trustedRoutingToken == null ? "" : trustedRoutingToken;
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
        return postBytes(feBaseUrl, fePath, body, inboundHeaders, rawQuery,
                new AtomicByteBudget(MAX_RESPONSE_BYTES).newReservation());
    }

    /**
     * Reads the FE body incrementally, reserving the shared response budget before each network
     * buffer is copied. Crossing either cap cancels the response immediately; a failed or
     * cancelled sub-call releases its partial reservation, while a successful byte array retains
     * it until the parent fanout finishes.
     */
    Mono<byte[]> postBytes(String feBaseUrl, String fePath, byte[] body,
                           HttpHeaders inboundHeaders, String rawQuery,
                           AtomicByteBudget.Reservation reservation) {
        return Mono.defer(() -> {
            AtomicBoolean retained = new AtomicBoolean(false);
            return webClient.post()
                    .uri(resolveUri(feBaseUrl, fePath, rawQuery))
                    // End-to-end headers first (Authorization, tenant, tracing — the caller's
                    // request must not lose them just because it took the split path), then the
                    // content type of the chunk body we re-serialized.
                    .headers(h -> {
                        DispatcherHeaders.copyEndToEnd(
                                inboundHeaders, h, DispatcherHeaders.FANOUT_SKIP);
                        if (!trustedRoutingToken.isBlank()) {
                            h.set(DispatcherHeaders.TRUSTED_ROUTING_HEADER, trustedRoutingToken);
                        }
                    })
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .exchangeToMono(response -> readBody(response, reservation)
                            .flatMap(bytes -> {
                                int status = response.rawStatusCode();
                                if (status >= 400) {
                                    HttpStatus resolved = HttpStatus.resolve(status);
                                    String reason = resolved == null
                                            ? "FE response" : resolved.getReasonPhrase();
                                    return Mono.error(WebClientResponseException.create(
                                            status, reason, response.headers().asHttpHeaders(),
                                            bytes, StandardCharsets.UTF_8));
                                }
                                retained.set(true);
                                return Mono.just(bytes);
                            }))
                    .timeout(overallTimeout)
                    .doFinally(ignored -> {
                        if (!retained.get()) {
                            reservation.release();
                        }
                    });
        });
    }

    private Mono<byte[]> readBody(
            org.springframework.web.reactive.function.client.ClientResponse response,
            AtomicByteBudget.Reservation reservation) {
        return Mono.defer(() -> {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            AtomicInteger responseBytes = new AtomicInteger();
            return response.body(BodyExtractors.toDataBuffers())
                    .handle((DataBuffer buffer, reactor.core.publisher.SynchronousSink<Integer> sink) -> {
                        try {
                            int readable = buffer.readableByteCount();
                            int current = responseBytes.get();
                            if (readable > MAX_RESPONSE_BYTES - current) {
                                sink.error(new DataBufferLimitException(
                                        "FE response exceeds " + MAX_RESPONSE_BYTES + " bytes"));
                                return;
                            }
                            if (!reservation.tryReserve(readable)) {
                                sink.error(new AggregateResponseTooLargeException(
                                        reservation.limit()));
                                return;
                            }
                            byte[] chunk = new byte[readable];
                            buffer.read(chunk);
                            output.write(chunk, 0, chunk.length);
                            responseBytes.addAndGet(readable);
                            sink.next(readable);
                        } finally {
                            DataBufferUtils.release(buffer);
                        }
                    })
                    .then(Mono.fromSupplier(output::toByteArray));
        });
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
