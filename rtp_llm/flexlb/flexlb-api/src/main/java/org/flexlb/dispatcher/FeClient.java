package org.flexlb.dispatcher;

import io.netty.channel.ChannelOption;
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

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FeClient {

    /** Hard in-memory ceiling for one FE sub-batch response. */
    static final int MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

    private final WebClient webClient;
    private final Duration overallTimeout;
    private final String trustedRoutingToken;

    // Avoid URI template parsing per chunk; bound the cache across discovery changes.
    private static final int URI_CACHE_MAX = 2048;
    private final Map<String, URI> uriCache = new ConcurrentHashMap<>();

    public FeClient(WebClient.Builder builder,
                    @Qualifier("dispatcherFeConnectionProvider") ConnectionProvider provider,
                    DispatchConfig cfg) {
        HttpClient httpClient = HttpClient.create(provider)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, DispatcherConfiguration.FE_CONNECT_TIMEOUT_MS)
                .responseTimeout(Duration.ofMillis(cfg.getBatchTimeoutMs()));
        this.webClient = builder.clone()
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
        this.overallTimeout = Duration.ofMillis(cfg.getBatchTimeoutMs() + cfg.getBodyReadMarginMs());
        this.trustedRoutingToken = cfg.isPreAssignBe() ? cfg.getTrustedRoutingToken() : "";
    }

    /**
     * Reads the FE body incrementally, reserving the shared response budget before each network buffer
     * is copied.
     */
    Mono<byte[]> postBytes(String feBaseUrl, String fePath, byte[] body,
                           HttpHeaders inboundHeaders, String rawQuery,
                           AtomicByteBudget.Reservation reservation) {
        return Mono.defer(() -> {
            AtomicBoolean retained = new AtomicBoolean(false);
            return webClient.post()
                    .uri(resolveUri(feBaseUrl, fePath, rawQuery))
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
                                if (status < 200 || status >= 300) {
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
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        return response.body(BodyExtractors.toDataBuffers())
                .handle((DataBuffer buffer, reactor.core.publisher.SynchronousSink<Integer> sink) -> {
                    try {
                        int readable = buffer.readableByteCount();
                        if (readable > MAX_RESPONSE_BYTES - output.size()) {
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
                        sink.next(readable);
                    } finally {
                        DataBufferUtils.release(buffer);
                    }
                })
                .then(Mono.fromSupplier(output::toByteArray));
    }

    /** Query-less calls (the overwhelmingly common case) hit the memo. */
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
