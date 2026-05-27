package org.flexlb.dispatcher;

import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.Locale;
import java.util.Set;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "enabled", havingValue = "true")
public class PassthroughClient {

    /** Per-request access log, shared with {@code /schedule} / {@code /batch_schedule} → pv.log. */
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    /**
     * Hop-by-hop headers from RFC 7230 §6.1 plus framing headers WebClient must compute itself
     * for the outbound connection. Forwarding any of these from the inbound request — or back on
     * the response — corrupts the new connection: an inbound {@code Transfer-Encoding: chunked}
     * double-frames the body WebClient is already about to chunk-encode; an inbound {@code Host}
     * routes to whatever the original client put there; {@code Proxy-Authorization} would be
     * relayed downstream against the original intent. Comparison is case-insensitive.
     */
    private static final Set<String> HOP_BY_HOP_LOWERCASE = Set.of(
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "host",
            "content-length");

    /**
     * SSE body-stream total duration cap. Streaming responses (e.g. {@code /v1/chat/completions}
     * with {@code stream=true}) tend to go straight to FE in production — the value-add of routing
     * them through the dispatcher (split + fanout) doesn't apply to a 1:1 stream. We still
     * passthrough SSE so it's not broken when it happens, just bound the body Flux to 10min as a
     * safety net against hung streams holding the connection pool indefinitely. Not exposed as
     * config: extreme long-stream workloads should bypass the dispatcher rather than tune this.
     */
    private static final int STREAM_TIMEOUT_MS = 600_000;

    private final WebClient webClient;
    private final FePool fePool;

    public PassthroughClient(@Qualifier("dispatcherPassthroughWebClient") WebClient webClient,
                             FePool fePool) {
        this.webClient = webClient;
        this.fePool = fePool;
    }

    /**
     * Uses the deprecated {@link WebClient.RequestHeadersSpec#exchange() exchange()} rather than
     * {@code exchangeToMono} because the upstream body {@link Flux} is handed to a
     * {@link ServerResponse} that subscribes lazily (at {@code writeTo} time);
     * {@code exchangeToMono} would release the upstream connection the instant its lambda
     * {@code Mono} completes — which is immediate, since {@code .body(BodyInserter)} is
     * synchronous — leaving nothing left to read by the time the downstream subscribes.
     * {@code exchange()} defers connection release until the body is consumed, which matches the
     * passthrough's intended lifetime.
     *
     * <p>With {@code .exchange()} the caller owns body consumption. The {@code doOnCancel}
     * covers the narrow assembly window — between {@code exchange()} resolving and the
     * downstream subscribing to {@code writeTo}'s body Flux — when a cancel would otherwise
     * leave the channel checked out until reactor-netty's idle reaper closed it. Mid-body
     * cancellation (the SSE-disconnect case) is handled implicitly by
     * {@code clientResponse.bodyToFlux}'s natural cancel propagation; this hook is only for
     * the headers-received-but-body-not-yet-subscribed gap.
     *
     * <p>PV emission fires the instant FE response headers are seen (success path) or the instant
     * any upstream step throws (pool pick, connect, response-timeout); body-stream errors during
     * SSE consumption are intentionally not captured — there's no meaningful "request time" for a
     * 10-minute stream and the response was already handed off downstream.
     */
    @SuppressWarnings("deprecation")
    public Mono<ServerResponse> forward(ServerRequest request) {
        URI src = request.uri();
        String fePath = src.getRawPath().startsWith("/dispatcher/")
                ? src.getRawPath().substring("/dispatcher".length())
                : src.getRawPath();
        DispatchPvLogData pv = DispatchPvLogData.passthrough(fePath, System.currentTimeMillis());
        return Mono.fromCallable(fePool::next)
                .doOnNext(pv::setFeHost)
                .flatMap(feBaseUrl -> {
                    String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
                    URI target = URI.create(feBaseUrl + pathAndQuery);
                    Flux<DataBuffer> bodyStream = request.bodyToFlux(DataBuffer.class);
                    return webClient.method(request.method())
                            .uri(target)
                            .headers(h -> copyEndToEndHeaders(request.headers().asHttpHeaders(), h))
                            .body(BodyInserters.fromDataBuffers(bodyStream))
                            .exchange()
                            .flatMap(clientResponse -> {
                                pv.finish(clientResponse.statusCode().value(), null);
                                pv.emit(pvLogger);
                                return ServerResponse.status(clientResponse.statusCode())
                                        .headers(h -> copyEndToEndHeaders(clientResponse.headers().asHttpHeaders(), h))
                                        .body(BodyInserters.fromDataBuffers(
                                                clientResponse.bodyToFlux(DataBuffer.class)
                                                        .timeout(Duration.ofMillis(STREAM_TIMEOUT_MS))))
                                        .doOnCancel(() -> clientResponse.releaseBody().subscribe());
                            });
                })
                .doOnError(e -> {
                    Logger.warn("passthrough forward failed: path={}, feHost={}, err={}",
                            fePath, pv.getFeHost(),
                            e.getClass().getSimpleName() + ": " + e.getMessage());
                    pv.finish(500, e.getClass().getSimpleName() + ": " + e.getMessage());
                    pv.emit(pvLogger);
                });
    }

    private static void copyEndToEndHeaders(HttpHeaders source, HttpHeaders sink) {
        source.forEach((name, values) -> {
            if (!HOP_BY_HOP_LOWERCASE.contains(name.toLowerCase(Locale.ROOT))) {
                sink.addAll(name, values);
            }
        });
    }
}
