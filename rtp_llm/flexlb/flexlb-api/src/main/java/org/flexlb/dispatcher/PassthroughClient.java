package org.flexlb.dispatcher;

import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpHeaders;
import org.springframework.http.client.reactive.ClientHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.BodyInserter;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.Set;
import java.util.TreeSet;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class PassthroughClient {

    /**
     * Hop-by-hop headers from RFC 7230 §6.1 plus framing headers WebClient must compute itself
     * for the outbound connection. Forwarding any of these from the inbound request — or back on
     * the response — corrupts the new connection: an inbound {@code Transfer-Encoding: chunked}
     * double-frames the body WebClient is already about to chunk-encode; an inbound {@code Host}
     * routes to whatever the original client put there; {@code Proxy-Authorization} would be
     * relayed downstream against the original intent. Comparison is case-insensitive.
     */
    private static final Set<String> HOP_BY_HOP = caseInsensitiveSet(
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
     * SSE body-stream inactivity cap: {@code Flux#timeout(Duration)} bounds the gap between
     * consecutive emissions, not total stream duration, so a healthy long stream that keeps
     * emitting is never cut while a stream that goes silent for 10min is. Streaming responses
     * (e.g. {@code /v1/chat/completions} with {@code stream=true}) tend to go straight to FE in
     * production — the value-add of routing them through the dispatcher (split + fanout) doesn't
     * apply to a 1:1 stream. We still passthrough SSE so it's not broken when it happens; this is
     * the safety net against hung streams holding the connection pool indefinitely. Not exposed as
     * config: extreme long-stream workloads should bypass the dispatcher rather than tune this.
     */
    private static final int STREAM_TIMEOUT_MS = 600_000;

    private final WebClient webClient;
    private final FePool fePool;
    private final DispatcherMetricsReporter metricsReporter;
    /**
     * Headers-phase cap for the forward. The underlying HttpClient deliberately has no
     * {@code responseTimeout} (it would also police inter-chunk gaps and kill healthy SSE
     * streams — pinned by PassthroughClientTest), but the wait for response <em>headers</em>
     * must still be bounded: an OOM'd FE that keeps its port open and never responds would
     * otherwise pin the request, its pooled connection, and its graceful-drain token forever.
     * The body stream is unaffected — it is bounded separately by {@link #STREAM_TIMEOUT_MS}.
     */
    private final Duration headersTimeout;

    public PassthroughClient(@Qualifier("dispatcherPassthroughWebClient") WebClient webClient,
                             FePool fePool,
                             DispatcherMetricsReporter metricsReporter,
                             DispatchConfig cfg) {
        this.webClient = webClient;
        this.fePool = fePool;
        this.metricsReporter = metricsReporter;
        this.headersTimeout = Duration.ofMillis(cfg.getBatchTimeoutMs());
    }

    /**
     * Forwards the request to one FE verbatim and streams the response back. Uses {@code exchange()}
     * so connection release is deferred until the body Flux is consumed at {@code writeTo} time;
     * {@code doOnCancel} releases the channel if the request is cancelled before the body is
     * subscribed. PV is emitted when FE response headers arrive or when an upstream step throws.
     *
     * <p>Upstream failures surface to the client as a 502 with the same {@code {error, message}}
     * JSON envelope the batch path uses, so callers parse one error shape regardless of which
     * dispatcher path handled them.
     */
    public Mono<ServerResponse> forward(ServerRequest request) {
        return forwardInternal(request,
                BodyInserters.fromDataBuffers(request.bodyToFlux(DataBuffer.class)));
    }

    /**
     * Variant for callers that already drained the request body (the batch handler sniffs the
     * body shape before deciding batch-vs-passthrough): forwards the captured bytes, since the
     * request's own stream can no longer be read.
     */
    public Mono<ServerResponse> forward(ServerRequest request, byte[] body) {
        return forwardInternal(request, BodyInserters.fromValue(body));
    }

    @SuppressWarnings("deprecation")
    private Mono<ServerResponse> forwardInternal(ServerRequest request,
                                                 BodyInserter<?, ? super ClientHttpRequest> bodyInserter) {
        URI src = request.uri();
        String rawPath = src.getRawPath();
        String fePath = normalizeFePath(rawPath);
        DispatchPvLogData pv = DispatchPvLogData.passthrough(fePath, System.currentTimeMillis());
        return Mono.fromCallable(fePool::next)
                .doOnNext(pv::setFeHost)
                .flatMap(feBaseUrl -> {
                    String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
                    URI target = URI.create(feBaseUrl + pathAndQuery);
                    return webClient.method(request.method())
                            .uri(target)
                            .headers(h -> copyEndToEndHeaders(request.headers().asHttpHeaders(), h))
                            .body(bodyInserter)
                            .exchange()
                            .timeout(headersTimeout)
                            .flatMap(clientResponse -> {
                                int status = clientResponse.rawStatusCode();
                                pv.finish(status, null);
                                pv.emit();
                                metricsReporter.reportRequest("passthrough", metricPathTag(fePath),
                                        status, pv.getCostMs());
                                return ServerResponse.status(status)
                                        .headers(h -> copyEndToEndHeaders(clientResponse.headers().asHttpHeaders(), h))
                                        .body(BodyInserters.fromDataBuffers(
                                                clientResponse.bodyToFlux(DataBuffer.class)
                                                        .timeout(Duration.ofMillis(STREAM_TIMEOUT_MS))))
                                        .doOnCancel(() -> clientResponse.releaseBody().subscribe());
                            });
                })
                .doOnError(e -> {
                    String reason = DispatcherResponses.briefReason(e);
                    Logger.warn("passthrough forward failed: path={}, feHost={}, err={}",
                            fePath, pv.getFeHost(), reason);
                    pv.finish(502, reason);
                    pv.emit();
                    metricsReporter.reportRequest("passthrough", metricPathTag(fePath), 502, pv.getCostMs());
                })
                .onErrorResume(e -> DispatcherResponses.error(
                        502, "passthrough_failed", String.valueOf(e.getMessage())));
    }

    /**
     * Strips the {@code /dispatcher} mount prefix. Bare {@code /dispatcher} (no trailing slash)
     * normalizes to the FE root path {@code /} — without this it would be forwarded verbatim and
     * 404 at FE with no hint that the caller merely dropped the slash.
     */
    private static String normalizeFePath(String rawPath) {
        if (rawPath.startsWith("/dispatcher/")) {
            return rawPath.substring("/dispatcher".length());
        }
        return rawPath.equals("/dispatcher") ? "/" : rawPath;
    }

    /**
     * The metric {@code path} tag stays bounded to the registered spec paths; everything else —
     * the catch-all passthrough accepts arbitrary client URIs — collapses to {@code other} so a
     * scanner or typo'd path cannot mint unbounded kmonitor tag values. pv.log keeps the full path.
     */
    private static String metricPathTag(String fePath) {
        return BatchEndpointSpec.BY_PATH.containsKey(fePath) ? fePath : "other";
    }

    private static void copyEndToEndHeaders(HttpHeaders source, HttpHeaders sink) {
        source.forEach((name, values) -> {
            if (!HOP_BY_HOP.contains(name)) {
                sink.addAll(name, values);
            }
        });
    }

    /** Case-insensitive membership without the per-header {@code toLowerCase} allocation. */
    private static Set<String> caseInsensitiveSet(String... names) {
        Set<String> set = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        set.addAll(java.util.Arrays.asList(names));
        return java.util.Collections.unmodifiableSet(set);
    }
}
