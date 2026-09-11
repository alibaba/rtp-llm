package org.flexlb.dispatcher;

import org.flexlb.dao.pv.DispatchPvLogData;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.http.client.reactive.ClientHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserter;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class PassthroughClient {

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
    private static final Duration DISCARD_RELEASE_TIMEOUT = Duration.ofSeconds(1);

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
     * so connection release is deferred until the body Flux is consumed at {@code writeTo} time
     * (the modern {@code exchangeToMono} would auto-release the body when the mapping function
     * returns, which for a streamed passthrough — the body is written later, not within the
     * function — would drop the response). An explicit response lease owns the narrow handoff from
     * received FE headers to the first body subscription. Cancellation/discard/assembly failure in
     * that window releases the FE body; once body writing starts, its own terminal signal owns the
     * connection. PV is emitted when FE response headers arrive or when an upstream step throws.
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
        AtomicReference<UpstreamResponseLease> pendingLease = new AtomicReference<>();
        return Mono.fromCallable(fePool::next)
                .doOnNext(pv::setFeHost)
                .flatMap(feBaseUrl -> {
                    String pathAndQuery = src.getRawQuery() == null ? fePath : fePath + "?" + src.getRawQuery();
                    URI target = URI.create(feBaseUrl + pathAndQuery);
                    return webClient.method(request.method())
                            .uri(target)
                            .headers(h -> DispatcherHeaders.copyEndToEnd(
                                    request.headers().asHttpHeaders(), h, DispatcherHeaders.TO_FE_SKIP))
                            .body(bodyInserter)
                            .exchange()
                            .timeout(headersTimeout)
                            .flatMap(clientResponse -> {
                                UpstreamResponseLease lease = new UpstreamResponseLease(clientResponse);
                                pendingLease.set(lease);
                                int status = clientResponse.rawStatusCode();
                                Mono<ServerResponse> response;
                                try {
                                    Flux<DataBuffer> leasedBody = Flux.defer(() -> {
                                        if (!lease.tryStartBody()) {
                                            return Flux.empty();
                                        }
                                        return clientResponse.bodyToFlux(DataBuffer.class)
                                                .timeout(Duration.ofMillis(STREAM_TIMEOUT_MS))
                                                .doFinally(lease::bodyTerminated);
                                    });
                                    response = ServerResponse.status(status)
                                            .headers(h -> DispatcherHeaders.copyEndToEnd(
                                                    clientResponse.headers().asHttpHeaders(), h, DispatcherHeaders.HOP_BY_HOP))
                                            .body(BodyInserters.fromDataBuffers(
                                                    leasedBody))
                                            .<ServerResponse>map(
                                                    delegate -> new UpstreamOwnedResponse(delegate, lease))
                                            .doOnError(ignored -> lease.releaseIfUnwritten())
                                            .doOnCancel(lease::releaseIfUnwritten);
                                } catch (RuntimeException assemblyFailure) {
                                    // Headers arrived but assembling the passthrough response threw before it was
                                    // handed to a self-releasing consumer; release the FE body now or the pooled
                                    // connection leaks. The shared 502 envelope below answers the caller.
                                    lease.releaseIfUnwritten();
                                    throw assemblyFailure;
                                }
                                // response is built and will be returned (then subscribed), so its doOnCancel /
                                // body consumption now owns FE-body release. PV/metrics are best-effort
                                // book-keeping: swallow a failure here (log WARN) instead of rethrowing.
                                // Rethrowing would fall into .doOnError and re-emit PV/metrics as a 502 — a
                                // double count — even though the caller still gets this good response.
                                try {
                                    pv.finish(status, null);
                                    pv.emit();
                                    metricsReporter.reportRequest("passthrough", metricPathTag(fePath),
                                            status, pv.getCostMs());
                                } catch (RuntimeException bookkeepingFailure) {
                                    Logger.warn("passthrough PV/metrics emit failed, response still returned: {}",
                                            bookkeepingFailure.toString());
                                }
                                return response;
                            });
                })
                .doOnCancel(() -> {
                    UpstreamResponseLease lease = pendingLease.get();
                    if (lease != null) {
                        lease.releaseIfUnwritten();
                    }
                })
                .doOnDiscard(UpstreamOwnedResponse.class, UpstreamOwnedResponse::discard)
                .doOnError(e -> {
                    String reason = DispatcherResponses.briefReason(e);
                    Logger.warn("passthrough forward failed: path={}, feHost={}, err={}",
                            fePath, pv.getFeHost(), reason);
                    pv.finish(502, reason);
                    pv.emit();
                    metricsReporter.reportRequest("passthrough", metricPathTag(fePath), 502, pv.getCostMs());
                })
                // Stable, non-revealing text: the exception message can carry the FE address the
                // client has no business learning. Full reason is in the WARN above and pv.log.
                .onErrorResume(e -> DispatcherResponses.error(
                        502, "passthrough_failed", "upstream request failed"));
    }

    /** Called by the router when an emitted response is discarded before {@code writeTo}. */
    static void releaseIfUnwritten(ServerResponse response) {
        if (response instanceof UpstreamOwnedResponse owned) {
            owned.discard();
        }
    }

    /**
     * Owns the FE body only until WebFlux subscribes it. State transitions are one-way, making
     * overlapping cancel/discard/write signals harmless and ensuring releaseBody is subscribed at
     * most once.
     */
    private static final class UpstreamResponseLease {
        private static final int PENDING = 0;
        private static final int BODY_STARTED = 1;
        private static final int TERMINAL = 2;

        private final ClientResponse response;
        private final AtomicInteger state = new AtomicInteger(PENDING);

        UpstreamResponseLease(ClientResponse response) {
            this.response = response;
        }

        boolean tryStartBody() {
            return state.compareAndSet(PENDING, BODY_STARTED);
        }

        void bodyTerminated(reactor.core.publisher.SignalType ignored) {
            state.compareAndSet(BODY_STARTED, TERMINAL);
        }

        void releaseIfUnwritten() {
            if (!state.compareAndSet(PENDING, TERMINAL)) {
                return;
            }
            response.releaseBody().timeout(DISCARD_RELEASE_TIMEOUT).subscribe(
                    ignored -> { },
                    error -> Logger.warn("failed to release discarded passthrough body: {}",
                            DispatcherResponses.briefReason(error)));
        }
    }

    /** ServerResponse wrapper that keeps the FE-body lease attached through the write boundary. */
    private static final class UpstreamOwnedResponse implements ServerResponse {
        private final ServerResponse delegate;
        private final UpstreamResponseLease lease;

        UpstreamOwnedResponse(ServerResponse delegate, UpstreamResponseLease lease) {
            this.delegate = delegate;
            this.lease = lease;
        }

        void discard() {
            lease.releaseIfUnwritten();
        }

        @Override
        public HttpStatus statusCode() {
            return delegate.statusCode();
        }

        @Override
        public int rawStatusCode() {
            return delegate.rawStatusCode();
        }

        @Override
        public HttpHeaders headers() {
            return delegate.headers();
        }

        @Override
        public MultiValueMap<String, ResponseCookie> cookies() {
            return delegate.cookies();
        }

        @Override
        public Mono<Void> writeTo(ServerWebExchange exchange, Context context) {
            return Mono.defer(() -> delegate.writeTo(exchange, context))
                    .doFinally(ignored -> lease.releaseIfUnwritten());
        }
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

}
