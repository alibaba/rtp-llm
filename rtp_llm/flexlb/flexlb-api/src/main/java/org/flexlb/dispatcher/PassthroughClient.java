package org.flexlb.dispatcher;

import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.buffer.DataBuffer;
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

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class PassthroughClient {
    private static final Duration STREAM_IDLE_TIMEOUT = Duration.ofMinutes(10);
    private final WebClient webClient;
    private final FePool fePool;
    private final DispatcherMetricsReporter metricsReporter;

    public PassthroughClient(@Qualifier("dispatcherPassthroughWebClient") WebClient webClient,
                             FePool fePool, DispatcherMetricsReporter metricsReporter, DispatchConfig cfg) {
        this.webClient = webClient.mutate()
                .filter((request, next) -> next.exchange(request)
                        .timeout(Duration.ofMillis(cfg.getBatchTimeoutMs())))
                .build();
        this.fePool = fePool;
        this.metricsReporter = metricsReporter;
    }

    public Mono<ServerResponse> forward(ServerRequest request) {
        return forward(request, BodyInserters.fromDataBuffers(request.bodyToFlux(DataBuffer.class)));
    }

    public Mono<ServerResponse> forward(ServerRequest request, byte[] body) {
        return forward(request, BodyInserters.fromValue(body));
    }

    // Acquire and consume the FE response in the same write subscription. exchangeToMono
    // handles release; an unwritten ServerResponse owns no connection or upstream buffers.
    private Mono<ServerResponse> forward(ServerRequest request,
                                         BodyInserter<?, ? super ClientHttpRequest> body) {
        String path = request.uri().getRawPath().substring("/dispatcher".length());
        String fePath = path.isEmpty() ? "/" : path;
        String query = request.uri().getRawQuery();
        long start = System.currentTimeMillis();
        java.util.concurrent.atomic.AtomicBoolean reported = new java.util.concurrent.atomic.AtomicBoolean();
        return ServerResponse.ok().build((exchange, context) -> Mono.fromSupplier(fePool::next)
                        .flatMap(host -> webClient.method(request.method())
                                .uri(URI.create(host + fePath + (query == null ? "" : "?" + query)))
                                .headers(h -> DispatcherHeaders.copyEndToEnd(
                                        request.headers().asHttpHeaders(), h, DispatcherHeaders.TO_FE_SKIP))
                                .body(body).exchangeToMono(response -> {
                    exchange.getResponse().setRawStatusCode(response.rawStatusCode());
                    DispatcherHeaders.copyEndToEnd(response.headers().asHttpHeaders(),
                            exchange.getResponse().getHeaders(), DispatcherHeaders.HOP_BY_HOP);
                    reported.set(true);
                    report(fePath, response.rawStatusCode(), start);
                    return exchange.getResponse().writeWith(response.bodyToFlux(DataBuffer.class)
                            .timeout(STREAM_IDLE_TIMEOUT));
                }))
                .onErrorResume(error -> {
                    if (exchange.getResponse().isCommitted()) {
                        return Mono.error(error);
                    }
                    String reason = error.toString();
                    Logger.warn("passthrough forward failed: path={}, err={}", fePath, reason);
                    if (!reported.get()) {
                        report(fePath, 502, start);
                    }
                    exchange.getResponse().getHeaders().clear();
                    return DispatcherResponses.error(502, "passthrough_failed", "upstream request failed")
                            .flatMap(response -> response.writeTo(exchange, context));
                }));
    }

    private void report(String path, int status, long start) {
        try {
            metricsReporter.reportRequest("passthrough", BatchEndpointSpec.BY_PATH.containsKey(path) ? path : "other",
                    status, System.currentTimeMillis() - start);
        } catch (RuntimeException error) {
            Logger.warn("passthrough metric failed: {}", error.toString());
        }
    }
}
