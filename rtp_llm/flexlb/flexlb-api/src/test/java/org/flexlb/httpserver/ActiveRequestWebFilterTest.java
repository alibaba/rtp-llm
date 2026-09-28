package org.flexlb.httpserver;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.server.reactive.ReactorHttpHandlerAdapter;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.adapter.WebHttpHandlerBuilder;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;
import reactor.netty.http.server.HttpServer;
import reactor.test.StepVerifier;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Timeout(10)
class ActiveRequestWebFilterTest {
    private final ActiveRequestWebFilter filter = new ActiveRequestWebFilter();

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preStopWaitsForStreamingResponseOrDisconnectWithoutWaitingOnItself(boolean disconnect) throws Exception {
        var events = Sinks.many().unicast().<String>onBackpressureBuffer();
        var streaming = new CountDownLatch(1);
        var received = new CountDownLatch(1);
        var stopping = new CountDownLatch(1);
        var handler = WebHttpHandlerBuilder.webHandler(exchange -> {
            String path = exchange.getRequest().getPath().value();
            if (path.equals("/hook/pre_stop")) {
                return Mono.fromRunnable(() -> {
                    stopping.countDown();
                    filter.drain();
                }).subscribeOn(Schedulers.boundedElastic()).then(Mono.defer(exchange.getResponse()::setComplete));
            }
            if (path.equals("/dispatcher/stream")) {
                exchange.getResponse().getHeaders().setContentType(MediaType.TEXT_EVENT_STREAM);
                return exchange.getResponse().writeAndFlushWith(events.asFlux().map(event -> Mono.just(
                        exchange.getResponse().bufferFactory().wrap(("data: " + event + "\n\n")
                                .getBytes(StandardCharsets.UTF_8))))).doOnSubscribe(ignored -> streaming.countDown());
            }
            return exchange.getResponse().setComplete();
        }).filter(filter).build();
        var server = HttpServer.create().host("127.0.0.1").port(0)
                .handle(new ReactorHttpHandlerAdapter(handler)).bindNow();
        var client = WebClient.create("http://127.0.0.1:" + server.port());
        var stream = client.get().uri("/dispatcher/stream").retrieve().bodyToFlux(String.class)
                .doOnNext(ignored -> received.countDown()).subscribe();
        try {
            assertTrue(streaming.await(2, TimeUnit.SECONDS));
            assertEquals(Sinks.EmitResult.OK, events.tryEmitNext("first"));
            assertTrue(received.await(2, TimeUnit.SECONDS));
            var stop = client.get().uri("/hook/pre_stop").retrieve().toBodilessEntity().toFuture();
            assertTrue(stopping.await(2, TimeUnit.SECONDS));
            assertThrows(TimeoutException.class, () -> stop.get(100, TimeUnit.MILLISECONDS));
            Integer rejected = Flux.interval(Duration.ofMillis(10))
                    .concatMap(ignored -> client.get().uri("/dispatcher/batch_infer")
                            .exchangeToMono(response -> response.releaseBody().thenReturn(response.rawStatusCode())))
                    .filter(status -> status == 503).next().block(Duration.ofSeconds(2));
            assertEquals(503, rejected);
            if (disconnect) {
                stream.dispose();
            } else {
                assertEquals(Sinks.EmitResult.OK, events.tryEmitComplete());
            }
            assertEquals(HttpStatus.OK, stop.get(2, TimeUnit.SECONDS).getStatusCode());
        } finally {
            events.tryEmitComplete();
            stream.dispose();
            server.disposeNow();
        }
    }

    @ParameterizedTest
    @CsvSource({"/rtp_llm/batch_schedule,true", "/dispatcher,true", "/dispatcher/batch_infer,true",
            "/dispatcher/v1/models,true", "/dispatcher/other,true", "/rtp_llm/master/info,false",
            "/health,false", "/hook/pre_stop,false", "/dispatcher/_dryrun,false", "/dispatcher/_dryrun/batch_infer,false"})
    void drainsOnlyServingExchangesUntilTheirResponseCompletes(String path, boolean serving) throws Exception {
        var exchange = MockServerWebExchange.from(MockServerHttpRequest.get(path));
        Sinks.Empty<Void> completed = Sinks.empty();
        var pending = filter.filter(exchange, current -> current.getResponse()
                .writeWith(completed.asMono().thenMany(Flux.empty()))).toFuture();
        var started = new CountDownLatch(1);
        var draining = CompletableFuture.runAsync(() -> {
            started.countDown();
            filter.drain();
        });
        try {
            assertTrue(started.await(2, TimeUnit.SECONDS));
            if (serving) {
                assertThrows(TimeoutException.class, () -> draining.get(100, TimeUnit.MILLISECONDS));
            } else {
                draining.get(2, TimeUnit.SECONDS);
            }
        } finally {
            completed.tryEmitEmpty();
            pending.get(2, TimeUnit.SECONDS);
            draining.get(2, TimeUnit.SECONDS);
        }
    }

    @Test
    void rejectsNewSubscriptionsAfterDrainWithoutRunningTheirHandler() {
        var exchange = MockServerWebExchange.from(MockServerHttpRequest.post("/dispatcher/batch_infer"));
        var called = new AtomicBoolean();
        Mono<Void> pending = filter.filter(exchange, ignored -> {
            called.set(true);
            return Mono.empty();
        });
        filter.drain();
        StepVerifier.create(pending).verifyComplete();
        assertFalse(called.get());
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, exchange.getResponse().getStatusCode());
        filter.drain();
    }

    @Test
    void slowRequestBodyReleasesOnDisconnect() throws Exception {
        var exchange = MockServerWebExchange.from(
                MockServerHttpRequest.post("/rtp_llm/batch_schedule").body(Flux.never()));
        var pending = filter.filter(exchange, current -> current.getRequest().getBody().then()).subscribe();
        var draining = CompletableFuture.runAsync(filter::drain);
        try {
            assertThrows(TimeoutException.class, () -> draining.get(100, TimeUnit.MILLISECONDS));
        } finally {
            pending.dispose();
            draining.get(2, TimeUnit.SECONDS);
        }
    }

    @Test
    void handlerExceptionDoesNotHoldDrainOpen() throws Exception {
        var exchange = MockServerWebExchange.from(MockServerHttpRequest.post("/dispatcher/batch_infer"));
        StepVerifier.create(filter.filter(exchange, ignored -> {
                    throw new IllegalStateException("injected handler failure");
                })).expectError(IllegalStateException.class).verify(Duration.ofSeconds(2));
        CompletableFuture.runAsync(filter::drain).get(2, TimeUnit.SECONDS);
    }

    @Test
    void hooksAndDryRunRemainAvailableDuringDrain() {
        filter.drain();
        for (String path : new String[]{"/hook/pre_stop", "/hook/process_ok", "/dispatcher/_dryrun/batch_infer"}) {
            var called = new AtomicBoolean();
            var exchange = MockServerWebExchange.from(MockServerHttpRequest.get(path));
            StepVerifier.create(filter.filter(exchange, ignored -> {
                called.set(true);
                return Mono.empty();
            })).verifyComplete();
            assertTrue(called.get(), path);
        }
    }
}
