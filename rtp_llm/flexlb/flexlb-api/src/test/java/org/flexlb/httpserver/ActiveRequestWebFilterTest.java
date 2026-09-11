package org.flexlb.httpserver;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ActiveRequestWebFilterTest {
    private final ActiveRequestCounter counter = new ActiveRequestCounter();
    private final ActiveRequestWebFilter filter = new ActiveRequestWebFilter(counter);

    @ParameterizedTest
    @CsvSource({"/rtp_llm/batch_schedule,1", "/dispatcher,1", "/dispatcher/batch_infer,1",
            "/dispatcher/v1/models,1", "/dispatcher/other,1",
            "/rtp_llm/master/info,0", "/health,0"})
    void countsOnlyServingRequestsUntilCompletion(String path, long expected) {
        MockServerWebExchange exchange = MockServerWebExchange.from(MockServerHttpRequest.get(path));
        Sinks.Empty<Void> completed = Sinks.empty();
        Mono<Void> processing = filter.filter(exchange, current -> current.getResponse().writeWith(completed.asMono().thenMany(Flux.empty())));
        assertEquals(0, counter.getCount(), "assembly must not acquire a token");
        StepVerifier.create(processing)
                .then(() -> assertEquals(expected, counter.getCount()))
                .then(completed::tryEmitEmpty)
                .verifyComplete();
        assertEquals(0, counter.getCount());
    }

    @Test
    void countsSlowRequestBodiesAndReleasesOnDisconnect() {
        MockServerWebExchange exchange = MockServerWebExchange.from(
                MockServerHttpRequest.post("/rtp_llm/batch_schedule").body(Flux.never()));
        StepVerifier.create(filter.filter(exchange, current -> current.getRequest().getBody().then()))
                .then(() -> assertEquals(1, counter.getCount()))
                .thenCancel().verify(Duration.ofSeconds(5));
        assertEquals(0, counter.getCount());
    }

    @Test
    void handlerExceptionReleasesToken() {
        MockServerWebExchange exchange = MockServerWebExchange.from(
                MockServerHttpRequest.post("/rtp_llm/batch_schedule"));
        StepVerifier.create(filter.filter(exchange, ignored -> {
                    assertEquals(1, counter.getCount());
                    throw new IllegalStateException("injected handler failure");
                }))
                .expectError(IllegalStateException.class).verify(Duration.ofSeconds(5));
        assertEquals(0, counter.getCount());
    }
}
