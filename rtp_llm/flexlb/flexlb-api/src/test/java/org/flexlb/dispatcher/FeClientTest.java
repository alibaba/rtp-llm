package org.flexlb.dispatcher;

import io.netty.buffer.PooledByteBufAllocator;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.NettyDataBufferFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.netty.resources.ConnectionProvider;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Timeout(15)
class FeClientTest {
    @Test
    void fanoutHeadersUseDispatcherCredentialAndCorrectWireFraming() throws Exception {
        try (MockWebServer server = new MockWebServer()) {
            server.enqueue(new MockResponse().setBody("{}"));
            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "caller");
            headers.set("Accept-Encoding", "gzip");
            headers.set("Content-Type", "text/plain");
            headers.set(DispatcherHeaders.TRUSTED_ROUTING_HEADER, "forged");
            DispatchConfig cfg = new DispatchConfig();
            cfg.setPreAssignBe(true);
            cfg.setTrustedRoutingToken("secret");
            FeClient client = new FeClient(WebClient.builder(), ConnectionProvider.newConnection(), cfg);
            assertNotNull(client.postBytes(server.url("/").toString().replaceAll("/$", ""), "/batch_infer",
                    "{}".getBytes(), headers, "q=a%2Fb", new AtomicByteBudget(FeClient.MAX_RESPONSE_BYTES).newReservation()).block());
            var sent = server.takeRequest(2, TimeUnit.SECONDS);
            assertEquals("/batch_infer?q=a%2Fb", sent.getPath());
            assertEquals("caller", sent.getHeader("Authorization"));
            assertEquals("secret", sent.getHeader(DispatcherHeaders.TRUSTED_ROUTING_HEADER));
            assertEquals("application/json", sent.getHeader("Content-Type"));
            assertNull(sent.getHeader("Accept-Encoding"));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void headerAndWholeBodyDeadlinesBoundEachCall(boolean trickle) throws Exception {
        try (MockWebServer server = new MockWebServer()) {
            server.enqueue(trickle ? new MockResponse().setBody("x".repeat(64)).throttleBody(1, 30, TimeUnit.MILLISECONDS)
                    : new MockResponse().setHeadersDelay(2, TimeUnit.SECONDS).setBody("{}"));
            ConnectionProvider connections = ConnectionProvider.builder("fe-deadline-test").build();
            try {
                DispatchConfig cfg = new DispatchConfig();
                cfg.setBatchTimeoutMs(300);
                cfg.setBodyReadMarginMs(200);
                FeClient client = new FeClient(WebClient.builder(), connections, cfg);
                StepVerifier.create(client.postBytes(server.url("/").toString(), "/batch_infer",
                        new byte[0], new HttpHeaders(), null, new AtomicByteBudget(FeClient.MAX_RESPONSE_BYTES).newReservation())).expectError().verify(Duration.ofSeconds(3));
            } finally {
                connections.disposeLater().block();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedAndCancelledReadsReleasePooledBuffersAndSharedBudget(boolean cancel) {
        NettyDataBufferFactory factory = new NettyDataBufferFactory(PooledByteBufAllocator.DEFAULT);
        var first = factory.allocateBuffer(4);
        first.write(new byte[4]);
        var second = factory.allocateBuffer(4);
        second.write(new byte[4]);
        Flux<DataBuffer> body = cancel ? Flux.concat(Mono.just(first), Mono.never()) : Flux.just(first, second);
        ClientResponse response = ClientResponse.create(HttpStatus.OK).body(body).build();
        FeClient client = new FeClient(WebClient.builder().exchangeFunction(request -> Mono.just(response)), ConnectionProvider.newConnection(), new DispatchConfig());
        AtomicByteBudget budget = new AtomicByteBudget(5);
        var reservation = budget.newReservation();
        var read = StepVerifier.create(client.postBytes("http://fe", "/", new byte[0], new HttpHeaders(), null, reservation));
        if (cancel) {
            read.thenAwait(Duration.ofMillis(20)).then(() -> assertEquals(4, reservation.bytes()))
                    .thenCancel().verify();
            second.release();
        } else {
            read.expectError(AggregateResponseTooLargeException.class).verify();
            assertEquals(0, second.getNativeBuffer().refCnt());
        }
        assertEquals(0, first.getNativeBuffer().refCnt());
        assertEquals(0, reservation.bytes());
        assertTrue(budget.newReservation().tryReserve(5));
    }
}
