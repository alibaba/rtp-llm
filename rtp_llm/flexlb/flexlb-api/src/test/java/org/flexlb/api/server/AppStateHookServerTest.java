package org.flexlb.api.server;

import org.flexlb.api.ApplicationLifecycle;
import org.flexlb.worker.status.ServiceMetricsReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.server.HandlerStrategies;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.Optional;

import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks registered lifecycle routes for localhost readiness access and shutdown responses.
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("AppStateHookServer Tests")
class AppStateHookServerTest {

    @Mock private ApplicationLifecycle applicationLifecycle;

    @Mock private ServiceMetricsReporter lifecycleReporter;

    @Mock private ServerRequest serverRequest;

    private AppStateHookServer server;

    @BeforeEach
    void setUp() {
        server = new AppStateHookServer(applicationLifecycle, lifecycleReporter);
    }

    @Test
    @DisplayName("Should return 200 for localhost request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLocalhostRequestAndProcessReady() {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 200 for loopback request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLoopbackRequestAndProcessReady() {
        when(serverRequest.remoteAddress())
                .thenReturn(
                        Optional.of(
                                new InetSocketAddress(InetAddress.getLoopbackAddress(), 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 503 for local request when process is not ready")
    void handleProcessOk_ShouldReturn503_WhenLocalRequestAndProcessNotReady() {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 403 for non-local request")
    void handleProcessOk_ShouldReturn403_WhenNonLocalRequest() {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress("192.168.1.100", 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 403)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 403 when remote address is empty")
    void handleProcessOk_ShouldReturn403_WhenRemoteAddressEmpty() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.empty());

        server.onApplicationReady();

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 403)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 200 for local host address request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLocalHostAddressRequestAndProcessReady()
            throws UnknownHostException {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress(InetAddress.getLocalHost(), 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = routeRequest("/hook/process_ok");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 200 when shutdown completes successfully")
    void handleAppStop_ShouldReturn200_WhenShutdownCompletes() {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));
        when(applicationLifecycle.offline()).thenReturn(true);

        Mono<ServerResponse> result = routeRequest("/hook/pre_stop");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();

        verify(applicationLifecycle).offline();
    }

    @Test
    @DisplayName("Should return 503 when shutdown leaves active requests pending")
    void handleAppStop_ShouldReturn503_WhenShutdownIncomplete() {
        when(serverRequest.remoteAddress())
                .thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));
        when(applicationLifecycle.offline()).thenReturn(false);

        Mono<ServerResponse> result = routeRequest("/hook/pre_stop");

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();

        verify(applicationLifecycle).offline();
    }

    private Mono<ServerResponse> routeRequest(String path) {
        MockServerHttpRequest.BaseBuilder<?> builder = MockServerHttpRequest.get(path);
        serverRequest.remoteAddress().ifPresent(builder::remoteAddress);
        ServerRequest request =
                ServerRequest.create(
                        MockServerWebExchange.from(builder.build()),
                        HandlerStrategies.withDefaults().messageReaders());
        return server.appStateHook().route(request).flatMap(handler -> handler.handle(request));
    }
}
