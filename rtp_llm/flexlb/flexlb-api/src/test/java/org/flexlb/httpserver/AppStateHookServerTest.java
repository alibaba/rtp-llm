package org.flexlb.httpserver;

import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulOnlineService;
import org.flexlb.service.grace.GracefulShutdownService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.Optional;

import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("AppStateHookServer Tests")
class AppStateHookServerTest {

    @Mock
    private GracefulShutdownService gracefulShutdownService;

    @Mock
    private GracefulOnlineService gracefulOnlineService;

    @Mock
    private GracefulLifecycleReporter lifecycleReporter;

    @Mock
    private ServerRequest serverRequest;

    private AppStateHookServer server;

    @BeforeEach
    void setUp() {
        server = new AppStateHookServer(gracefulOnlineService, gracefulShutdownService, lifecycleReporter);
    }

    @Test
    @DisplayName("Should return 200 for localhost request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLocalhostRequestAndProcessReady() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 200 for loopback request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLoopbackRequestAndProcessReady() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.of(new InetSocketAddress(InetAddress.getLoopbackAddress(), 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 503 for local request when process is not ready")
    void handleProcessOk_ShouldReturn503_WhenLocalRequestAndProcessNotReady() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.of(new InetSocketAddress("127.0.0.1", 12345)));

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 503)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 403 for non-local request")
    void handleProcessOk_ShouldReturn403_WhenNonLocalRequest() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.of(new InetSocketAddress("192.168.1.100", 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 403)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 403 when remote address is empty")
    void handleProcessOk_ShouldReturn403_WhenRemoteAddressEmpty() {
        when(serverRequest.remoteAddress()).thenReturn(Optional.empty());

        server.onApplicationReady();

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 403)
                .verifyComplete();
    }

    @Test
    @DisplayName("Should return 200 for local host address request when process is ready")
    void handleProcessOk_ShouldReturn200_WhenLocalHostAddressRequestAndProcessReady() throws UnknownHostException {
        when(serverRequest.remoteAddress()).thenReturn(Optional.of(new InetSocketAddress(InetAddress.getLocalHost(), 12345)));

        server.onApplicationReady();

        Mono<ServerResponse> result = server.handleProcessOk(serverRequest);

        StepVerifier.create(result)
                .expectNextMatches(response -> response.statusCode().value() == 200)
                .verifyComplete();
    }
}