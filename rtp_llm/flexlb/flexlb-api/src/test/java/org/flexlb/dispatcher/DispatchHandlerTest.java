package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.same;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DispatchHandlerTest {

    @Test
    void handlePassthroughDelegatesToClient() {
        PassthroughClient passthrough = mock(PassthroughClient.class);
        ServerResponse stubResponse = mock(ServerResponse.class);
        ServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .body(Mono.empty());
        when(passthrough.forward(same(request))).thenReturn(Mono.just(stubResponse));

        DispatchHandler handler = new DispatchHandler(passthrough);

        StepVerifier.create(handler.handlePassthrough(request))
                .assertNext(r -> assertSame(stubResponse, r))
                .verifyComplete();

        verify(passthrough).forward(same(request));
    }
}
