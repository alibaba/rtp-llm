package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DispatchRouterTest {

    @Test
    void nonDispatcherPathsAreNotMatched() {
        DispatchHandler handler = mock(DispatchHandler.class);
        RouterFunction<ServerResponse> routes = new DispatchRouter(handler).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
        client.get().uri("/health").exchange().expectStatus().isNotFound();
        client.post().uri("/chat/completions").bodyValue("{}").exchange().expectStatus().isNotFound();
        verifyNoInteractions(handler);
    }

    @Test
    void dispatcherBatchInferGoesToHandleBatch() {
        DispatchHandler handler = mock(DispatchHandler.class);
        when(handler.handleBatch(any()))
                .thenReturn(ServerResponse.ok().bodyValue("batch-handled"));
        WebTestClient client = WebTestClient.bindToRouterFunction(new DispatchRouter(handler).routes()).build();

        client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("batch-handled");
        verify(handler).handleBatch(any());
    }

    @Test
    void dispatcherAnyOtherPathGoesToPassthrough() {
        DispatchHandler handler = mock(DispatchHandler.class);
        when(handler.handlePassthrough(any()))
                .thenReturn(ServerResponse.ok().bodyValue("pass"));
        WebTestClient client = WebTestClient.bindToRouterFunction(new DispatchRouter(handler).routes()).build();

        client.get().uri("/dispatcher/v1/models").exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("pass");
        verify(handler).handlePassthrough(any());
    }
}
