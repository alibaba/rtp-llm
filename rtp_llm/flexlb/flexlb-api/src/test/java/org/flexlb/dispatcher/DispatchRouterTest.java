package org.flexlb.dispatcher;

import static org.flexlb.dispatcher.BatchEndpointSpec.FailedItemFactory;

import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DispatchRouterTest {

    private static final BatchEndpointSpec BATCH_INFER =
            new BatchEndpointSpec("/batch_infer", "prompt_batch", "response_batch",
                    FailedItemFactory.NULL, null);
    private static final BatchEndpointSpec EMBEDDINGS =
            new BatchEndpointSpec("/v1/embeddings", "input", "data",
                    FailedItemFactory.EMBEDDING_NULL, EmbeddingPostMerger.INSTANCE);

    @Test
    void nonDispatcherPathsAreNotMatched() {
        GenericBatchHandler batch = mock(GenericBatchHandler.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        RouterFunction<ServerResponse> routes =
                new DispatchRouter(batch, passthrough, List.of(BATCH_INFER)).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
        client.get().uri("/health").exchange().expectStatus().isNotFound();
        client.post().uri("/chat/completions").bodyValue("{}").exchange().expectStatus().isNotFound();
        verifyNoInteractions(batch);
        verifyNoInteractions(passthrough);
    }

    @Test
    void dispatcherAnyOtherPathGoesToPassthrough() {
        GenericBatchHandler batch = mock(GenericBatchHandler.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        when(passthrough.forward(any()))
                .thenReturn(ServerResponse.ok().bodyValue("pass"));
        WebTestClient client = WebTestClient.bindToRouterFunction(
                new DispatchRouter(batch, passthrough, List.of(BATCH_INFER)).routes()).build();

        client.get().uri("/dispatcher/v1/models").exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("pass");
        verify(passthrough).forward(any());
        verifyNoInteractions(batch);
    }

    @Test
    void registersOneRouteForEachSpec() {
        GenericBatchHandler batch = mock(GenericBatchHandler.class);
        when(batch.handle(any(), eq(BATCH_INFER)))
                .thenReturn(ServerResponse.ok().bodyValue(BATCH_INFER.getPath()));
        when(batch.handle(any(), eq(EMBEDDINGS)))
                .thenReturn(ServerResponse.ok().bodyValue(EMBEDDINGS.getPath()));
        PassthroughClient passthrough = mock(PassthroughClient.class);
        when(passthrough.forward(any()))
                .thenReturn(ServerResponse.ok().bodyValue("pass"));

        List<BatchEndpointSpec> specs = List.of(BATCH_INFER, EMBEDDINGS);
        RouterFunction<ServerResponse> routes =
                new DispatchRouter(batch, passthrough, specs).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("/batch_infer");
        client.post().uri("/dispatcher/v1/embeddings").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("/v1/embeddings");
        client.post().uri("/dispatcher/v1/models").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("pass");
        client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
    }
}
