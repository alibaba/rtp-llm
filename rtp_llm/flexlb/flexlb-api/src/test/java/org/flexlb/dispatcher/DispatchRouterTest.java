package org.flexlb.dispatcher;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.flexlb.dispatcher.BatchEndpointSpec.FailedItemFactory;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DispatchRouterTest {

    private static final BatchEndpointSpec BATCH_INFER =
            new BatchEndpointSpec("/batch_infer", "prompt_batch", "response_batch",
                    FailedItemFactory.NULL, null, false, false);
    private static final BatchEndpointSpec EMBEDDINGS =
            new BatchEndpointSpec("/v1/embeddings", "input", "data",
                    FailedItemFactory.EMBEDDING_NULL, EmbeddingMerger.INSTANCE, true, true);

    @Test
    void nonDispatcherPathsAreNotMatched() {
        BatchHandler batch = mock(BatchHandler.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        RouterFunction<ServerResponse> routes =
                new DispatchRouter(batch, passthrough, mock(DispatcherInspectionHandler.class), new ActiveRequestCounter(), List.of(BATCH_INFER)).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
        client.get().uri("/health").exchange().expectStatus().isNotFound();
        client.post().uri("/chat/completions").bodyValue("{}").exchange().expectStatus().isNotFound();
        verifyNoInteractions(batch);
        verifyNoInteractions(passthrough);
    }

    @Test
    void dispatcherAnyOtherPathGoesToPassthrough() {
        BatchHandler batch = mock(BatchHandler.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        when(passthrough.forward(any()))
                .thenReturn(ServerResponse.ok().bodyValue("pass"));
        WebTestClient client = WebTestClient.bindToRouterFunction(
                new DispatchRouter(batch, passthrough, mock(DispatcherInspectionHandler.class), new ActiveRequestCounter(), List.of(BATCH_INFER)).routes()).build();

        client.get().uri("/dispatcher/v1/models").exchange()
                .expectStatus().isOk()
                .expectBody(String.class).isEqualTo("pass");
        verify(passthrough).forward(any());
        verifyNoInteractions(batch);
    }

    @Test
    void registersOneRouteForEachSpec() {
        BatchHandler batch = mock(BatchHandler.class);
        when(batch.handle(any(), eq(BATCH_INFER)))
                .thenReturn(ServerResponse.ok().bodyValue(BATCH_INFER.getPath()));
        when(batch.handle(any(), eq(EMBEDDINGS)))
                .thenReturn(ServerResponse.ok().bodyValue(EMBEDDINGS.getPath()));
        PassthroughClient passthrough = mock(PassthroughClient.class);
        when(passthrough.forward(any()))
                .thenReturn(ServerResponse.ok().bodyValue("pass"));

        List<BatchEndpointSpec> specs = List.of(BATCH_INFER, EMBEDDINGS);
        RouterFunction<ServerResponse> routes =
                new DispatchRouter(batch, passthrough, mock(DispatcherInspectionHandler.class), new ActiveRequestCounter(), specs).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("/batch_infer");
        client.post().uri("/dispatcher/v1/embeddings").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("/v1/embeddings");
        client.post().uri("/dispatcher/v1/models").bodyValue("{}").exchange()
                .expectStatus().isOk().expectBody(String.class).isEqualTo("pass");
        client.post().uri("/batch_infer").bodyValue("{}").exchange().expectStatus().isNotFound();
    }

    @Test
    void servingRoutesAreCountedForGracefulDrain() {
        ActiveRequestCounter counter = new ActiveRequestCounter();
        long[] seenInFlight = {-1};
        BatchHandler batch = mock(BatchHandler.class);
        when(batch.handle(any(), eq(BATCH_INFER))).thenReturn(Mono.defer(() -> {
            seenInFlight[0] = counter.getCount();
            return ServerResponse.ok().bodyValue("ok");
        }));
        PassthroughClient passthrough = mock(PassthroughClient.class);
        RouterFunction<ServerResponse> routes =
                new DispatchRouter(batch, passthrough, mock(DispatcherInspectionHandler.class), counter, List.of(BATCH_INFER)).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.post().uri("/dispatcher/batch_infer").bodyValue("{}").exchange().expectStatus().isOk();

        assertEquals(1, seenInFlight[0], "request must be counted while in flight so graceful drain waits for it");
        assertEquals(0, counter.getCount(), "token must be released after the response completes");
    }

    @Test
    void streamingPassthroughStaysCountedUntilResponseBodyCompletes() {
        ActiveRequestCounter counter = new ActiveRequestCounter();
        List<Long> countsWhileBodyStreaming = new CopyOnWriteArrayList<>();
        BatchHandler batch = mock(BatchHandler.class);
        PassthroughClient passthrough = mock(PassthroughClient.class);
        // A passthrough carries the FE body as a Flux subscribed lazily at writeTo time — after the
        // ServerResponse Mono has already completed. The token must survive that gap, otherwise a
        // mid-stream graceful drain would consider this request done and shut down the connection.
        Flux<String> feBody = Flux.just("chunk-a", "chunk-b")
                .doOnNext(chunk -> countsWhileBodyStreaming.add(counter.getCount()));
        when(passthrough.forward(any())).thenReturn(
                ServerResponse.ok().body(BodyInserters.fromPublisher(feBody, String.class)));
        RouterFunction<ServerResponse> routes = new DispatchRouter(
                batch, passthrough, mock(DispatcherInspectionHandler.class), counter, List.of(BATCH_INFER)).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();

        client.get().uri("/dispatcher/v1/models").exchange().expectStatus().isOk();

        assertFalse(countsWhileBodyStreaming.isEmpty(), "response body must have been streamed");
        assertTrue(countsWhileBodyStreaming.stream().allMatch(count -> count == 1L),
                "request must stay counted for graceful drain while its response body is still streaming");
        assertEquals(0, counter.getCount(), "token must be released once the response body completes");
    }
}
