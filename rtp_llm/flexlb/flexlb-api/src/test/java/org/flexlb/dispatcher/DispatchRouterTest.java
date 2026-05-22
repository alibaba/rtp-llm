package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DispatchRouterTest {

    @Test
    void batchInferRoutesToHandleBatch() {
        DispatchHandler handler = mock(DispatchHandler.class);
        when(handler.handleBatch(any())).thenReturn(ServerResponse.ok().bodyValue("BATCH"));
        when(handler.handlePassthrough(any())).thenReturn(ServerResponse.ok().bodyValue("PASS"));

        RouterFunction<ServerResponse> rf = new DispatchRouter(handler).routes();
        WebTestClient client = WebTestClient.bindToRouterFunction(rf).build();

        client.post().uri("/batch_infer").bodyValue("{}")
                .exchange().expectStatus().isOk().expectBody(String.class).isEqualTo("BATCH");
        client.get().uri("/worker_status")
                .exchange().expectStatus().isOk().expectBody(String.class).isEqualTo("PASS");
    }
}
